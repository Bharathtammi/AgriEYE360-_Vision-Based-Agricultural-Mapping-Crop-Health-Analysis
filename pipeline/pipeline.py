"""
Cotton leaf disease pipeline: one script, one command per stage.

  status      show where everything stands (crop counts, models, datasets)
  presort     score unsorted crops by colour; move likely-diseased / likely-healthy
              ones into review folders so manual sorting starts with the best candidates
  leafdata    build a YOLO-seg "leaf" dataset from SAM 3's outlines (no manual work)
  trainleaf   train a fast YOLO leaf segmenter on it (replaces slow SAM 3 at run time)
  traincls    build the healthy/diseased dataset from your sorted crops and train a classifier
  prelabel    let the classifier sort the remaining unsorted crops for you to check
  predict     photo(s) -> find leaves -> classify each -> annotated image + CSV + summary

Typical order:
  presort -> (you sort ~300+300) -> traincls -> prelabel -> (you fix) -> traincls again
  leafdata -> trainleaf          (can run any time, independent of sorting)
  predict                        (needs a classifier; uses the leaf model, or SAM 3 if none)

Run with the venv Python, e.g.
  "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" "F:\\Project and completed courses\\Agri EYE 360\\pipeline.py" status
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sam3_autolabel import box_iou, image_id, make_tiles  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

WORK = ROOT / "disease_autolabel"
CROPS = WORK / "crops"
MANIFEST = WORK / "manifest.jsonl"
MODELS = WORK / "models"
LEAF_DATA = WORK / "dataset_leaf"
LEAF_MODEL = MODELS / "leaf_seg" / "weights" / "best.pt"
CLS_MODEL = MODELS / "disease_cls" / "weights" / "best.pt"
TILE, OVERLAP, IMGSZ = 1280, 0.2, 640  # same scale SAM 3 saw (1280 tile -> 644)
CLS_IMGSZ = 224

SORT_DIRS = ["unsorted", "review_likely_diseased", "review_likely_healthy",
             "auto_diseased", "auto_healthy", "healthy", "diseased"]


# ----------------------------------------------------------------- helpers
def read_manifest():
    recs = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("crop"):
                recs.append(r)
    return recs


def masked_crop(img, poly_px, box):
    """Leaf crop with background blacked out (same as sam3_autolabel)."""
    x1, y1, x2, y2 = box
    m = Image.new("L", (x2 - x1, y2 - y1), 0)
    ImageDraw.Draw(m).polygon([(x - x1, y - y1) for x, y in poly_px], fill=255)
    out = Image.new("RGB", m.size)
    out.paste(img.crop((x1, y1, x2, y2)), mask=m)
    return out


def merge_dets(dets, thr=0.6):
    dets = sorted(dets, key=lambda d: -d[1])
    kept = []
    for d in dets:
        if all(box_iou(d[0], k[0]) < thr for k in kept):
            kept.append(d)
    return kept


def move_unique(src, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst_dir / src.name))


def ultra():
    from ultralytics import YOLO
    return YOLO


# ------------------------------------------------------------------ status
def cmd_status(args):
    print("Crops:")
    for d in SORT_DIRS:
        p = CROPS / d
        if p.exists():
            print(f"  {len(list(p.glob('*.png'))):6d}  {p}")
    for name, p in [("Leaf model", LEAF_MODEL), ("Disease classifier", CLS_MODEL)]:
        print(f"{name:20s}: {p if p.exists() else 'not trained yet'}")
    if (LEAF_DATA / "data.yaml").exists():
        n = {s: len(list((LEAF_DATA / "images" / s).glob("*.jpg"))) for s in ("train", "val")}
        print(f"Leaf dataset        : {n} tiles in {LEAF_DATA}")


# ----------------------------------------------------------------- presort
def disease_score(path):
    """Cheap colour heuristic. Higher = more likely diseased.
    brown/yellow fraction + pale-speck fraction, on leaf pixels only."""
    im = Image.open(path).convert("RGB")
    if min(im.size) < 40:                # edge-on slivers: can't judge disease
        return None, 0
    im.thumbnail((256, 256))
    rgb = np.asarray(im)
    leaf = rgb.max(axis=2) > 12          # background is pure black
    # shrink the leaf mask so the cut-out edge doesn't count as spots
    inner = np.asarray(Image.fromarray(leaf.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(7))) > 0
    n = int(inner.sum())
    if n < 600:
        return None, n
    hsv = np.asarray(im.convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # PIL hue 0-255: 8-30 = orange/brown (~11-42 deg). Sunlit yellow-green (~40-55)
    # and purple/red variety colour (<8, >200) are excluded.
    brown = inner & (h >= 8) & (h <= 30) & (s > 80) & (v > 40)
    lum = np.asarray(im.convert("L")).astype(int)
    vmed = np.asarray(im.convert("L").filter(ImageFilter.MedianFilter(11))).astype(int)
    pale = inner & (lum - vmed > 30) & (s < 110)      # pale specks
    dark = inner & (vmed - lum > 30)                  # dark spots / lesions
    score = 1.5 * brown.sum() / n + 4.0 * pale.sum() / n + 3.0 * dark.sum() / n
    return float(score), n


def cmd_presort(args):
    src = CROPS / "unsorted"
    files = sorted(src.glob("*.png"))
    print(f"Scoring {len(files)} crops ...")
    scored = []
    t = time.time()
    for i, f in enumerate(files, 1):
        sc, n = disease_score(f)
        if sc is not None:
            scored.append((sc, f))
        if i % 2000 == 0:
            print(f"  {i}/{len(files)}  ({time.time() - t:.0f}s)")
    scored.sort(key=lambda x: -x[0])
    top = [f for _, f in scored[: args.n]]
    # healthy candidates: lowest scores, but sample across the low half for variety
    low = [f for _, f in scored[len(scored) // 2:]]
    random.Random(0).shuffle(low)
    bottom = low[: args.n]
    with (WORK / "presort_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["crop", "score"])
        w.writerows((f.name, f"{sc:.4f}") for sc, f in scored)
    for f in top:
        move_unique(f, CROPS / "review_likely_diseased")
    for f in bottom:
        move_unique(f, CROPS / "review_likely_healthy")
    print(f"\nMoved {len(top)} crops to  {CROPS / 'review_likely_diseased'}")
    print(f"Moved {len(bottom)} crops to {CROPS / 'review_likely_healthy'}")
    print("Check them and move each into crops\\diseased or crops\\healthy (delete non-leaves).")
    print(f"Scores saved to {WORK / 'presort_scores.csv'}")


# ---------------------------------------------------------------- leafdata
def cmd_leafdata(args):
    import cv2

    recs = read_manifest()
    by_src = defaultdict(list)
    for r in recs:
        by_src[r["source"]].append(r)
    srcs = sorted(by_src)
    random.Random(0).shuffle(srcs)
    n_val = max(1, round(len(srcs) * 0.15))
    val = set(srcs[:n_val])

    shutil.rmtree(LEAF_DATA, ignore_errors=True)
    for s in ("train", "val"):
        (LEAF_DATA / "images" / s).mkdir(parents=True)
        (LEAF_DATA / "labels" / s).mkdir(parents=True)

    n_tiles = defaultdict(int)
    n_inst = 0
    for k, src in enumerate(srcs, 1):
        split = "val" if src in val else "train"
        img = Image.open(src).convert("RGB")
        W, H = img.size
        leaves = [(np.array(r["polygon"]) * [W, H], r["box"]) for r in by_src[src]]
        for x0, y0, x1, y1 in make_tiles(W, H, TILE, OVERLAP):
            tw, th = x1 - x0, y1 - y0
            lines = []
            for poly, (bx1, by1, bx2, by2) in leaves:
                if bx2 <= x0 or bx1 >= x1 or by2 <= y0 or by1 >= y1:
                    continue
                # clip leaf to the tile by rasterising, then re-trace its outline
                m = np.zeros((th, tw), np.uint8)
                cv2.fillPoly(m, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cs:
                    continue
                c = max(cs, key=cv2.contourArea)
                if cv2.contourArea(c) < 60 or len(c) < 3:
                    continue
                c = c.reshape(-1, 2) / [tw, th]
                lines.append("0 " + " ".join(f"{x:.5f} {y:.5f}" for x, y in c))
            name = f"{image_id(Path(src))}_{x0}_{y0}"
            img.crop((x0, y0, x1, y1)).save(LEAF_DATA / "images" / split / f"{name}.jpg", quality=92)
            (LEAF_DATA / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            n_tiles[split] += 1
            n_inst += len(lines)
        print(f"[{k}/{len(srcs)}] {Path(src).name}  ({split})")

    (LEAF_DATA / "data.yaml").write_text(
        f"path: {LEAF_DATA.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['leaf']\n")
    print(f"\nLeaf dataset: {dict(n_tiles)} tiles, {n_inst} leaf outlines -> {LEAF_DATA}")


def cmd_trainleaf(args):
    if not (LEAF_DATA / "data.yaml").exists():
        raise SystemExit("Run 'leafdata' first.")
    import train_guard
    clear_new("leaf_seg")
    model = ultra()(args.base)
    model.train(data=str(LEAF_DATA / "data.yaml"), epochs=args.epochs, imgsz=IMGSZ,
                batch=args.batch, device=args.device, workers=2, patience=15,
                project=str(MODELS), name="leaf_seg_new", exist_ok=True, plots=True)
    # new vs current on the same validation tiles; the better one becomes models/leaf_seg + graphs redrawn
    train_guard.decide("leaf_seg", "seg", LEAF_DATA / "data.yaml", IMGSZ)
    print(f"\nLeaf model: {LEAF_MODEL}")


def clear_new(name):
    """A '<name>_new' folder left by an interrupted run is set aside, never mixed into the new run."""
    left = MODELS / f"{name}_new"
    if left.exists():
        from datetime import datetime
        shutil.move(str(left), str(MODELS / f"{name}_interrupted_{datetime.now():%Y%m%d_%H%M}"))


# ---------------------------------------------------------------- traincls
def cmd_traincls(args):
    counts = {c: len(list((CROPS / c).glob("*.png"))) for c in ("healthy", "diseased")}
    print("Sorted crops:", counts)
    if min(counts.values()) < 20:
        raise SystemExit("Need at least ~20 crops in each of crops\\healthy and crops\\diseased "
                         "(300+ each recommended).")
    subprocess.run([sys.executable, str(ROOT / "build_disease_dataset.py")], check=True)
    import train_guard
    clear_new("disease_cls")
    model = ultra()(args.base)
    model.train(data=str(WORK / "dataset_cls"), epochs=args.epochs, imgsz=CLS_IMGSZ,
                batch=args.batch, device=args.device, workers=2, patience=10,
                project=str(MODELS), name="disease_cls_new", exist_ok=True,
                save_period=1)  # keep every epoch's weights -> per-epoch charts (model_metrics.py)
    # new vs current on the same (freshly built) validation leaves; the better one becomes models/disease_cls
    train_guard.decide("disease_cls", "cls", WORK / "dataset_cls", CLS_IMGSZ)
    print(f"\nClassifier: {CLS_MODEL}")


# ---------------------------------------------------------------- classify
# Scores in this band are "needs checking": on the test set the model is 95% right outside
# it and only ~72% right inside it. Labels stay diseased/healthy (threshold 0.5); the band
# is reported alongside so reports can flag leaves for a human look.
CHECK_BAND = (0.2, 0.8)


def needs_check(p):
    return CHECK_BAND[0] <= p < CHECK_BAND[1]


def classify(model, images, batch=64):
    """Returns list of (label, p_diseased)."""
    names = model.names
    d_idx = [k for k, v in names.items() if v == "diseased"][0]
    out = []
    for i in range(0, len(images), batch):
        for r in model.predict(images[i:i + batch], imgsz=CLS_IMGSZ, verbose=False):
            p = float(r.probs.data[d_idx])
            out.append(("diseased" if p >= 0.5 else "healthy", p))
    return out


def cmd_prelabel(args):
    if not CLS_MODEL.exists():
        raise SystemExit("Train the classifier first (traincls).")
    model = ultra()(str(CLS_MODEL))
    files = sorted((CROPS / "unsorted").glob("*.png"))
    print(f"Classifying {len(files)} unsorted crops ...")
    rows = []
    for i in range(0, len(files), 256):
        chunk = files[i:i + 256]
        imgs = [Image.open(f).convert("RGB") for f in chunk]
        for f, (lab, p) in zip(chunk, classify(model, imgs)):
            rows.append((f, lab, p))
        print(f"  {min(i + 256, len(files))}/{len(files)}")
    with (WORK / "prelabel_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["crop", "label", "p_diseased"])
        w.writerows((f.name, lab, f"{p:.4f}") for f, lab, p in rows)
    moved = defaultdict(int)
    for f, lab, p in rows:
        move_unique(f, CROPS / f"auto_{lab}")
        moved[lab] += 1
    print(f"\nMoved: {dict(moved)}")
    print(f"Review {CROPS / 'auto_diseased'} and {CROPS / 'auto_healthy'}; move correct ones into "
          "crops\\diseased / crops\\healthy, then run traincls again.")
    print(f"Probabilities: {WORK / 'prelabel_scores.csv'} (values near 0.5 are the least certain)")


# ----------------------------------------------------------------- predict
def make_leaf_runner(use_sam3, conf, device):
    """Load the leaf finder once; returns fn(BGR tile array) -> ultralytics Results."""
    if use_sam3:
        from sam3_autolabel import build_predictor
        pred, _ = build_predictor(ROOT / "sam3.pt", conf, device)

        def run(arr):
            pred.set_image(arr)
            return pred(text=["cotton leaf", "leaf"])
        return run
    model = ultra()(str(LEAF_MODEL))
    return lambda arr: model.predict(arr, imgsz=IMGSZ, conf=conf, device=device,
                                     retina_masks=True, verbose=False)


def find_leaves(img, run):
    """Returns merged list of (box, score, poly_px) for the whole image."""
    W, H = img.size
    dets = []
    for x0, y0, x1, y1 in make_tiles(W, H, TILE, OVERLAP):
        arr = np.ascontiguousarray(np.asarray(img.crop((x0, y0, x1, y1)))[:, :, ::-1])
        res = run(arr)
        res = res[0] if isinstance(res, (list, tuple)) else res
        if res is None or res.masks is None or len(res.masks) == 0:
            continue
        for box, score, poly in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                    res.masks.xy):
            if len(poly) >= 3:
                dets.append((box + [x0, y0, x0, y0], float(score), poly + [x0, y0]))
    return merge_dets(dets)


def cmd_predict(args):
    if not CLS_MODEL.exists():
        raise SystemExit("Train the classifier first (traincls).")
    use_sam3 = args.sam3 or not LEAF_MODEL.exists()
    if use_sam3 and not args.sam3:
        print("No leaf model yet - using SAM 3 to find leaves (slow on CPU). "
              "Run leafdata + trainleaf for fast predictions.")
    cls_model = ultra()(str(CLS_MODEL))
    leaf_run = make_leaf_runner(use_sam3, args.conf, args.device)
    inputs = []
    for p in args.images:
        p = Path(p)
        inputs += sorted(q for q in p.rglob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
            if p.is_dir() else [p]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for k, path in enumerate(inputs, 1):
        t = time.time()
        img = Image.open(path).convert("RGB")
        W, H = img.size
        leaves = []
        for box, sc, poly in find_leaves(img, leaf_run):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if (x2 - x1) * (y2 - y1) < args.min_px ** 2:
                continue
            leaves.append(((x1, y1, x2, y2), sc, poly))
        crops = [masked_crop(img, poly, box) for box, _, poly in leaves]
        labels = classify(cls_model, crops) if crops else []

        # annotated image
        base = img.copy()
        base.thumbnail((2048, 2048))
        s = base.size[0] / W
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for (box, _, poly), (lab, p) in zip(leaves, labels):
            # red = diseased, green = healthy, yellow outline = needs checking (unsure)
            col = (255, 40, 40) if lab == "diseased" else (40, 220, 40)
            edge = (255, 220, 0) if needs_check(p) else col
            dr.polygon([(x * s, y * s) for x, y in poly], fill=col + (90,), outline=edge + (255,),
                       width=4 if needs_check(p) else 2)
        n_d = sum(lab == "diseased" for lab, _ in labels)
        n_c = sum(needs_check(p) for _, p in labels)
        n = len(labels)
        pct = 100 * n_d / n if n else 0.0
        dr.rectangle((0, 0, 560, 34), fill=(0, 0, 0, 170))
        dr.text((8, 10), f"leaves {n}   diseased {n_d} ({pct:.1f}%)   needs checking {n_c}",
                fill=(255, 255, 255, 255))
        stem = f"{path.stem}_{abs(hash(str(path))) % 10**6:06d}"
        Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB").save(out / f"{stem}_result.jpg", quality=88)

        with (out / f"{stem}_leaves.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["leaf", "x1", "y1", "x2", "y2", "leaf_score", "label", "p_diseased", "needs_check"])
            for i, ((box, sc, _), (lab, p)) in enumerate(zip(leaves, labels)):
                w.writerow([i, *box, f"{sc:.3f}", lab, f"{p:.3f}", int(needs_check(p))])
        summary.append(dict(image=str(path), leaves=n, diseased=n_d, pct_diseased=round(pct, 2), needs_check=n_c))
        print(f"[{k}/{len(inputs)}] {path.name}: {n} leaves, {n_d} diseased ({pct:.1f}%), "
              f"{n_c} need checking  {time.time() - t:.0f}s")

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "leaves", "diseased", "pct_diseased", "needs_check"])
        w.writeheader()
        w.writerows(summary)
    print(f"\nResults in {out.resolve()}  (summary.csv, *_result.jpg, *_leaves.csv)")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)

    p = sub.add_parser("presort")
    p.add_argument("--n", type=int, default=600, help="crops to pull into each review folder")
    p.set_defaults(fn=cmd_presort)

    sub.add_parser("leafdata").set_defaults(fn=cmd_leafdata)

    p = sub.add_parser("trainleaf")
    p.add_argument("--base", default="yolo11n-seg.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_trainleaf)

    p = sub.add_parser("traincls")
    p.add_argument("--base", default="yolo11n-cls.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_traincls)

    sub.add_parser("prelabel").set_defaults(fn=cmd_prelabel)

    p = sub.add_parser("predict")
    p.add_argument("images", nargs="+", help="image files or folders")
    p.add_argument("--out", default=str(WORK / "predictions"))
    p.add_argument("--sam3", action="store_true", help="find leaves with SAM 3 instead of the leaf model")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--min-px", type=int, default=40)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_predict)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
