"""
Export the cotton boll boxes as a Roboflow-ready dataset (YOLO detection format, 2 classes).

  python export_roboflow_bolls.py              # all labelled views
  python export_roboflow_bolls.py --checked    # only views where EVERY box was hand-sorted

Labels come from SAM 3 (agrieye_output/objects/sam3_labels.jsonl via boll_crops/manifest.csv), with your
hand sorting (sort_bolls.py) applied on top:
  boll_crops/open/      -> open_boll
  boll_crops/closed/    -> closed_boll
  boll_crops/not_boll/  -> box removed
  not hand-sorted       -> your Roboflow boll model (boll_cls_roboflow.csv), else SAM 3's label ("flower" -> closed_boll)

Output: roboflow_export/1_cotton_boll_detection/
  train/images, train/labels, valid/images, valid/labels   (split by photo, ~80/20, all 4 views of a photo together)
  data.yaml
  box_sources.csv      every box: hand-checked / boll model / SAM-only
  preview.jpg          a few views with the boxes drawn
roboflow_export/1_cotton_boll_detection.zip   <- upload this file to Roboflow
"""
import argparse
import csv
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "agrieye_output"
CROPS = OUT / "boll_crops"
DST = ROOT / "roboflow_export" / "1_cotton_boll_detection"
CLASSES = ["open_boll", "closed_boll"]
HAND = {"open": "open_boll", "closed": "closed_boll", "not_boll": None}
SAM = {"open_boll": "open_boll", "closed_boll": "closed_boll", "flower": "closed_boll"}
MODEL = {"open_boll": "open_boll", "closed_boll": "closed_boll", "not_boll": None}


def shared_valid_ids():
    """Photo ids in the valid split shared by all datasets (make_pov_dataset.py build), or None."""
    f = OUT / "pov_dataset" / "split.csv"
    if not f.exists():
        return None
    return {r["photo_id"] for r in csv.DictReader(open(f, encoding="utf-8")) if r["split"] == "valid"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checked", action="store_true", help="only views where every box was hand-sorted")
    ap.add_argument("--valid", type=float, default=0.2)
    args = ap.parse_args()

    hand = {f.name: lab for lab in HAND for f in (CROPS / lab).glob("*.jpg")}
    mf = OUT / "objects" / "boll_cls_roboflow.csv"
    model = {r["crop"]: r["pred"] for r in csv.DictReader(open(mf, encoding="utf-8"))} if mf.exists() else {}
    views = defaultdict(list)
    for r in csv.DictReader(open(CROPS / "manifest.csv", encoding="utf-8")):
        views[r["pov"]].append(r)

    if DST.exists():
        shutil.rmtree(DST)
    for s in ("train", "valid", "test"):
        (DST / s / "images").mkdir(parents=True)
        (DST / s / "labels").mkdir(parents=True)

    from make_roboflow_datasets import SPLIT, split_map
    sp = split_map() if SPLIT.exists() else None
    valid_ids = shared_valid_ids()
    if valid_ids is None:
        photos = sorted({rs[0]["photo_id"] for rs in views.values()})
        random.Random(0).shuffle(photos)
        valid_ids = set(photos[:max(1, round(len(photos) * args.valid))])

    src_rows, n = [], defaultdict(int)
    kept = []
    for pov, rs in sorted(views.items()):
        p = Path(pov)
        if not p.exists():
            print(f"(missing view, skipped: {p})")
            continue
        if args.checked and any(r["crop"] not in hand for r in rs):
            continue
        im = Image.open(p)
        W, H = im.size
        lines, boxes = [], []
        for r in rs:
            if r["crop"] in hand:
                cls, source = HAND[hand[r["crop"]]], "hand-checked"
            elif r["crop"] in model and model[r["crop"]] in MODEL:
                cls, source = MODEL[model[r["crop"]]], "boll model"
            else:
                cls, source = SAM[r["sam_label"]], "SAM 3 only"
            src_rows.append(dict(image=p.name, crop=r["crop"], sam_label=r["sam_label"],
                                 final_label=cls or "removed (not a boll)", source=source))
            if cls is None:
                n["removed"] += 1
                continue
            x1, y1 = max(0.0, float(r["x1"])), max(0.0, float(r["y1"]))
            x2, y2 = min(W, float(r["x2"])), min(H, float(r["y2"]))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            c = CLASSES.index(cls)
            lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            boxes.append((c, x1, y1, x2, y2, source))
            n[cls] += 1
            n[source] += 1
        split = sp.get((rs[0]["photo_id"], rs[0]["direction"]), "train") if sp else ("valid" if rs[0]["photo_id"] in valid_ids else "train")
        name = f"{rs[0]['station']}_{rs[0]['date']}_{p.stem}"
        shutil.copy2(p, DST / split / "images" / f"{name}.jpg")
        (DST / split / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n[f"{split} images"] += 1
        kept.append((p, boxes))

    (DST / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n", encoding="utf-8")
    with open(DST / "box_sources.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(src_rows[0].keys())); w.writeheader(); w.writerows(src_rows)

    # preview: 6 views with the most boxes
    T = 480
    pick = sorted(kept, key=lambda k: -len(k[1]))[:6]
    sheet = Image.new("RGB", (3 * T, 2 * T), (30, 30, 30))
    for i, (p, boxes) in enumerate(pick):
        im = Image.open(p).convert("RGB"); d = ImageDraw.Draw(im)
        for c, x1, y1, x2, y2, src in boxes:
            col = (255, 255, 255) if c == 0 else (255, 80, 200)
            d.rectangle([x1, y1, x2, y2], outline=col, width=4 if src == "hand-checked" else 2)
        d.rectangle([0, 0, im.width, 40], fill=(0, 0, 0)); d.text((10, 12), f"{p.stem}  white=open  pink=closed  thick=hand-checked", fill=(255, 255, 255))
        im.thumbnail((T, T)); sheet.paste(im, ((i % 3) * T, (i // 3) * T))
    sheet.save(DST / "preview.jpg", quality=88)

    zp = DST.parent / f"{DST.name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in DST.rglob("*"):
            if f.is_file() and f.name not in ("preview.jpg", "box_sources.csv"):
                z.write(f, f.relative_to(DST))
    print(f"images: train {n['train images']}, valid {n['valid images']}, test {n['test images']}")
    print(f"boxes: open_boll {n['open_boll']}, closed_boll {n['closed_boll']}  (removed as not a boll: {n['removed']})")
    print(f"hand-checked boxes {n['hand-checked']}, boll model {n['boll model']}, SAM 3 only {n['SAM 3 only']}")
    print(f"Folder: {DST}\nUpload: {zp}")


if __name__ == "__main__":
    main()
