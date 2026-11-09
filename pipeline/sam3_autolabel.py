"""
Step 1 of the cotton disease pipeline: auto-segment cotton leaves with SAM 3.

For every image it finds, SAM 3 (text-prompted) segments each leaf and this
script saves:
  out/crops/unsorted/<image-id>__<n>.png   leaf crop, background blacked out
  out/previews/<image-id>.jpg               overlay for quick visual QC
  out/manifest.jsonl                        one JSON line per crop (source image,
                                            box, normalized polygon, score)

Afterwards, sort the crops from crops/unsorted/ into crops/healthy/ and
crops/diseased/ (move the files), then run build_disease_dataset.py.

The script is resumable: images already listed in manifest.jsonl are skipped.

Usage:
  python sam3_autolabel.py --limit 5                      # quick test
  python sam3_autolabel.py                                # everything
  python sam3_autolabel.py --prompts "cotton leaf" "leaf" --conf 0.3

Requires: pip install -U ultralytics   (SAM 3 support: ultralytics >= 8.3.237)
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
# High-res 360 panoramas: best leaf detail. (The 800x600 POV views are low quality.)
DEFAULT_SOURCES = [ROOT / "AgriEye 360 field data" / "Data set"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


SKIP_DIRS = {".venv", "venv", "site-packages", "Lib", "__pycache__", ".git"}
MIN_SIDE = 256  # ignore icons / thumbnails


def find_images(sources):
    for src in sources:
        for p in sorted(Path(src).rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTS or "_annotated" in p.stem:
                continue
            if SKIP_DIRS.intersection(p.parts):
                continue
            try:
                with Image.open(p) as im:
                    if min(im.size) < MIN_SIDE:
                        continue
            except Exception:
                continue
            yield p


def make_tiles(W, H, tile, overlap):
    """Top-left corners of overlapping tiles covering a W x H image.
    Images not much larger than one tile are processed whole."""
    if max(W, H) <= tile * 1.5:
        return [(0, 0, W, H)]
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    return [(x, y, min(x + tile, W), min(y + tile, H)) for y in ys for x in xs]


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    # IoU, or how much of the smaller box is covered (catches a leaf
    # cut in half at a tile edge overlapping its full version).
    return max(inter / (area(a) + area(b) - inter), inter / min(area(a), area(b)))


def image_id(path):
    """Stable, filesystem-safe id. Folder names repeat (front.png, left.png ...)
    so include a short hash of the full path."""
    h = hashlib.md5(str(path).encode()).hexdigest()[:8]
    return f"{path.parent.name}_{path.stem}_{h}".replace(" ", "_")


def content_hash(path):
    """Hash of the decoded pixels (small grayscale thumbnail), so copies that
    differ only in file metadata/EXIF are still recognised as duplicates."""
    with Image.open(path) as im:
        im.draft("L", (256, 128))  # fast reduced-size JPEG decode
        small = im.convert("L").resize((64, 32))
    return hashlib.md5((np.asarray(small) // 16).tobytes()).hexdigest()


def load_done(manifest):
    done = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["source"])
    return done


def build_predictor(model_path, conf, device):
    from ultralytics.models.sam import SAM3SemanticPredictor

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=str(model_path),
        device=device,
        imgsz=644,  # SAM 3 needs a multiple of 14; avoids a warning per tile
        save=False,
        verbose=False,
    )
    print(f"Loading SAM 3 on {device} ...")
    return SAM3SemanticPredictor(overrides=overrides), device


def detect(predictor, img, prompts, tile, overlap):
    """Run SAM 3 over the image (tiled if large). Returns detections as
    (box_xyxy, score, cls, polygon_px) in full-image pixel coordinates."""
    W, H = img.size
    dets = []
    for x0, y0, x1, y1 in make_tiles(W, H, tile, overlap):
        crop = np.ascontiguousarray(np.asarray(img.crop((x0, y0, x1, y1)))[:, :, ::-1])  # RGB -> BGR
        predictor.set_image(crop)
        results = predictor(text=prompts)
        res = results[0] if isinstance(results, (list, tuple)) else results
        if res.masks is None or len(res.masks) == 0:
            continue
        for box, score, cls, poly in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int),
            res.masks.xy,
        ):
            if len(poly) >= 3:
                dets.append((box + [x0, y0, x0, y0], float(score), int(cls), poly + [x0, y0]))

    # Merge duplicates from overlapping tiles: keep the highest-scoring one.
    dets.sort(key=lambda d: -d[1])
    kept = []
    for d in dets:
        if all(box_iou(d[0], k[0]) < 0.6 for k in kept):
            kept.append(d)
    return kept


def process_image(predictor, path, prompts, args, out):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    records = []

    iid = image_id(path)
    min_area = args.min_px ** 2
    n = 0
    for box, score, cls, poly in detect(predictor, img, prompts, args.tile, args.overlap):
        polyn = poly / [W, H]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if (x2 - x1) * (y2 - y1) < min_area:
            continue

        # Mask out the background so the crop shows only this leaf.
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).polygon([tuple(p) for p in poly], fill=255)
        leaf = Image.new("RGB", (W, H))
        leaf.paste(img, mask=mask)
        crop = leaf.crop((x1, y1, x2, y2))

        name = f"{iid}__{n:03d}.png"
        crop.save(out / "crops" / "unsorted" / name)
        records.append(
            dict(
                crop=name,
                source=str(path),
                width=W,
                height=H,
                prompt=prompts[cls] if 0 <= cls < len(prompts) else str(cls),
                score=round(float(score), 4),
                box=[x1, y1, x2, y2],
                polygon=np.round(polyn, 5).tolist(),
            )
        )
        n += 1

    return img, records


def save_preview(img, records, path):
    # Downscale first, then draw, so outlines stay visible on huge panoramas.
    base = img.copy()
    base.thumbnail((2048, 2048))
    W, H = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for r in records:
        pts = [(x * W, y * H) for x, y in r["polygon"]]
        draw.polygon(pts, fill=(0, 255, 0, 90), outline=(0, 255, 0, 255), width=2)
        draw.text((pts[0][0], pts[0][1]), r["crop"].split("__")[-1][:3], fill=(255, 255, 0, 255))
    Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB").save(path, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", type=Path, default=DEFAULT_SOURCES)
    ap.add_argument("--model", type=Path, default=ROOT / "sam3.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "disease_autolabel")
    ap.add_argument("--prompts", nargs="+", default=["cotton leaf", "leaf"])
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--min-px", type=int, default=40,
                    help="drop leaves whose box is smaller than about N x N pixels (default 40)")
    ap.add_argument("--tile", type=int, default=1280,
                    help="tile size in px for large images (panoramas); smaller = finds smaller leaves, slower")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0, help="process at most N new images (0 = all)")
    ap.add_argument("--device", default=None, help="cpu, cuda, cuda:0 ... (default: auto)")
    args = ap.parse_args()

    out = args.out
    (out / "crops" / "unsorted").mkdir(parents=True, exist_ok=True)
    (out / "crops" / "healthy").mkdir(parents=True, exist_ok=True)
    (out / "crops" / "diseased").mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done = load_done(manifest)

    # Skip duplicate photos (the dataset folders overlap heavily).
    seen, todo, dups = set(), [], 0
    for p in find_images(args.sources):
        h = content_hash(p)
        if h in seen:
            dups += 1
            continue
        seen.add(h)
        if str(p) not in done:
            todo.append(p)
    if dups:
        print(f"Skipped {dups} duplicate files.")
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} images already done, {len(todo)} to process. Prompts: {args.prompts}")
    if not todo:
        return

    predictor, _ = build_predictor(args.model, args.conf, args.device)

    total_crops = 0
    t0 = time.time()
    with manifest.open("a", encoding="utf-8") as mf:
        for i, path in enumerate(todo, 1):
            t = time.time()
            try:
                img, records = process_image(predictor, path, args.prompts, args, out)
            except Exception as e:  # keep going on a bad image
                print(f"[{i}/{len(todo)}] FAILED {path}: {e}")
                continue
            save_preview(img, records, out / "previews" / f"{image_id(path)}.jpg")
            if not records:
                # Record images with no leaves so they are not re-run.
                records = [dict(crop=None, source=str(path))]
            for r in records:
                mf.write(json.dumps(r) + "\n")
            mf.flush()
            n = sum(1 for r in records if r["crop"])
            total_crops += n
            eta = (time.time() - t0) / i * (len(todo) - i) / 60
            print(f"[{i}/{len(todo)}] {n:3d} leaves  {time.time() - t:5.1f}s  ETA {eta:6.1f} min  {path.name}")

    print(f"\nDone. {total_crops} crops in {out / 'crops' / 'unsorted'}")
    print("Next: move crops into crops/healthy/ and crops/diseased/, then run build_disease_dataset.py")


if __name__ == "__main__":
    main()
