"""
Let SAM 3 (text-prompted vision-language segmentation) segment leaves BY ITSELF:
no training, no labels - you only give an image and words.

  python vlm_segment.py <image or folder> [more ...] [--prompt "leaf"] [--prompt "open cotton boll"] [--conf 0.3]

For each image, in sample_results/vlm_segment/:
  <name>_overlay.jpg     every segmented object outlined + numbered (colour per prompt)
  <name>_mask.png        black/white mask of all objects
  <name>_objects.csv     one row per object: prompt, score, box, area in pixels
  <name>_leaves/         each object cut out with black background
"""
import argparse
import colorsys
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sam3_autolabel import build_predictor, detect

ROOT = Path(__file__).resolve().parent
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--prompt", action="append", help='what to segment (repeat for several), default "leaf"')
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--out", default=str(ROOT / "sample_results" / "vlm_segment"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    prompts = args.prompt or ["leaf"]
    files = []
    for i in args.inputs:
        p = Path(i)
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pred, _ = build_predictor(ROOT / "sam3.pt", args.conf, args.device)
    cols = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(k / max(1, len(prompts)) * 0.8 + 0.3, 0.9, 1.0))
            for k in range(len(prompts))]
    for f in files:
        img = Image.open(f).convert("RGB")
        dets = detect(pred, img, prompts, args.tile, 0.2)
        ov = img.copy(); d = ImageDraw.Draw(ov, "RGBA")
        mask = Image.new("L", img.size, 0); dm = ImageDraw.Draw(mask)
        cut = out / f"{f.stem}_leaves"; cut.mkdir(exist_ok=True)
        rows = []
        for k, (box, score, cls, poly) in enumerate(dets):
            pts = [tuple(map(float, p)) for p in poly]
            c = cols[cls % len(cols)]
            d.polygon(pts, fill=c + (80,), outline=c + (255,), width=3)
            d.text((float(box[0]) + 3, float(box[1]) + 2), str(k), fill=(255, 255, 0))
            dm.polygon(pts, fill=255)
            one = Image.new("L", img.size, 0); ImageDraw.Draw(one).polygon(pts, fill=255)
            b = [max(0, int(box[0])), max(0, int(box[1])), min(img.width, int(box[2]) + 1), min(img.height, int(box[3]) + 1)]
            piece = Image.composite(img, Image.new("RGB", img.size), one).crop(b)
            piece.save(cut / f"{k:03d}.png")
            rows.append(dict(id=k, prompt=prompts[cls], score=round(score, 3), x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                             area_px=int(np.asarray(one).astype(bool).sum())))
        tag = f"{len(dets)} objects  |  prompts: {', '.join(prompts)}  |  SAM 3, no training"
        d.rectangle([0, 0, img.width, 30], fill=(0, 0, 0, 170)); d.text((8, 8), tag, fill=(255, 255, 255))
        ov.save(out / f"{f.stem}_overlay.jpg", quality=88)
        mask.save(out / f"{f.stem}_mask.png")
        with open(out / f"{f.stem}_objects.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "prompt", "score", "x1", "y1", "x2", "y2", "area_px"])
            w.writeheader(); w.writerows(rows)
        print(f"{f.name}: {len(dets)} objects -> {out / (f.stem + '_overlay.jpg')}")


if __name__ == "__main__":
    main()

