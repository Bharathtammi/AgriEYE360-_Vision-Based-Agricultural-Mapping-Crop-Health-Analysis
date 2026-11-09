"""
Add public cotton leaf datasets to the disease classifier (2 classes: healthy / diseased).

Every image from external_data/extracted/**/<class folder>/ is
  1. shrunk (max side --size px) so the leaf has about the scale the leaf-finder knows
  2. cut out with the leaf-finder (largest leaf, background blacked out) - same look as the field crops
  3. labelled healthy ("Healthy Leaf" folders) or diseased (every other folder)
Output: external_data/crops/{healthy,diseased}/ext_<dataset>_<class>_<file>.png
        external_data/crops/failed.txt   images where no leaf was found (left out)

  python prepare_external.py --limit 10     # quick look (writes external_data/crops_preview.jpg)
  python prepare_external.py                # all images
"""
import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

import pipeline as P

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "external_data"
SRC = EXT / "extracted"
OUT = EXT / "crops"


def label_of(folder):
    return "healthy" if "healthy" in folder.lower() else "diseased"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="per class, for a quick test")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run = P.make_leaf_runner(False, 0.3, args.device)
    files = sorted(p for p in SRC.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    by_cls = {}
    for f in files:
        by_cls.setdefault(f.parent, []).append(f)
    for c in ("healthy", "diseased"):
        (OUT / c).mkdir(parents=True, exist_ok=True)
    failed, done, preview = [], 0, []
    for folder, fs in sorted(by_cls.items()):
        if args.limit:
            fs = fs[:args.limit]
        ds = folder.parent.parent.name.split("_")[0] if folder.parent.parent != SRC else folder.parent.name
        tag = re.sub(r"[^A-Za-z0-9]+", "", folder.name)
        lab = label_of(folder.name)
        ok = 0
        for f in fs:
            name = f"ext_{re.sub(r'[^A-Za-z0-9]+', '', ds)[:12]}_{tag}_{f.stem}.png"
            dst = OUT / lab / name
            if dst.exists():
                ok += 1
                continue
            im = Image.open(f).convert("RGB")
            im.thumbnail((args.size, args.size))
            leaves = P.find_leaves(im, run)
            best = None
            for box, sc, poly in leaves:
                b = [max(0, int(box[0])), max(0, int(box[1])), min(im.width, int(box[2])), min(im.height, int(box[3]))]
                a = (b[2] - b[0]) * (b[3] - b[1])
                if best is None or a > best[0]:
                    best = (a, b, poly)
            # too small, or a thin strip (only part of a leaf): leave out
            if best is None or best[0] < 0.08 * im.width * im.height or \
                    min(best[1][2] - best[1][0], best[1][3] - best[1][1]) < 0.3 * min(im.size):
                failed.append(str(f))
                continue
            crop = P.masked_crop(im, best[2], best[1])
            crop.save(dst)
            ok += 1
            if len(preview) < 60 and args.limit:
                preview.append((im, best[1], crop, f"{tag[:14]} -> {lab}"))
        done += ok
        print(f"{ok:5d}/{len(fs):<5d} {lab:8s} {folder.parent.parent.name[:22]} / {folder.name}")
    (OUT / "failed.txt").write_text("\n".join(failed), encoding="utf-8")
    print(f"\n{done} crops, {len(failed)} images with no leaf found -> {OUT / 'failed.txt'}")
    if preview:
        T = 150
        sheet = Image.new("RGB", (6 * (2 * T + 8), ((len(preview) - 1) // 6 + 1) * (T + 18)), (235, 235, 235))
        for i, (im, b, crop, txt) in enumerate(preview):
            a = im.copy(); ImageDraw.Draw(a).rectangle(b, outline=(255, 220, 0), width=4); a.thumbnail((T, T))
            c = crop.copy(); c.thumbnail((T, T))
            x, y = (i % 6) * (2 * T + 8), (i // 6) * (T + 18)
            sheet.paste(a, (x, y)); sheet.paste(c, (x + T + 2, y))
            ImageDraw.Draw(sheet).text((x + 2, y + T + 2), txt, fill=(0, 0, 0))
        sheet.save(EXT / "crops_preview.jpg", quality=85)
        print(EXT / "crops_preview.jpg")


if __name__ == "__main__":
    main()
