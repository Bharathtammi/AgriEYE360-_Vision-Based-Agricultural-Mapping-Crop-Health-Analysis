"""
Review the hand-sorted leaf crops (no AI model needed - works without torch).

  python review_leaf_sorting.py [--n 400]

Writes disease_autolabel/check_sheets/review_*.jpg:
  review_by_visit.jpg          per visit: 12 random diseased + 12 random healthy (consistency check)
  review_random_diseased.jpg   60 random diseased crops
  review_random_healthy.jpg    60 random healthy crops
  review_spotty_healthy.jpg    healthy crops NOT yet strictly reviewed, most "spotty/brown" first
and disease_autolabel/review_healthy.csv (top --n spotty healthy crops) so you can re-check them with
  python sort_crops.py --review healthy        (H = keep healthy, D = diseased, X = not usable)
The previous review_healthy.csv is kept as review_healthy_round1.csv.
"""
import argparse
import csv
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

import pipeline as P

W = P.WORK
SHEETS = W / "check_sheets"
T = 120


def spot_score(path):
    """Spots/specks only (pale specks + dark lesions vs. the local leaf colour).
    Unlike pipeline.disease_score it ignores overall brown/orange colour, which is
    mostly evening light or ageing leaves, not disease."""
    import numpy as np
    from PIL import ImageFilter
    im = Image.open(path).convert("RGB")
    if min(im.size) < 40:
        return None
    im.thumbnail((256, 256))
    leaf = np.asarray(im).max(axis=2) > 12
    inner = np.asarray(Image.fromarray(leaf.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(9))) > 0
    n = int(inner.sum())
    if n < 600:
        return None
    g = im.convert("L")
    lum = np.asarray(g).astype(int)
    med = np.asarray(g.filter(ImageFilter.MedianFilter(11))).astype(int)
    s = np.asarray(im.convert("HSV"))[..., 1].astype(int)
    pale = inner & (lum - med > 30) & (s < 110)
    dark = inner & (med - lum > 35)
    return float(4.0 * pale.sum() / n + 3.0 * dark.sum() / n)


def tile(path, text):
    im = Image.open(path).convert("RGB")
    im.thumbnail((T, T))
    t = Image.new("RGB", (T, T + 14), (255, 255, 255))
    t.paste(im, ((T - im.width) // 2, (T - im.height) // 2))
    ImageDraw.Draw(t).text((3, T + 1), text, fill=(0, 0, 0))
    return t


def sheet(groups, path, per_row=12):
    """groups: list of (title, [(path, text)])"""
    rows = []
    for title, items in groups:
        bar = Image.new("RGB", (per_row * (T + 4), 22), (40, 40, 40))
        ImageDraw.Draw(bar).text((6, 5), title, fill=(255, 255, 255))
        rows.append(bar)
        for i in range(0, len(items), per_row):
            r = Image.new("RGB", (per_row * (T + 4), T + 18), (235, 235, 235))
            for k, (p, txt) in enumerate(items[i:i + per_row]):
                r.paste(tile(p, txt), (k * (T + 4) + 2, 2))
            rows.append(r)
    out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (235, 235, 235))
    y = 0
    for r in rows:
        out.paste(r, (0, y)); y += r.height
    out.save(path, quality=88)
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="spotty healthy crops to list for re-checking")
    args = ap.parse_args()
    rng = random.Random(7)
    src = {}
    for l in P.MANIFEST.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("crop"):
                src[r["crop"]] = r["source"]

    def visit(name):
        m = re.search(r"(20\d{2})(\d{2})(\d{2})_\d{6}", Path(src.get(name, "")).name)
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "older 360 set"

    dis = sorted((P.CROPS / "diseased").glob("*.png"))
    hea = sorted((P.CROPS / "healthy").glob("*.png"))
    by = defaultdict(lambda: {"d": [], "h": []})
    for p in dis:
        by[visit(p.name)]["d"].append(p)
    for p in hea:
        by[visit(p.name)]["h"].append(p)

    SHEETS.mkdir(parents=True, exist_ok=True)
    groups = []
    for v in sorted(by):
        d, h = by[v]["d"], by[v]["h"]
        pct = 100 * len(d) / max(1, len(d) + len(h))
        groups.append((f"{v}: DISEASED  ({len(d)} crops, {pct:.0f}% of leaves)  - 12 random",
                       [(p, "diseased") for p in rng.sample(d, min(12, len(d)))]))
        groups.append((f"{v}: HEALTHY  ({len(h)} crops)  - 12 random",
                       [(p, "healthy") for p in rng.sample(h, min(12, len(h)))]))
    sheet(groups, SHEETS / "review_by_visit.jpg")
    sheet([("DISEASED - 60 random", [(p, visit(p.name)) for p in rng.sample(dis, 60)])],
          SHEETS / "review_random_diseased.jpg")
    sheet([("HEALTHY - 60 random", [(p, visit(p.name)) for p in rng.sample(hea, 60)])],
          SHEETS / "review_random_healthy.jpg")

    ok_h = set((W / "reviewed_ok_healthy.txt").read_text(encoding="utf-8").split()) \
        if (W / "reviewed_ok_healthy.txt").exists() else set()
    todo = [p for p in hea if p.name not in ok_h]
    print(f"Scoring {len(todo)} healthy crops not yet strictly reviewed ...")
    scored = [(spot_score(p), p) for p in todo]
    scored = sorted(((s, p) for s, p in scored if s is not None), key=lambda x: -x[0])   # None = too small to judge
    sheet([(f"HEALTHY, not yet reviewed - most spotty/brown first (top 96 of {len(todo)})",
            [(p, f"{s:.3f} {visit(p.name)[5:]}") for s, p in scored[:96]])],
          SHEETS / "review_spotty_healthy.jpg")
    rh = W / "review_healthy.csv"
    if rh.exists() and not (W / "review_healthy_round1.csv").exists():
        shutil.copy2(rh, W / "review_healthy_round1.csv")
    with rh.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "spot_score"])
        for s, p in scored[:args.n]:
            w.writerow([str(p), round(s, 4)])
    print(f"{rh}  ({min(args.n, len(scored))} crops for: sort_crops.py --review healthy)")


if __name__ == "__main__":
    main()
