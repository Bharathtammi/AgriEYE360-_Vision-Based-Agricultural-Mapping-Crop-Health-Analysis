"""
Cotton boll datasets for Roboflow (both made from the 4-POV views).

  python make_roboflow_datasets.py

roboflow_export/
  1_cotton_boll_detection(.zip)        Object Detection   open_boll / closed_boll boxes on the views
                                                          (export_roboflow_bolls.py; SAM-only boxes need fixing in Roboflow)
  2_cotton_boll_classification(.zip)   Classification     open_boll / closed_boll / not_boll - ONLY your hand-sorted bolls

The leaf datasets (3, 4, 5) are made by make_pov_dataset.py, which also calls this script.
Train / valid split = agrieye_output/pov_dataset/split.csv (same for all datasets) when it exists.
"""
import csv
from collections import defaultdict
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "roboflow_export"
CROPS = ROOT / "agrieye_output" / "boll_crops"
SPLIT = ROOT / "agrieye_output" / "pov_dataset" / "split.csv"


def fresh(name):
    d = OUT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def zip_dir(d):
    zp = d.parent / f"{d.name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in d.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(d))
    return zp


def count(d):
    return {f"{s.name}/{c.name}": len(list(c.iterdir())) for s in sorted(d.iterdir()) if s.is_dir()
            for c in sorted(s.iterdir()) if c.is_dir()}


def split_map():
    """(photo_id, direction) -> train / valid / test, about 70 / 20 / 10 (Roboflow needs a test split).
    Unit = plant-visit + view direction: the 4 views of a photo do not overlap (90 deg each), so they
    show different bolls; all repeat shots of the same plant-visit + direction stay in one split.
    Open bolls are rare and bunched, so units are placed to hit the ratios for open bolls AND all bolls."""
    target = {"train": 0.7, "valid": 0.2, "test": 0.1}
    rows = list(csv.DictReader(open(SPLIT, encoding="utf-8")))
    group_of = {r["photo_id"]: r["group"] for r in rows}
    opened = {f.name for f in (CROPS / "open").glob("*.jpg")}
    n_open, n_all = defaultdict(int), defaultdict(int)
    for r in csv.DictReader(open(CROPS / "manifest.csv", encoding="utf-8")):
        u = (group_of.get(r["photo_id"]), r["direction"])
        n_all[u] += 1
        n_open[u] += r["crop"] in opened
    units = sorted({(r["group"], d) for r in rows for d in "NESW"}, key=lambda u: (-n_open[u], -n_all[u], u))
    tot = {s: [0, 0] for s in target}
    where = {}
    for u in units:
        o_sum = sum(t[0] for t in tot.values()) + n_open[u]
        a_sum = sum(t[1] for t in tot.values()) + n_all[u]
        # how far each split is below its target share (open bolls weigh most), pick the neediest
        need = lambda s: (2 * (target[s] - tot[s][0] / max(1, o_sum)) if n_open[u] else 0) + (target[s] - tot[s][1] / max(1, a_sum))
        s = max(target, key=need)
        tot[s][0] += n_open[u]; tot[s][1] += n_all[u]
        where[u] = s
    return {(r["photo_id"], d): where.get((r["group"], d), "train") for r in rows for d in "NESW"}


def boll_classification():
    """Clean crops (no drawn box) of the hand-sorted bolls, cut again from the 4-POV views."""
    d = fresh("2_cotton_boll_classification")
    names = {"open": "open_boll", "closed": "closed_boll", "not_boll": "not_boll"}
    hand = {f.name: lab for lab in names for f in (CROPS / lab).glob("*.jpg")}
    rows = [r for r in csv.DictReader(open(CROPS / "manifest.csv", encoding="utf-8")) if r["crop"] in hand]
    sp = split_map()
    cache = {}
    for r in rows:
        p = r["pov"]
        if p not in cache:
            cache.clear(); cache[p] = Image.open(p).convert("RGB")
        im = cache[p]
        x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        m = 0.15 * max(x2 - x1, y2 - y1)                       # a little context around the boll
        box = (int(max(0, x1 - m)), int(max(0, y1 - m)), int(min(im.width, x2 + m)), int(min(im.height, y2 + m)))
        split = sp.get((r["photo_id"], r["direction"]), "train")
        o = d / split / names[hand[r["crop"]]]; o.mkdir(parents=True, exist_ok=True)
        im.crop(box).save(o / r["crop"], quality=95)
    return d


def sky_negatives(d, per_view=4, seed=0):
    """Hard negatives: sky / cloud patches (boll-sized, away from every boll box) -> not_boll/sky_*.jpg.
    White clouds are what the model mixes up with open cotton, so cloudy patches are preferred."""
    import numpy as np
    rows = list(csv.DictReader(open(CROPS / "manifest.csv", encoding="utf-8")))
    boxes, info = {}, {}
    for r in rows:
        boxes.setdefault(r["pov"], []).append([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
        info[r["pov"]] = r
    sp = split_map()
    rng = random.Random(seed)
    sizes = sorted(max(float(r["x2"]) - float(r["x1"]), float(r["y2"]) - float(r["y1"])) for r in rows)
    n, sheet = 0, []
    for pov, bs in sorted(boxes.items()):
        im = Image.open(pov).convert("RGB")
        a = np.asarray(im).astype(int)
        R, G, B = a[..., 0], a[..., 1], a[..., 2]
        mx, mn = a.max(2), a.min(2)
        blue = (B > R + 20) & (B > G) & (mx > 110)
        cloud = (mx > 190) & (mx - mn < 35)
        sky = blue | cloud
        sky[int(a.shape[0] * 0.75):] = False                     # ground / lower plant: never sky
        ii = np.cumsum(np.cumsum(np.pad(sky, ((1, 0), (1, 0))).astype(np.int32), 0), 1)
        cc = np.cumsum(np.cumsum(np.pad(cloud, ((1, 0), (1, 0))).astype(np.int32), 0), 1)
        frac = lambda I, x1, y1, x2, y2: (I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]) / max(1, (x2 - x1) * (y2 - y1))
        got, tries = [], 0
        while len(got) < per_view and tries < 400:
            tries += 1
            s = int(sizes[rng.randrange(len(sizes))] * 1.3)      # same size spread as the boll crops (+ margin)
            s = max(48, min(s, 260))
            x1, y1 = rng.randrange(0, im.width - s), rng.randrange(0, int(im.height * 0.75) - s)
            x2, y2 = x1 + s, y1 + s
            if frac(ii, x1, y1, x2, y2) < 0.85:
                continue
            want_cloud = len(got) < per_view - 1                 # mostly cloudy patches, one plain sky
            if want_cloud and frac(cc, x1, y1, x2, y2) < 0.25:
                continue
            g = a[y1:y2, x1:x2].mean(2)                          # clouds are smooth, cotton fibre is textured:
            if (np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2 > 4.0:
                continue                                         # skip textured patches (could be a missed boll)
            if any(not (x2 < b[0] - 20 or x1 > b[2] + 20 or y2 < b[1] - 20 or y1 > b[3] + 20) for b in bs + got):
                continue
            got.append([x1, y1, x2, y2])
        r = info[pov]
        split = sp.get((r["photo_id"], r["direction"]), "train")
        o = d / split / "not_boll"; o.mkdir(parents=True, exist_ok=True)
        for k, b in enumerate(got):
            c = im.crop(tuple(b)); c.save(o / f"sky_{r['photo_id']}_{r['direction']}_{k}.jpg", quality=95)
            sheet.append(c); n += 1
    if sheet:
        T = 110
        sh = Image.new("RGB", (16 * T, ((min(len(sheet), 128) + 15) // 16) * T), (30, 30, 30))
        for i, c in enumerate(sheet[:128]):
            c = c.copy(); c.thumbnail((T - 4, T - 4)); sh.paste(c, ((i % 16) * T + 2, (i // 16) * T + 2))
        sh.save(OUT / "2_sky_negatives_preview.jpg", quality=88)
    return n


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "export_roboflow_bolls.py")], check=True)
    d = boll_classification()
    print(f"sky / cloud negatives added to not_boll: {sky_negatives(d)}")
    zp = zip_dir(d)
    print(f"\n{d.name}  [Classification]\n  {count(d)}\n  upload: {zp}  ({zp.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
