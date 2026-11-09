"""
4-POV training data: EVERY unique 360 photo in all Data set folders -> N/E/S/W views -> all datasets
are built from those views only (no raw panorama goes into training).

  python make_pov_dataset.py build     # find unique photos, slice 4 POVs, move your hand labels onto the views
  python make_pov_dataset.py export    # Roboflow datasets 3/4/5 from the views (+ re-runs boll export 1/2)
  python make_pov_dataset.py all       # both

How the labels get onto the views:
  your hand-sorted leaves (disease_autolabel/crops/diseased|healthy|rejected) were outlined on the panoramas;
  every outline is projected with the exact inverse of the slicer's maths (same 90 deg FOV, pitch 0, 1440 px,
  north-aligned with the camera heading) into each view. A leaf cut by a view edge is kept if >= 60% of it
  is inside the view.

Train / valid split (shared by ALL datasets, file split.csv): grouped by plant + visit, so repeat shots of the
same plant on the same day are always in the same split; ~20% of the groups of each visit date go to valid.

Output: agrieye_output/pov_dataset/
  catalog.csv          one row per unique photo (source file, date, heading, plant, group, split)
  views/<date>/<photo_id>_<N|E|S|W>.jpg
  leaves.jsonl         one row per leaf per view: polygon on the view, hand label, % inside the view
  check_projection.jpg a few views with the moved outlines drawn (green healthy, red diseased, grey rejected)
  split.csv
"""
import csv
import json
import math
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import agrieye as A
import pipeline as P

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "AgriEye 360 field data" / "Data set"
OUT = ROOT / "agrieye_output" / "pov_dataset"
VIEWS = OUT / "views"
STATIONS = ROOT / "agrieye_output" / "stations.csv"
STATIONS_AUTO = ROOT / "agrieye_output" / "stations_backup_auto.csv"
EXPORT = ROOT / "roboflow_export"
DIRS = {"N": 0, "E": 90, "S": 180, "W": 270}
FOV, SIZE = 90, 1440
KEEP_INSIDE = 0.6          # leaf cut by the view edge: keep if this much of it is inside
MIN_SIDE = 40              # leaf crops smaller than this are too small to judge disease (pipeline rule)
MAX_HEALTHY = 2500         # same cap as disease model v2 (diseased ~1250)


# ------------------------------------------------------------------ build
def project(poly_n, W, H, yaw_deg):
    """Panorama outline (0..1 coords) -> pixel outline on the perspective view; None if behind the camera."""
    u = poly_n[:, 0] * W
    v = poly_n[:, 1] * H
    lon = u / (W - 1) * 2 * math.pi - math.pi
    lat = math.pi / 2 - v / (H - 1) * math.pi
    x3, z3, y = np.cos(lat) * np.sin(lon), np.cos(lat) * np.cos(lon), np.sin(lat)
    yw = math.radians(yaw_deg)
    x = x3 * math.cos(yw) - z3 * math.sin(yw)
    z = x3 * math.sin(yw) + z3 * math.cos(yw)
    if (z <= 0.1).any():
        return None
    f = (SIZE / 2) / math.tan(math.radians(FOV) / 2)
    gx, gy = f * x / z, -f * y / z
    k = (SIZE - 1) / SIZE
    return np.stack([(gx + SIZE / 2) * k, (gy + SIZE / 2) * k], 1)


def clip(poly):
    """Clip a pixel outline to the view. Returns (outline, fraction inside) or (None, 0)."""
    full = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1)))
    if full < 60:
        return None, 0.0
    lo, hi = poly.min(0), poly.max(0)
    if hi[0] < 0 or hi[1] < 0 or lo[0] >= SIZE or lo[1] >= SIZE:
        return None, 0.0
    if lo[0] >= 0 and lo[1] >= 0 and hi[0] < SIZE and hi[1] < SIZE:
        return poly, 1.0
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None, 0.0
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 60 or len(c) < 3:
        return None, 0.0
    return c.reshape(-1, 2).astype(float), min(1.0, cv2.contourArea(c) / full)


def catalog():
    """Unique photos over all Data set folders (content hash), with plant + group."""
    st = {r["photo_id"]: r["station"].strip() for r in csv.DictReader(open(STATIONS, encoding="utf-8"))}
    auto = {r["photo_id"]: r["station"].strip() for r in csv.DictReader(open(STATIONS_AUTO, encoding="utf-8"))} \
        if STATIONS_AUTO.exists() else {}
    files_of, hash_of = defaultdict(list), {}
    for f in A.find_images([DATA]):
        h = A.content_hash(f)
        hash_of[str(f)] = h
        files_of[h].append(f)
    seen = {}
    for h, fs in files_of.items():
        # the same picture can be saved twice with different timestamps (e.g. "360 files data\2.jpg" =
        # overall dataset image_20260731_080718.jpg): keep the copy whose id is in stations.csv
        cands = []
        for f in fs:
            m = A.read_meta(f)
            if m["panorama"] and m["datetime"]:
                pid = m["datetime"].strftime("%Y%m%d_%H%M%S")
                cands.append((pid not in st and pid not in auto, pid, f, m))
                if pid in st:
                    break
        if not cands:
            continue
        _, pid, f, m = min(cands, key=lambda c: c[0])
        plant = st.get(pid, "")
        repeat = ""
        if not plant and auto.get(pid):                 # repeat shot you blanked -> same group as its twin
            plant, repeat = auto[pid], "repeat shot"
        date = m["datetime"].strftime("%Y-%m-%d")
        seen[h] = dict(photo_id=pid, date=date, time=m["datetime"].strftime("%H:%M:%S"),
                       heading=f"{m['heading'] or 0:.2f}", width=m["width"], height=m["height"],
                       plant=plant or "unassigned", note=repeat or ("" if plant else "no plant ID"),
                       group=f"{plant or pid}_{date}", source=str(f))
    return list(seen.values()), hash_of


def make_split(rows, valid=0.2):
    by_date = defaultdict(set)
    for r in rows:
        by_date[r["date"]].add(r["group"])
    val = set()
    for d, gs in sorted(by_date.items()):
        gs = sorted(gs); random.Random(0).shuffle(gs)
        if len(gs) >= 3:
            val |= set(gs[:max(1, round(len(gs) * valid))])
    for r in rows:
        r["split"] = "valid" if r["group"] in val else "train"


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Finding unique 360 photos in all Data set folders ...")
    rows, hash_of = catalog()
    rows.sort(key=lambda r: (r["date"], r["time"]))
    make_split(rows)
    A.write_csv(OUT / "catalog.csv", rows, list(rows[0].keys()))
    with open(OUT / "split.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["photo_id", "group", "split"])
        w.writerows([r["photo_id"], r["group"], r["split"]] for r in rows)
    print(f"{len(rows)} unique 360 photos; dates {sorted({r['date'] for r in rows})}")

    # slice
    for k, r in enumerate(rows, 1):
        outs = {d: VIEWS / r["date"] / f"{r['photo_id']}_{d}.jpg" for d in DIRS}
        if all(p.exists() for p in outs.values()):
            continue
        img = A.imread(r["source"])
        for d, compass in DIRS.items():
            yaw = ((compass - float(r["heading"]) + 180) % 360) - 180
            A.imwrite(outs[d], A.equirect_to_perspective(img, FOV, yaw, 0, SIZE))
        print(f"  sliced [{k}/{len(rows)}] {r['date']} {r['photo_id']}")

    # move hand labels onto the views
    label = {f.name: c for c in ("diseased", "healthy", "rejected") for f in (P.CROPS / c).glob("*.png")}
    photo_by_hash = {A.content_hash(Path(r["source"])): r for r in rows}
    src_photo = {}
    n = defaultdict(int)
    with open(OUT / "leaves.jsonl", "w", encoding="utf-8") as fh:
        for rec in P.read_manifest():
            s = rec["source"]
            if s not in src_photo:
                h = hash_of.get(s) or (A.content_hash(Path(s)) if Path(s).exists() else None)
                src_photo[s] = photo_by_hash.get(h)
            ph = src_photo[s]
            if ph is None:
                n["leaf from a non-360 photo (skipped)"] += 1
                continue
            poly_n = np.asarray(rec["polygon"], float)
            placed = False
            for d, compass in DIRS.items():
                yaw = ((compass - float(ph["heading"]) + 180) % 360) - 180
                pp = project(poly_n, rec["width"], rec["height"], yaw)
                if pp is None:
                    continue
                cp, frac = clip(pp)
                if cp is None or frac < 0.05:
                    continue
                fh.write(json.dumps(dict(photo_id=ph["photo_id"], date=ph["date"], plant=ph["plant"], dir=d,
                                         split=ph["split"], crop=rec["crop"], label=label.get(rec["crop"], "unsorted"),
                                         inside=round(frac, 3),
                                         polygon=[[round(float(x), 1), round(float(y), 1)] for x, y in cp])) + "\n")
                placed = placed or frac >= KEEP_INSIDE
            n["leaves placed on a view" if placed else "leaves outside the 4 views (above/below 45 deg)"] += 1
    for k, v in n.items():
        print(f"  {k}: {v}")
    check_sheet()


def load_leaves():
    return [json.loads(l) for l in (OUT / "leaves.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def check_sheet():
    by_view = defaultdict(list)
    for l in load_leaves():
        by_view[(l["date"], l["photo_id"], l["dir"])].append(l)
    pick = sorted(by_view, key=lambda k: -sum(x["label"] == "diseased" for x in by_view[k]))[:6]
    T = 600
    sheet = Image.new("RGB", (3 * T, 2 * T), (30, 30, 30))
    col = {"diseased": (230, 40, 40), "healthy": (40, 220, 60), "rejected": (160, 160, 160), "unsorted": (255, 200, 0)}
    for i, (date, pid, d) in enumerate(pick):
        im = Image.open(VIEWS / date / f"{pid}_{d}.jpg").convert("RGB"); dr = ImageDraw.Draw(im, "RGBA")
        for l in by_view[(date, pid, d)]:
            c = col[l["label"]]
            dr.polygon([tuple(p) for p in l["polygon"]], fill=c + (60,), outline=c + (255,), width=3)
        dr.rectangle([0, 0, SIZE, 44], fill=(0, 0, 0, 200))
        dr.text((10, 14), f"{date} {pid} {d}  green=healthy red=diseased grey=rejected (your labels, moved from panorama)",
                fill=(255, 255, 255))
        im.thumbnail((T, T)); sheet.paste(im, ((i % 3) * T, (i // 3) * T))
    sheet.save(OUT / "check_projection.jpg", quality=88)
    print(f"Check: {OUT / 'check_projection.jpg'}")


# ------------------------------------------------------------------ export
def fresh(name):
    d = EXPORT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def zip_dir(d):
    import zipfile
    zp = d.parent / f"{d.name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in d.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(d))
    return zp


def view_name(l):
    return f"{l['plant']}_{l['date']}_{l['photo_id']}_{l['dir']}"


def export():
    leaves = load_leaves()
    by_view = defaultdict(list)
    for l in leaves:
        by_view[(l["date"], l["photo_id"], l["dir"], l["split"], l["plant"])].append(l)
    cat = list(csv.DictReader(open(OUT / "catalog.csv", encoding="utf-8")))

    # 3) leaf segmentation: every view, every leaf outline (all your sorted SAM 3 leaves, like the old dataset)
    d3 = fresh("3_leaf_segmentation")
    n3 = defaultdict(int)
    for r in cat:
        for dd in DIRS:
            key = (r["date"], r["photo_id"], dd, r["split"], r["plant"])
            ls = by_view.get(key, [])
            if not ls:
                continue
            nm = view_name(dict(plant=r["plant"], date=r["date"], photo_id=r["photo_id"], dir=dd))
            for kind in ("images", "labels"):
                (d3 / r["split"] / kind).mkdir(parents=True, exist_ok=True)
            shutil.copy2(VIEWS / r["date"] / f"{r['photo_id']}_{dd}.jpg", d3 / r["split"] / "images" / f"{nm}.jpg")
            (d3 / r["split"] / "labels" / f"{nm}.txt").write_text("\n".join(
                "0 " + " ".join(f"{x / SIZE:.5f} {y / SIZE:.5f}" for x, y in l["polygon"]) for l in ls) + "\n", encoding="utf-8")
            n3[r["split"] + " views"] += 1
            n3[r["split"] + " leaves"] += len(ls)
    (d3 / "data.yaml").write_text("train: train/images\nval: valid/images\n\nnc: 1\nnames: ['leaf']\n", encoding="utf-8")

    # 4) leaf disease (field): your healthy / diseased labels, cut from the views
    d4 = fresh("4_leaf_disease_field")
    keep = [l for l in leaves if l["label"] in ("healthy", "diseased") and l["inside"] >= KEEP_INSIDE]
    best = {}                                   # a leaf seen in 2 views (on the edge): keep the more complete one
    for l in keep:
        if l["crop"] not in best or l["inside"] > best[l["crop"]]["inside"]:
            best[l["crop"]] = l
    keep = list(best.values())
    hl = [l for l in keep if l["label"] == "healthy"]
    random.Random(0).shuffle(hl)
    keep = [l for l in keep if l["label"] == "diseased"] + hl[:MAX_HEALTHY]
    n4 = defaultdict(int)
    keep.sort(key=lambda l: (l["date"], l["photo_id"], l["dir"]))
    cur, img = None, None
    for l in keep:
        v = VIEWS / l["date"] / f"{l['photo_id']}_{l['dir']}.jpg"
        if v != cur:
            cur, img = v, Image.open(v).convert("RGB")
        poly = np.asarray(l["polygon"])
        b = [int(max(0, poly[:, 0].min())), int(max(0, poly[:, 1].min())),
             int(min(SIZE, poly[:, 0].max() + 1)), int(min(SIZE, poly[:, 1].max() + 1))]
        if min(b[2] - b[0], b[3] - b[1]) < MIN_SIDE:
            n4["too small on the view (skipped)"] += 1
            continue
        o = d4 / l["split"] / l["label"]; o.mkdir(parents=True, exist_ok=True)
        P.masked_crop(img, [tuple(p) for p in poly], b).save(o / f"{view_name(l)}_{Path(l['crop']).stem}.png")
        n4[f"{l['split']}/{l['label']}"] += 1

    # 5) field (4-POV crops above) + outside public leaves (ordinary photos, not 360 - used as they are)
    d5 = fresh("5_leaf_disease_field_outside")
    shutil.copytree(d4, d5, dirs_exist_ok=True)
    v3 = P.WORK / "dataset_cls_v3"
    n5 = 0
    for a, b in (("train", "train"), ("val", "valid")):
        for c in ("healthy", "diseased"):
            for f in (v3 / a / c).glob("ext_*"):
                (d5 / b / c).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d5 / b / c / f.name); n5 += 1

    print("\n1/2) cotton bolls")
    subprocess.run([sys.executable, str(ROOT / "make_roboflow_datasets.py")], check=True)
    for d, info in ((d3, dict(n3)), (d4, dict(n4)), (d5, f"field crops above + {n5} outside leaves")):
        zp = zip_dir(d)
        print(f"\n{d.name}: {info}\n  upload: {zp}  ({zp.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("build", "all"):
        build()
    if cmd in ("export", "all"):
        export()
