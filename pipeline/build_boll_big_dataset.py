"""
Build the BIG training set for the local boll detector, with a FAIR test.

  python build_boll_big_dataset.py

train : ONE copy of every different picture in pov_all_labels/bolls_dataset (SAM 3 boxes + R1, hand-reviewed), EXCEPT
        - a picture that is (near-)identical to a hand-checked valid or test view - wherever its copies are in time
          (the camera saved the same frame again under new timestamps, up to 38 times),
        - the whole photo (4 views) of such a picture, and every photo taken the same day within 15 s of it.
        Copies = the same bytes, or a 24 px grey thumbnail differing < SAME on average (0-255).
        Views that are in the hand-checked TRAIN set use the hand-checked labels instead of the automatic ones.
valid : the 28 hand-checked validation views of roboflow_export/1_cotton_boll_detection (unchanged)
test  : the 18 hand-checked test views (unchanged) -> scores comparable with the current model and R2.

Output: agrieye_output/boll_detector/data_big/{train,valid,test}/{images,labels}  (images are hard links)
        data_big/sources.csv  (every train view: where its labels come from)
"""
import csv
import hashlib
import math
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "roboflow_export" / "1_cotton_boll_detection"
NEW = ROOT / "agrieye_output" / "pov_all_labels" / "bolls_dataset"
POV = ROOT / "agrieye_output" / "pov_all"
DST = ROOT / "agrieye_output" / "boll_detector" / "data_big"
RADIUS_M = 5.0   # not used: the field is only ~10 m wide, GPS cannot separate plants
WINDOW_S = 15    # neighbouring shots (photos were taken every 4-5 s while walking)
SAME = 1.0       # thumbnail difference below this = the same picture (copies measure 0.0-0.2; different photos 30+)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def thumb(p):
    return np.asarray(Image.open(p).convert("L").resize((24, 24)), np.float32)


def metres(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    x = (lo2 - lo1) * math.cos((la1 + la2) / 2)
    return 6371000 * math.hypot(x, la2 - la1)


def main():
    if DST.exists():                                         # keep the old build (and its label backups): rename
        old = DST.with_name(f"data_big_before_rebuild_{datetime.now():%Y%m%d_%H%M%S}")
        DST.rename(old)
        print(f"  old build kept as {old}")
    cat = {r["file"]: r for r in csv.DictReader(open(POV / "catalog.csv", encoding="utf-8"))}
    views = [r for r in csv.DictReader(open(NEW / "views.csv", encoding="utf-8")) if not r["copy_of"]]
    T = np.stack([thumb(POV / r["date"] / r["view"]) for r in views])
    # 0) group copies of the same picture: same bytes, or near-identical thumbnail
    grp, first = list(range(len(views))), {}
    for i, r in enumerate(views):
        grp[i] = first.setdefault(hashlib.md5((POV / r["date"] / r["view"]).read_bytes()).hexdigest(), i)
    for a in sorted(set(grp)):
        if grp[a] != a:
            continue
        for b in np.where(np.abs(T - T[a]).mean((1, 2)) < SAME)[0]:
            if grp[b] != a:
                old_g = grp[b]
                grp = [a if g == old_g else g for g in grp]
    members = defaultdict(list)
    for i, g in enumerate(grp):
        members[g].append(i)
    # 1) which group is each hand-checked view (same slicing -> near-identical pixels)
    match = {}                                               # group -> (old split, old label file)
    held = set()                                             # photo files showing a hand-checked valid / test picture
    for img in sorted(OLD.glob("*/images/*.jpg")):
        split = img.parent.parent.name
        d = np.abs(T - thumb(img)).mean((1, 2)); i = int(d.argmin())
        if d[i] > 3:
            print(f"  no match for {img.name} (diff {d[i]:.1f}) - skipped"); continue
        if split in ("valid", "test"):
            for j in set(np.where(d < SAME)[0]) | {i}:           # EVERY copy, wherever it is in time
                for m in members[grp[j]]:
                    held.add(views[m]["file"])
        elif grp[i] not in match:
            match[grp[i]] = (split, OLD / split / "labels" / f"{img.stem}.txt")
    # 2) repeat shots: same date, within RADIUS_M of a held-out photo
    def gps(f):
        r = cat[f]
        return (float(r["lat"]), float(r["lon"])) if r["lat"] and r["lon"] else None
    def secs(f):
        hh, mm, ss = map(int, cat[f]["time"].split(":"))
        return hh * 3600 + mm * 60 + ss
    near = set()
    for f in {v["file"] for v in views}:
        if f in held:
            continue
        for h in held:
            if cat[f]["date"] == cat[h]["date"] and abs(secs(f) - secs(h)) <= WINDOW_S:
                near.add(f); break
    # 3) write train: one copy per picture
    cnt, rows = Counter(), []
    cnt["views in bolls_dataset (not marked copy_of)"] = len(views)
    cnt["different pictures among them"] = len(members)
    for g, ms in sorted(members.items()):
        files = {views[m]["file"] for m in ms}
        if files & held:
            cnt["pictures left out: a copy of a hand-checked valid/test view"] += 1; continue
        if files & near:
            cnt[f"pictures left out: shot within {WINDOW_S} s of a valid/test photo"] += 1; continue
        r = views[ms[0]]                                     # the earliest copy
        name, stem = r["view"], Path(r["view"]).stem
        src = "automatic (SAM 3 + R1)"
        lab = NEW / "labels" / f"{stem}.txt"
        if g in match:
            lab, src = match[g][1], "hand-checked"
        link(POV / r["date"] / name, DST / "train" / "images" / name)
        copy(lab, DST / "train" / "labels" / f"{stem}.txt")
        cnt[f"train: {src}"] += 1
        rows.append(dict(view=name, date=r["date"], labels=src, copies=len(ms)))
    # 4) valid / test = hand-checked, unchanged
    for split in ("valid", "test"):
        for img in sorted((OLD / split / "images").glob("*.jpg")):
            link(img, DST / split / "images" / img.name)
            copy(OLD / split / "labels" / f"{img.stem}.txt", DST / split / "labels" / f"{img.stem}.txt")
            cnt[f"{split}: hand-checked"] += 1
    with open(DST / "sources.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["view", "date", "labels", "copies"]); w.writeheader(); w.writerows(rows)
    boxes = Counter()
    for split in ("train", "valid", "test"):
        for l in (DST / split / "labels").glob("*.txt"):
            for line in l.read_text().splitlines():
                if line.strip():
                    boxes[(split, line.split()[0])] += 1
    for k, v in sorted(cnt.items()):
        print(f"  {k:<50}{v:>6}")
    for split in ("train", "valid", "test"):
        print(f"  {split:<6} boxes: open {boxes[(split, '0')]:>6}   closed {boxes[(split, '1')]:>6}")
    print(f"Files: {DST}")


if __name__ == "__main__":
    main()
