"""
Slice EVERY 360 photo of the overall dataset into 4 compass-aligned views (N, E, S, W).

  python slice_all_povs.py

Input : AgriEye 360 field data/Data set/overall dataset/   (all files; normal phone photos cannot be sliced and are listed)
Output: agrieye_output/pov_all/<date>/<photo>_<N|E|S|W>.jpg     1440 x 1440 px, 90 deg field of view, level (pitch 0)
        agrieye_output/pov_all/catalog.csv   one row per input file: date, time, GPS, heading, size, md5,
                                             duplicate_of (exact byte copy of an earlier file), views written
Same projection and compass alignment as agrieye.py slice (yaw = compass direction - camera heading), so
these views line up with the 140 views used so far. Exact copies are sliced too but marked, so a copy and its
twin can be kept in the same train / valid / test split. Resumable: views that already exist are skipped.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np

import agrieye as A

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "AgriEye 360 field data" / "Data set" / "overall dataset"
DST = ROOT / "agrieye_output" / "pov_all"
DIRS = {"N": 0, "E": 90, "S": 180, "W": 270}
FOV, PITCH, SIZE = 90, 0, 1440


def main():
    A.Image.MAX_IMAGE_PIXELS = None
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in SRC.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    first, rows, made, skipped = {}, [], 0, []
    for k, p in enumerate(files, 1):
        m = A.read_meta(p)
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        dup = first.get(md5, "")
        first.setdefault(md5, p.name)
        dt = m["datetime"]
        date = dt.strftime("%Y-%m-%d") if dt else "unknown"
        row = dict(file=p.name, date=date, time=dt.strftime("%H:%M:%S") if dt else "",
                   lat=m["lat"] or "", lon=m["lon"] or "", heading="" if m["heading"] is None else m["heading"],
                   width=m["width"], height=m["height"], md5=md5, duplicate_of=dup, views=0)
        if not m["panorama"]:
            row["views"] = "not 360 - skipped"; skipped.append(p.name); rows.append(row); continue
        outs = {d: DST / date / f"{p.stem}_{d}.jpg" for d in DIRS}
        if all(o.exists() for o in outs.values()):
            row["views"] = 4; rows.append(row); continue
        img = A.imread(p)
        heading = float(m["heading"]) if m["heading"] is not None else 0.0
        for d, compass in DIRS.items():
            if not outs[d].exists():
                yaw = ((compass - heading + 180) % 360) - 180            # compass bearing -> image yaw
                outs[d].parent.mkdir(parents=True, exist_ok=True)
                A.imwrite(outs[d], A.equirect_to_perspective(img, FOV, yaw, PITCH, SIZE))
                made += 1
        row["views"] = 4; rows.append(row)
        if k % 20 == 0 or k == len(files):
            print(f"  {k}/{len(files)} photos, {made} new views", flush=True)
    with open(DST / "catalog.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    n360 = sum(1 for r in rows if r["views"] == 4)
    print(f"\n{len(files)} files: {n360} sliced into 4 views = {4 * n360} views "
          f"({sum(1 for r in rows if r['duplicate_of'] and r['views'] == 4)} are exact copies of another file)")
    if skipped:
        print("Not 360 (not sliced):", ", ".join(skipped))
    per = {}
    for r in rows:
        if r["views"] == 4:
            per[r["date"]] = per.get(r["date"], 0) + 4
    print("Views per date:", ", ".join(f"{d}: {n}" for d, n in sorted(per.items())))
    print(f"Files: {DST}")


if __name__ == "__main__":
    main()
