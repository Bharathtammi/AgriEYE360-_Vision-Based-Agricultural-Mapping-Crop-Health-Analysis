"""
Cut every SAM 3 boll/flower detection into a crop for hand-sorting with sort_bolls.py.

  python make_boll_crops.py

Output: agrieye_output/boll_crops/
  unsorted/<photo_id>_<dir>_<k>.jpg   detection with context, box drawn in green
  open/ closed/ not_boll/             filled by sort_bolls.py
  manifest.csv                        crop -> pov, station, date, box, SAM label, score; also the
                                      sorting order (SAM "open" first, then "flower", then "closed")
Photos that are blank in stations.csv (repeat shots) are left out.
Safe to re-run: crops that already exist (sorted or not) are not touched.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "agrieye_output"
DST = OUT / "boll_crops"
NAMES = {0: "open_boll", 1: "closed_boll", 2: "flower"}
ORDER = {0: 0, 2: 1, 1: 2}       # sorting order: open, flower, closed
FOLDERS = ["unsorted", "open", "closed", "not_boll"]
SIZE = 360


def find_pov(r):
    p = Path(r["pov"])
    if p.exists():
        return p
    hits = list((OUT / "povs").glob(f"*/{r['date']}/{p.name}"))
    return hits[0] if hits else None


def main():
    st = {r["photo_id"]: r["station"].strip() for r in csv.DictReader(open(OUT / "stations.csv", encoding="utf-8"))}
    recs = [json.loads(l) for l in (OUT / "objects" / "sam3_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    extra = OUT / "objects" / "sam3_labels_extra.jsonl"          # boll_extra_views.py: 22 Jul + repeat shots
    if extra.exists():
        recs += [json.loads(l) for l in extra.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in csv.DictReader(open(OUT / "pov_dataset" / "catalog.csv", encoding="utf-8")):
            if not st.get(r["photo_id"]):
                st[r["photo_id"]] = r["plant"]
    for f in FOLDERS:
        (DST / f).mkdir(parents=True, exist_ok=True)
    existing = {p.name for f in FOLDERS for p in (DST / f).glob("*.jpg")}
    rows, made, skipped = [], 0, 0
    for r in recs:
        if not st.get(r["photo_id"]):
            skipped += 1
            continue
        p = find_pov(r)
        img = None
        for k, d in enumerate(r["dets"]):
            name = f"{r['photo_id']}_{r['direction']}_{k:03d}.jpg"
            x1, y1, x2, y2 = d["box"]
            rows.append(dict(crop=name, station=st[r["photo_id"]], date=r["date"], photo_id=r["photo_id"],
                             direction=r["direction"], pov=str(p) if p else r["pov"],
                             x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1),
                             sam_label=NAMES[d["cls"]], score=round(d["score"], 3),
                             order=ORDER[d["cls"]]))
            if name in existing:
                continue
            if img is None:
                img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            s = max(x2 - x1, y2 - y1) * 2.2 + 40
            a, b = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
            c, e = int(min(img.shape[1], cx + s / 2)), int(min(img.shape[0], cy + s / 2))
            crop = img[b:e, a:c].copy()
            k_ = SIZE / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=k_, fy=k_, interpolation=cv2.INTER_CUBIC if k_ > 1 else cv2.INTER_AREA)
            cv2.rectangle(crop, (int((x1 - a) * k_), int((y1 - b) * k_)), (int((x2 - a) * k_), int((y2 - b) * k_)),
                          (60, 230, 60), 2)
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(DST / "unsorted" / name))
            made += 1
    rows.sort(key=lambda r: (r["order"], -r["score"]))
    with open(DST / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by = {n: sum(r["sam_label"] == n for r in rows) for n in NAMES.values()}
    print(f"{len(rows)} detections ({by}); {made} new crops; {skipped} repeat-shot views left out")
    print(f"Crops:    {DST / 'unsorted'}\nManifest: {DST / 'manifest.csv'}")


if __name__ == "__main__":
    main()
