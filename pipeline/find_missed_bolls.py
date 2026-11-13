"""
Find cotton bolls that are MISSING from the labels (SAM 3 never boxed them), for you to check by hand.

  python find_missed_bolls.py                 # train + valid views (the 18 test views are left untouched)
  python find_missed_bolls.py --conf 0.15     # fewer, surer suggestions (default 0.10)
  python find_missed_bolls.py --include-test  # also the test views (then the detector test is no longer fair)

How: the best local boll detector (agrieye_output/boll_detector/boll_det) looks at every labelled view
(roboflow_export/1_cotton_boll_detection). A detection becomes a CANDIDATE when it
  - overlaps no existing label (IoU < 0.3 and its centre is not inside a labelled box), and
  - overlaps no box you already rejected as "not a boll" (boll_crops/not_boll).
Each candidate is cut out like the boll crops you sorted before (box drawn in green, ~360 px).

Then sort them with the same grid sorter:
  python sort_bolls_grid.py --dir boll_candidates
and rebuild the detection dataset (your accepted bolls are added automatically):
  python export_roboflow_bolls.py

Output: agrieye_output/boll_candidates/  unsorted/ open/ closed/ not_boll/ manifest.csv
Re-running adds only new candidates; crops you already sorted stay where they are.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "agrieye_output"
DATA = ROOT / "roboflow_export" / "1_cotton_boll_detection"
CROPS = OUT / "boll_crops"
DST = OUT / "boll_candidates"
DET = OUT / "boll_detector" / "boll_det" / "weights" / "best.pt"
NAMES = ["open_boll", "closed_boll"]
SIZE = 360
FIELDS = ["crop", "station", "date", "photo_id", "direction", "pov", "split", "x1", "y1", "x2", "y2",
          "sam_label", "score", "order"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def inside(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--include-test", action="store_true")
    a = ap.parse_args()
    if not DET.exists():
        raise SystemExit(f"No boll detector yet: {DET}  (train_boll_detector.py train)")
    from ultralytics import YOLO
    det = YOLO(str(DET))

    # every view of the dataset -> its original POV file, plant, date (from the boll-crop manifest)
    info, rejected = {}, defaultdict(list)
    not_boll = {f.name for f in (CROPS / "not_boll").glob("*.jpg")}
    for r in csv.DictReader(open(CROPS / "manifest.csv", encoding="utf-8")):
        key = f"{r['station']}_{r['date']}_{Path(r['pov']).stem}"
        info[key] = r
        if r["crop"] in not_boll:
            rejected[key].append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))

    for f in ("unsorted", "open", "closed", "not_boll"):
        (DST / f).mkdir(parents=True, exist_ok=True)
    old = list(csv.DictReader(open(DST / "manifest.csv", encoding="utf-8"))) if (DST / "manifest.csv").exists() else []
    have = defaultdict(list)
    for r in old:
        have[(r["photo_id"], r["direction"])].append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))

    splits = ["train", "valid"] + (["test"] if a.include_test else [])
    rows, n = list(old), defaultdict(int)
    for split in splits:
        for img_path in sorted((DATA / split / "images").glob("*.jpg")):
            key = img_path.stem
            r = info.get(key)
            if r is None:
                print(f"(no manifest entry for {key}, skipped)"); continue
            img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            labels = []
            lab_file = DATA / split / "labels" / f"{key}.txt"
            for l in lab_file.read_text().splitlines() if lab_file.exists() else []:
                if l.strip():
                    _, x, y, w, h = map(float, l.split())
                    labels.append(((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
            res = det.predict(str(img_path), imgsz=a.imgsz, conf=a.conf, iou=0.5, max_det=1000, device="cpu", verbose=False)[0]
            k = sum(1 for x in old if x["photo_id"] == r["photo_id"] and x["direction"] == r["direction"])
            for b, s, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                x1, y1, x2, y2 = b
                if x2 - x1 < 6 or y2 - y1 < 6:
                    continue
                cen = ((x1 + x2) / 2, (y1 + y2) / 2)
                if any(iou(b, L) >= 0.3 or inside(cen, L) for L in labels):
                    n["already labelled"] += 1; continue
                if any(iou(b, R) >= 0.3 for R in rejected[key]):
                    n["you rejected before"] += 1; continue
                if any(iou(b, E) >= 0.5 for E in have[(r["photo_id"], r["direction"])]):
                    n["already a candidate"] += 1; continue
                name = f"m_{r['photo_id']}_{r['direction']}_{k:03d}.jpg"; k += 1
                cx, cy = cen
                side = max(x2 - x1, y2 - y1) * 2.2 + 40
                a0, b0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
                c0, e0 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
                crop = img[b0:e0, a0:c0].copy()
                f = SIZE / max(crop.shape[:2])
                crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
                cv2.rectangle(crop, (int((x1 - a0) * f), int((y1 - b0) * f)), (int((x2 - a0) * f), int((y2 - b0) * f)),
                              (60, 230, 60), 2)
                cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(DST / "unsorted" / name))
                rows.append(dict(crop=name, station=r["station"], date=r["date"], photo_id=r["photo_id"],
                                 direction=r["direction"], pov=r["pov"], split=split,
                                 x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1),
                                 sam_label=NAMES[int(c)], score=round(float(s), 3), order=0 if int(c) == 0 else 1))
                have[(r["photo_id"], r["direction"])].append(tuple(b))
                n[f"new candidates ({split})"] += 1
                n[f"  suggested {NAMES[int(c)]}"] += 1
        print(f"{split}: done")
    rows.sort(key=lambda r: (int(r["order"]), -float(r["score"])))      # suggested open first, surest first
    with open(DST / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print("\n" + "\n".join(f"{k}: {v}" for k, v in n.items()))
    print(f"\nCandidates to check: {sum(1 for _ in (DST / 'unsorted').glob('*.jpg'))} in {DST / 'unsorted'}\n"
          f'Next: "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" "{ROOT / "sort_bolls_grid.py"}" --dir boll_candidates')


if __name__ == "__main__":
    main()
