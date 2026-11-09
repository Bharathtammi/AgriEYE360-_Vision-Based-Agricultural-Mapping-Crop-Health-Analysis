"""
Score every crop not yet sorted with the current classifier and list the ones it is
least sure about (p_diseased closest to 0.5). The sorter then shows only those:

  "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" "F:\\Project and completed courses\\Agri EYE 360\\score_uncertain.py" --n 300
  "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" "F:\\Project and completed courses\\Agri EYE 360\\sort_crops.py" --uncertain
"""

import argparse
import csv
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "disease_autolabel"
CROPS = WORK / "crops"
MODEL = WORK / "models" / "disease_cls" / "weights" / "best.pt"
POOLS = ["auto_diseased", "auto_healthy", "review_likely_diseased", "review_likely_healthy", "unsorted"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--min-px", type=int, default=60, help="ignore tiny crops")
args = ap.parse_args()

model = YOLO(str(MODEL))
d_idx = [k for k, v in model.names.items() if v == "diseased"][0]
files = [f for p in POOLS for f in sorted((CROPS / p).glob("*.png"))]
files = [f for f in files if max(Image.open(f).size) >= args.min_px]
print(f"Scoring {len(files)} unsorted crops ...")
rows = []
for i in range(0, len(files), 128):
    chunk = files[i:i + 128]
    for f, r in zip(chunk, model.predict([str(f) for f in chunk], imgsz=224, verbose=False, device="cpu")):
        rows.append((f, float(r.probs.data[d_idx])))
    print(f"  {min(i + 128, len(files))}/{len(files)}", end="\r")
rows.sort(key=lambda x: abs(x[1] - 0.5))
out = WORK / "uncertain_crops.csv"
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "p_diseased"])
    w.writerows((str(f), f"{p:.3f}") for f, p in rows[:args.n])
print(f"\nLeast-certain {min(args.n, len(rows))} crops -> {out}")
