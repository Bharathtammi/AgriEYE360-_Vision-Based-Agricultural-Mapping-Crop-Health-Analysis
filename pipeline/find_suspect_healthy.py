"""
List crops labelled "healthy" that the classifier thinks look diseased, most suspicious
first, so you can re-check them.

Writes  disease_autolabel\\review_healthy.csv  and a contact sheet
disease_autolabel\\check_sheets\\suspect_healthy.jpg

Then review with:
  "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" "F:\\Project and completed courses\\Agri EYE 360\\sort_crops.py" --review healthy
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "disease_autolabel"
CROPS = WORK / "crops"
MODEL = WORK / "models" / "disease_cls" / "weights" / "best.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500, help="how many suspects to list")
args = ap.parse_args()

model = YOLO(str(MODEL))
d_idx = [k for k, v in model.names.items() if v == "diseased"][0]
files = sorted((CROPS / "healthy").glob("*.png"))
print(f"Scoring {len(files)} healthy crops ...")
rows = []
for i in range(0, len(files), 128):
    chunk = files[i:i + 128]
    for f, r in zip(chunk, model.predict([str(f) for f in chunk], imgsz=224, verbose=False, device="cpu")):
        rows.append((f, float(r.probs.data[d_idx])))
rows.sort(key=lambda x: -x[1])
top = rows[:args.n]
with (WORK / "review_healthy.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "p_diseased"])
    w.writerows((str(f), f"{p:.3f}") for f, p in top)
print(f"p_diseased >= 0.5: {sum(p >= 0.5 for _, p in rows)} | listed: {len(top)} "
      f"(lowest listed p = {top[-1][1]:.2f})")

sheet = Image.new("RGB", (8 * 160, 6 * 175), "black")
dr = ImageDraw.Draw(sheet)
for i, (f, p) in enumerate(top[:48]):
    im = Image.open(f).convert("RGB")
    im.thumbnail((150, 150))
    x, y = (i % 8) * 160, (i // 8) * 175
    sheet.paste(im, (x + 5, y + 5))
    dr.text((x + 5, y + 158), f"#{i + 1} p_dis {p:.2f}", fill="yellow")
(WORK / "check_sheets").mkdir(exist_ok=True)
sheet.save(WORK / "check_sheets" / "suspect_healthy.jpg")
print(f"List: {WORK / 'review_healthy.csv'}")
