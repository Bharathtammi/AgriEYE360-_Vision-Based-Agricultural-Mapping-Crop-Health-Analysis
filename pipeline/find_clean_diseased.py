"""
Rank crops labelled "diseased" from cleanest-looking to most clearly diseased, so you
can re-check the suspicious ones first.

Uses two independent signals and averages their ranks:
  1. colour/spot heuristic (pipeline.disease_score): brown/yellow + pale specks + dark spots
  2. the current classifier's p_diseased

Writes  disease_autolabel\\review_diseased.csv  (cleanest first) and a contact sheet
disease_autolabel\\check_sheets\\cleanest_diseased.jpg

Then review with:
  "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" "F:\\Project and completed courses\\Agri EYE 360\\sort_crops.py" --review
"""

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pipeline import disease_score  # noqa: E402

WORK = ROOT / "disease_autolabel"
CROPS = WORK / "crops"
MODEL = WORK / "models" / "disease_cls" / "weights" / "best.pt"

files = sorted((CROPS / "diseased").glob("*.png"))
print(f"Scoring {len(files)} diseased crops (colour/spots) ...")
heur = {}
for i, f in enumerate(files, 1):
    sc, _ = disease_score(f)
    heur[f] = sc if sc is not None else 0.0
    if i % 500 == 0:
        print(f"  {i}/{len(files)}")

pmod = {}
if MODEL.exists():
    from ultralytics import YOLO
    m = YOLO(str(MODEL))
    d_idx = [k for k, v in m.names.items() if v == "diseased"][0]
    print("Scoring with classifier ...")
    for i in range(0, len(files), 128):
        chunk = files[i:i + 128]
        for f, r in zip(chunk, m.predict([str(f) for f in chunk], imgsz=224, verbose=False, device="cpu")):
            pmod[f] = float(r.probs.data[d_idx])


def ranks(d):
    order = sorted(files, key=lambda f: d.get(f, 1.0))  # lowest score = cleanest
    return {f: i / len(files) for i, f in enumerate(order)}


rh = ranks(heur)
rm = ranks(pmod) if pmod else rh
combined = sorted(files, key=lambda f: (rh[f] + rm[f]) / 2)

out = WORK / "review_diseased.csv"
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "spot_score", "p_diseased"])
    for f in combined:
        w.writerow([str(f), f"{heur[f]:.3f}", f"{pmod.get(f, -1):.3f}"])

sheet = Image.new("RGB", (8 * 160, 6 * 175), "black")
dr = ImageDraw.Draw(sheet)
for i, f in enumerate(combined[:48]):
    im = Image.open(f).convert("RGB")
    im.thumbnail((150, 150))
    x, y = (i % 8) * 160, (i // 8) * 175
    sheet.paste(im, (x + 5, y + 5))
    dr.text((x + 5, y + 158), f"#{i + 1} spot {heur[f]:.2f} p {pmod.get(f, -1):.2f}", fill="yellow")
(WORK / "check_sheets").mkdir(exist_ok=True)
sheet.save(WORK / "check_sheets" / "cleanest_diseased.jpg")
print(f"Ranked list: {out}")
