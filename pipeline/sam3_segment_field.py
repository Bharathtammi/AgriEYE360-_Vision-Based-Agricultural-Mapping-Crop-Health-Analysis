"""
SAM 3 segmentation of the whole field (text prompts only), optionally followed by the YOLO disease model.

Runs on every 4-direction view listed in agrieye_output/stations.csv (repeat shots skipped).

  python sam3_segment_field.py                       # pure SAM 3, prompt "leaf"
  python sam3_segment_field.py --classify            # SAM 3 finds leaves -> YOLO says healthy / diseased
  python sam3_segment_field.py --classify --cls-model <best.pt>   # another disease model (e.g. v3)
  python sam3_segment_field.py --station P02 P06     # only some plants

Output (pure):      agrieye_output/sam3_segmentation/<plant>/<date>/
Output (classify):  agrieye_output/sam3_yolo_disease/<plant>/<date>/
  <photo>_<dir>_overlay.jpg    outlines (classify: green healthy, red diseased, yellow outline = needs checking)
  <photo>_<dir>_mask.png       black/white mask of all leaves
  <photo>_<dir>_objects.csv    one row per leaf (score, box, area; classify: label, p_diseased, needs_check)
  <photo>_<dir>_polygons.json  the SAM 3 outlines
  summary.csv                  per plant / visit (classify: leaves, diseased, % diseased, needs checking)
Resumable: views already done are skipped (use --force to redo).
"""
import argparse
import colorsys
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sam3_autolabel import build_predictor, detect

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "agrieye_output"
DIRS = ("N", "E", "S", "W")
MIN_SIDE = 40          # smaller leaf crops are too small to judge disease (same rule as the pipeline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", action="append")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--station", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--classify", action="store_true", help="run the YOLO disease model on every SAM 3 leaf")
    ap.add_argument("--cls-model", default=None, help="disease model weights (default: current v2)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    prompts = args.prompt or ["leaf"]
    dst = OUT / ("sam3_yolo_disease" if args.classify else "sam3_segmentation")
    rows = [r for r in csv.DictReader(open(OUT / "stations.csv", encoding="utf-8")) if r["station"].strip()]
    if args.station:
        rows = [r for r in rows if r["station"] in args.station]
    views = [(r, d, OUT / "povs" / r["station"] / r["date"] / f"{r['photo_id']}_{d}.jpg") for r in rows for d in DIRS]
    views = [v for v in views if v[2].exists()]
    cols = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((0.33 + k * 0.17) % 1, 0.9, 1.0)) for k in range(len(prompts))]

    P = cls = None
    if args.classify:
        import pipeline as P
        from ultralytics import YOLO
        cls_path = Path(args.cls_model) if args.cls_model else P.CLS_MODEL
        cls = YOLO(str(cls_path))
        print(f"Disease model: {cls_path}")
    pred = None
    for k, (r, d, p) in enumerate(views, 1):
        od = dst / r["station"] / r["date"]
        stem = f"{r['photo_id']}_{d}"
        if (od / f"{stem}_objects.csv").exists() and not args.force:
            continue
        if pred is None:
            pred, _ = build_predictor(ROOT / "sam3.pt", args.conf, args.device)
        od.mkdir(parents=True, exist_ok=True)
        img = Image.open(p).convert("RGB")
        dets = detect(pred, img, prompts, 1280, 0.2)
        objs, crops, crop_idx = [], [], []
        mask = Image.new("L", img.size, 0); dm = ImageDraw.Draw(mask)
        for i, (box, score, c, poly) in enumerate(dets):
            pts = [tuple(map(float, q)) for q in poly]
            one = Image.new("L", img.size, 0); ImageDraw.Draw(one).polygon(pts, fill=255)
            dm.polygon(pts, fill=255)
            b = [max(0, int(box[0])), max(0, int(box[1])), min(img.width, int(box[2]) + 1), min(img.height, int(box[3]) + 1)]
            objs.append(dict(id=i, prompt=prompts[c], score=round(score, 3), x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                             area_px=int(np.asarray(one).astype(bool).sum())))
            if cls is not None:
                if min(b[2] - b[0], b[3] - b[1]) >= MIN_SIDE and prompts[c] == "leaf":
                    crops.append(P.masked_crop(img, np.asarray(poly), b)); crop_idx.append(i)
                objs[-1].update(label="too_small", p_diseased="", needs_check="")
        if cls is not None and crops:
            for i, (lab, pd) in zip(crop_idx, P.classify(cls, crops)):
                objs[i].update(label=lab, p_diseased=round(pd, 3), needs_check=int(P.needs_check(pd)))
        # overlay
        ov = img.copy(); dr = ImageDraw.Draw(ov, "RGBA")
        for o, (box, score, c, poly) in zip(objs, dets):
            pts = [tuple(map(float, q)) for q in poly]
            if cls is None:
                col = line = cols[c % len(cols)]
            else:
                col = {"diseased": (230, 40, 40), "healthy": (40, 200, 60)}.get(o["label"], (150, 150, 150))
                line = (255, 200, 0) if o["needs_check"] == 1 else col
            dr.polygon(pts, fill=col + (70,), outline=line + (255,), width=4 if line != col else 2)
            dr.text((float(box[0]) + 3, float(box[1]) + 2), str(o["id"]), fill=(255, 255, 0))
        if cls is None:
            head = ", ".join(f"{pr}: {sum(o['prompt'] == pr for o in objs)}" for pr in prompts) + "   (SAM 3, text prompt only)"
        else:
            j = [o for o in objs if o["label"] in ("healthy", "diseased")]
            nd = sum(o["label"] == "diseased" for o in j)
            head = (f"leaves {len(j)}   diseased {nd} ({100 * nd / max(1, len(j)):.0f}%)   "
                    f"check {sum(o['needs_check'] == 1 for o in j)}   (SAM 3 segment -> YOLO disease)")
        dr.rectangle([0, 0, img.width, 30], fill=(0, 0, 0, 170))
        dr.text((8, 8), f"{r['station']} {r['date']} {d}   {head}", fill=(255, 255, 255))
        ov.save(od / f"{stem}_overlay.jpg", quality=88)
        mask.save(od / f"{stem}_mask.png")
        (od / f"{stem}_polygons.json").write_text(json.dumps(
            [dict(id=o["id"], prompt=o["prompt"], polygon=[[round(float(x), 1), round(float(y), 1)] for x, y in det[3]])
             for o, det in zip(objs, dets)]), encoding="utf-8")
        fields = ["id", "prompt", "score", "x1", "y1", "x2", "y2", "area_px"] + (["label", "p_diseased", "needs_check"] if cls is not None else [])
        with open(od / f"{stem}_objects.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(objs)
        print(f"[{k}/{len(views)}] {r['station']} {r['date']} {d}: {head}", flush=True)
    summarise(dst, views, prompts, args.classify)


def summarise(dst, views, prompts, classify):
    agg = defaultdict(lambda: defaultdict(float))
    for r, d, p in views:
        f = dst / r["station"] / r["date"] / f"{r['photo_id']}_{d}_objects.csv"
        if not f.exists():
            continue
        w, h = Image.open(p).size
        a = agg[(r["station"], r["date"])]
        a["view_px"] += w * h
        for o in csv.DictReader(open(f, encoding="utf-8")):
            a[f"{o['prompt']} count"] += 1
            a[f"{o['prompt']} area"] += int(o["area_px"])
            if classify and o.get("label") in ("healthy", "diseased"):
                a["leaves judged"] += 1
                a["diseased"] += o["label"] == "diseased"
                a["needs checking"] += o["needs_check"] == "1"
                a["confident diseased"] += o["label"] == "diseased" and o["needs_check"] != "1"
    out = []
    for (st, dt), a in sorted(agg.items()):
        row = dict(plant=st, date=dt)
        for pr in prompts:
            row[f"{pr} count"] = int(a[f"{pr} count"])
            row[f"{pr} % of view"] = round(100 * a[f"{pr} area"] / max(1, a["view_px"]), 1)
        if classify:
            row["leaves judged"] = int(a["leaves judged"])
            row["diseased"] = int(a["diseased"])
            row["% diseased"] = round(100 * a["diseased"] / max(1, a["leaves judged"]), 1)
            row["needs checking"] = int(a["needs checking"])
            row["confident diseased %"] = round(100 * a["confident diseased"] / max(1, a["leaves judged"]), 1)
        out.append(row)
    if out:
        with open(dst / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(out[0].keys())); wr.writeheader(); wr.writerows(out)
        print(f"\nSummary: {dst / 'summary.csv'}")
        for row in out:
            print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
