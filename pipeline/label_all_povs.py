"""
Label ALL 4-direction views (agrieye_output/pov_all, from slice_all_povs.py) for COTTON BOLLS and LEAVES.
Method "A" = the same as the field results:

  SAM 3 (text prompts, conf 0.3, 960 px tiles) finds
     bolls  : "open cotton boll", "closed cotton boll", "flower"  -> cleaned with agrieye.refine (same rules as autolabel)
     leaves : "leaf"  (outline polygons)
  L2 (disease classifier v2) judges every leaf big enough to judge (>= 40 px): healthy / diseased (+ needs_check band)
  Bolls are judged open / closed / not a boll afterwards by R1 on Roboflow: label_all_bolls_r1.py (needs your key)

  python label_all_povs.py              # all views; resumable (views already done are skipped)
  python label_all_povs.py --limit 4    # quick timing test

Exact byte copies (catalog.csv duplicate_of) are NOT run again: they get the labels of the original file.
SAM 3 reads each tile ONCE and is asked for bolls and leaves in turn (saves ~half the time).

Output: agrieye_output/pov_all_labels/
  sam3.jsonl          one line per view: view, file, date, direction, copy_of, bolls [box, score, cls],
                      leaves [box, score, polygon, label, p_diseased, needs_check]
  previews/<date>/    a picture per view: bolls (white = open, red = closed, pink = flower), leaves (green / red)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import agrieye as A
import pipeline as P
from sam3_autolabel import box_iou, make_tiles

ROOT = Path(__file__).resolve().parent
VIEWS = ROOT / "agrieye_output" / "pov_all"
OUT = ROOT / "agrieye_output" / "pov_all_labels"
JSONL = OUT / "sam3.jsonl"
BOLL_PROMPTS = ["open cotton boll", "closed cotton boll", "flower"]     # = agrieye_config.json objects.prompts
NAMES = ["open_boll", "closed_boll", "flower"]
TILE, CONF, MIN_SIDE = 960, 0.3, 40


def simplify(poly):
    import cv2
    p = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    p = cv2.approxPolyDP(p, 1.5, True).reshape(-1, 2)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in p]


def label_view(pred, cls_model, img):
    W, H = img.size
    bolls, leaves = [], []
    for x0, y0, x1, y1 in make_tiles(W, H, TILE, 0.2):
        arr = np.ascontiguousarray(np.asarray(img.crop((x0, y0, x1, y1)))[:, :, ::-1])
        pred.set_image(arr)                                         # encode the tile once
        r = pred(text=BOLL_PROMPTS); r = r[0] if isinstance(r, (list, tuple)) else r
        if r.boxes is not None and len(r.boxes):
            for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                bolls.append(dict(box=[float(b[0] + x0), float(b[1] + y0), float(b[2] + x0), float(b[3] + y0)],
                                  score=float(s), cls=int(c)))
        r = pred(text=["leaf"]); r = r[0] if isinstance(r, (list, tuple)) else r
        if r.masks is not None and len(r.masks):
            for b, s, poly in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.masks.xy):
                if len(poly) >= 3:
                    leaves.append(dict(box=[float(b[0] + x0), float(b[1] + y0), float(b[2] + x0), float(b[3] + y0)],
                                       score=float(s), poly=(np.asarray(poly) + [x0, y0]).tolist()))
    bolls = A.refine(img, A.merge(bolls), NAMES)                   # same clean-up as agrieye autolabel
    leaves.sort(key=lambda d: -d["score"])                          # merge leaves cut by tile overlaps
    kept = []
    for d in leaves:
        if all(box_iou(d["box"], k["box"]) < 0.6 for k in kept):
            kept.append(d)
    crops, idx = [], []
    for i, d in enumerate(kept):
        b = [max(0, int(d["box"][0])), max(0, int(d["box"][1])), min(W, int(d["box"][2]) + 1), min(H, int(d["box"][3]) + 1)]
        d.update(label="too_small", p_diseased="", needs_check="")
        if min(b[2] - b[0], b[3] - b[1]) >= MIN_SIDE:
            crops.append(P.masked_crop(img, np.asarray(d["poly"]), b)); idx.append(i)
    for i, (lab, pd) in zip(idx, P.classify(cls_model, crops) if crops else []):
        kept[i].update(label=lab, p_diseased=round(float(pd), 3), needs_check=int(P.needs_check(pd)))
    for d in kept:
        d["poly"] = simplify(d["poly"]); d["box"] = [round(v, 1) for v in d["box"]]; d["score"] = round(d["score"], 3)
    return bolls, kept


def preview(img, bolls, leaves, path):
    im = img.copy(); dr = ImageDraw.Draw(im, "RGBA")
    for d in leaves:
        col = {"diseased": (230, 40, 40), "healthy": (40, 200, 60)}.get(d["label"], (150, 150, 150))
        dr.polygon([tuple(q) for q in d["poly"]], fill=col + (60,), outline=col + (255,), width=2)
    bc = [(255, 255, 255), (230, 60, 60), (255, 120, 220)]
    for d in bolls:
        dr.rectangle(d["box"], outline=bc[d["cls"]] + (255,), width=4)
    im.thumbnail((900, 900)); path.parent.mkdir(parents=True, exist_ok=True); im.save(path, quality=82)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cat = list(csv.DictReader(open(VIEWS / "catalog.csv", encoding="utf-8")))
    stem = {r["file"]: Path(r["file"]).stem for r in cat}
    jobs = [(r, d) for r in cat if str(r["views"]) == "4" for d in "NESW"]
    OUT.mkdir(parents=True, exist_ok=True)
    done = {}
    if JSONL.exists():
        for l in JSONL.read_text(encoding="utf-8").splitlines():
            if l.strip():
                j = json.loads(l); done[(j["file"], j["direction"])] = j
    todo = [(r, d) for r, d in jobs if (r["file"], d) not in done and not r["duplicate_of"]]
    copies = [(r, d) for r, d in jobs if (r["file"], d) not in done and r["duplicate_of"]]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(jobs)} views: {len(done)} done, {len(todo)} to label with SAM 3, {len(copies)} exact copies (labels reused)")
    if todo:
        from ultralytics import YOLO
        from sam3_autolabel import build_predictor
        pred, _ = build_predictor(ROOT / "sam3.pt", CONF, "cpu")
        cls_model = YOLO(str(P.CLS_MODEL))
        t0 = time.time()
        with JSONL.open("a", encoding="utf-8") as fh:
            for k, (r, d) in enumerate(todo, 1):
                view = VIEWS / r["date"] / f"{stem[r['file']]}_{d}.jpg"
                img = Image.open(view).convert("RGB")
                bolls, leaves = label_view(pred, cls_model, img)
                j = dict(view=str(view), file=r["file"], date=r["date"], direction=d, copy_of="",
                         width=img.width, height=img.height, bolls=bolls, leaves=leaves)
                fh.write(json.dumps(j) + "\n"); fh.flush()
                done[(r["file"], d)] = j
                preview(img, bolls, leaves, OUT / "previews" / r["date"] / f"{stem[r['file']]}_{d}.jpg")
                el = time.time() - t0
                print(f"  {k}/{len(todo)} {view.name}: bolls {len(bolls)}, leaves {len(leaves)} "
                      f"({sum(x['label'] == 'diseased' for x in leaves)} diseased)  "
                      f"{el / k:.0f} s/view, ~{el / k * (len(todo) - k) / 3600:.1f} h left", flush=True)
    # exact copies: same labels as the original file, written under the copy's own name
    n = 0
    with JSONL.open("a", encoding="utf-8") as fh:
        for r, d in copies:
            src = done.get((r["duplicate_of"], d))
            if src is None:
                continue                              # original not labelled yet (e.g. --limit): next run
            j = dict(src, view=str(VIEWS / r["date"] / f"{stem[r['file']]}_{d}.jpg"), file=r["file"],
                     date=r["date"], copy_of=r["duplicate_of"])
            fh.write(json.dumps(j) + "\n"); n += 1
    if n:
        print(f"{n} copy views written with their original's labels")
    print(f"Files: {OUT}")


if __name__ == "__main__":
    main()
