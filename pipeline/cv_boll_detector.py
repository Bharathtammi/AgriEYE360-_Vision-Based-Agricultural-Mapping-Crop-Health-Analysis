"""
Step 4: 5-FOLD CROSS-VALIDATION of the local boll detector on the clean photo split (build_photo_split.py).

  python cv_boll_detector.py              # train the folds that are not done yet (resumes), then score
  python cv_boll_detector.py report       # only score the folds that are done (no training)

Fold k: test = fold k, valid = fold k+1, train = the other 3 folds (photos never shared between them).
Every one of the 140 different pictures is tested exactly once, by a model that never saw its photo.
Training settings = the old hand-set model (YOLO11n, 30 epochs, 1280 px, batch 4, no brightness change).
The ACTIVE model (agrieye_output/boll_detector/boll_det) is NOT touched.
Output: agrieye_output/boll_detector/cv/  fold{k}/ (data, run/weights/best.pt, predictions.json), report.txt
"""
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import eval_roboflow_detect as E
import rescore_boll_detector as R

ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / "agrieye_output" / "boll_detector" / "data_photo_split"
CV = ROOT / "agrieye_output" / "boll_detector" / "cv"
FOLDS, EPOCHS, IMGSZ = 5, 30, 1280


def fold_data(k, rows):
    d = CV / f"fold{k}" / "data"
    if d.exists():
        shutil.rmtree(d)
    role = {k: "test", (k + 1) % FOLDS: "valid"}
    for r in rows:
        s = role.get(int(r["fold"]), "train")
        src = SPLIT / r["split"]
        for sub in ("images", "labels"):
            (d / s / sub).mkdir(parents=True, exist_ok=True)
        img = d / s / "images" / r["view"]
        try:
            os.link(src / "images" / r["view"], img)
        except OSError:
            shutil.copy2(src / "images" / r["view"], img)
        shutil.copy2(src / "labels" / f"{Path(r['view']).stem}.txt", d / s / "labels" / f"{Path(r['view']).stem}.txt")
    y = d / "data.yaml"
    y.write_text(f"path: {d.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                 "nc: 2\nnames: ['open_boll', 'closed_boll']\n", encoding="utf-8")
    return y


def run_fold(k, rows):
    from ultralytics import YOLO
    f = CV / f"fold{k}"
    if (f / "predictions.json").exists():
        print(f"fold {k}: done already", flush=True); return
    y = fold_data(k, rows)
    print(f"fold {k}: training ({EPOCHS} epochs) ...", flush=True)
    YOLO("yolo11n.pt").train(
        data=str(y), epochs=EPOCHS, imgsz=IMGSZ, batch=4, device="cpu", workers=2, patience=EPOCHS,
        hsv_v=0.0, hsv_s=0.3, hsv_h=0.01, fliplr=0.5, degrees=10, mosaic=1.0, seed=0,
        project=str(f), name="run", exist_ok=True, plots=True, verbose=False)
    m = YOLO(str(f / "run" / "weights" / "best.pt"))
    preds = {}
    for img in sorted((f / "data" / "test" / "images").glob("*.jpg")):
        r = m.predict(str(img), imgsz=IMGSZ, conf=0.01, iou=0.5, max_det=1000, device="cpu", verbose=False)[0]
        preds[img.stem] = [(int(c), float(s), [float(v) for v in b]) for b, s, c in
                           zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    (f / "predictions.json").write_text(json.dumps(preds))
    print(f"fold {k}: done", flush=True)


def report(rows):
    gts = {}
    for r in rows:
        lf = SPLIT / r["split"] / "labels" / f"{Path(r['view']).stem}.txt"
        gts[Path(r["view"]).stem] = E.ground_truth(lf, 1440, 1440)
    fold_of = {Path(r["view"]).stem: int(r["fold"]) for r in rows}
    allp, per = {}, []
    lines = ["5-fold cross-validation, local YOLO11n boll detector, clean photo split (no photo in two sets)",
             f"Training per fold: 3 folds (~84 pictures), {EPOCHS} epochs, {IMGSZ} px; valid = next fold; test = the fold.",
             "Scoring A (as eval_roboflow_detect.py): AP@50 101-point, duplicates merged at overlap 0.5.", "",
             f"{'fold':<8}{'views':>6}{'mAP50':>8}{'mAP50-95':>10}{'open AP50':>11}{'closed AP50':>13}{'P':>7}{'R':>7}"]
    for k in range(FOLDS):
        pf = CV / f"fold{k}" / "predictions.json"
        if not pf.exists():
            lines.append(f"fold {k:<3}  not trained yet"); continue
        p = {s: [(c, sc, tuple(b)) for c, sc, b in v] for s, v in json.loads(pf.read_text()).items()}
        allp.update(p)
        g = {s: gts[s] for s in gts if fold_of[s] == k}
        _, s = E.evaluate(p, g)
        per.append(s)
        lines.append(f"fold {k:<3}{len(g):>6}{100 * s['map50']:>7.1f}%{100 * s['map']:>9.1f}%"
                     f"{100 * s['cls']['open_boll']['ap50']:>10.1f}%{100 * s['cls']['closed_boll']['ap50']:>12.1f}%"
                     f"{100 * s['P']:>6.1f}%{100 * s['R']:>6.1f}%")
    if per:
        m = lambda key: (100 * np.mean([s[key] for s in per]), 100 * np.std([s[key] for s in per]))
        lines.append(f"{'mean':<8}{'':>6}{m('map50')[0]:>7.1f}%{m('map')[0]:>9.1f}%   (+/- {m('map50')[1]:.1f} mAP50 between folds)")
        g = {s: gts[s] for s in allp}
        lines += ["", f"All {len(g)} test views pooled (each predicted by the model that never saw its photo):",
                  f"{'':4}{'classes':<22}{'bolls scored':<26}{'mAP50':>7}{'mAP50-95':>10}{'P':>7}{'R':>7}"]
        for one in (False, True):
            for px in (0, 24):
                s = R.score(allp, g, one, px)
                lines.append(f"{'':4}{'1 class (boll)' if one else '2 (open / closed)':<22}"
                             f"{'all sizes' if px == 0 else f'>= {px} px (smaller ignored)':<26}"
                             f"{100 * s['map50']:>6.1f}%{100 * s['map']:>9.1f}%{100 * s['P']:>6.1f}%{100 * s['R']:>6.1f}%")
    lines += ["", f"P / R at confidence {E.SHOW_CONF}. Not comparable with the old 18-view test, R2 or Roboflow "
                  "(those tests shared photos / copies with training)."]
    CV.mkdir(parents=True, exist_ok=True)
    (CV / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines) + f"\n\nSaved: {CV / 'report.txt'}", flush=True)


def main():
    rows = list(csv.DictReader(open(SPLIT / "split.csv", encoding="utf-8")))
    if not (len(sys.argv) > 1 and sys.argv[1] == "report"):
        for k in range(FOLDS):
            run_fold(k, rows)
    report(rows)


if __name__ == "__main__":
    main()
