"""
Step 1 of the 'road to 90%': RE-SCORE the active boll detector - no training, no new predictions.

  python rescore_boll_detector.py

Uses the saved predictions of the active model (sample_results/local_boll_detect_test/predictions.csv, 18 test views;
sample_results/boll_model_comparison/raw_local_boll_det.json, 24 fair views) and scores them four ways:
  2 classes (open / closed)  vs  1 class ("boll": open + closed together, duplicates across classes merged at 0.5)
  all sizes                  vs  ignore bolls smaller than MIN_PX across (too far away to judge - about 1.5 m+)
Ignoring works like COCO's size ranges: a small hand box is neither a miss nor a hit, and a small predicted box
that matches nothing is not counted as a false alarm. Bigger bolls are scored exactly as before.
Same AP formula as eval_roboflow_detect.py (scoring A), so the first row equals the existing report.
Output: sample_results/local_boll_detect_test/rescore.txt
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import eval_roboflow_detect as E

ROOT = Path(__file__).resolve().parent
TEST = ROOT / "roboflow_export" / "1_cotton_boll_detection" / "test"
OLD = ROOT / "roboflow_export" / "1_cotton_boll_detection"
PRED = ROOT / "sample_results" / "local_boll_detect_test" / "predictions.csv"
FAIR = ROOT / "agrieye_output" / "boll_detector" / "fair_views.csv"
FAIR_PRED = ROOT / "sample_results" / "boll_model_comparison" / "raw_local_boll_det.json"
OUT = ROOT / "sample_results" / "local_boll_detect_test" / "rescore.txt"
SIZES = [0, 24, 32]


def side(b):
    return max(b[2] - b[0], b[3] - b[1])


def match(preds, gts, c, t, min_px):
    rows, n = [], 0
    for img, g_all in gts.items():
        g = [(x[1], side(x[1]) < min_px or (len(x) > 2 and bool(x[2]))) for x in g_all if x[0] == c]   # x[2] = ignore zone
        n += sum(not ig for _, ig in g)
        used = [False] * len(g)
        for _, conf, box in sorted([p for p in preds.get(img, []) if p[0] == c], key=lambda p: -p[1]):
            best, bi = 0, -1
            for i, (gb, _) in enumerate(g):
                if not used[i]:
                    v = E.iou(box, gb)
                    if v > best:
                        best, bi = v, i
            if best >= t:
                used[bi] = True
                if not g[bi][1]:
                    rows.append((conf, 1))               # a hit on a small (ignored) box counts for nothing
            elif side(box) >= min_px:
                rows.append((conf, 0))                   # small unmatched predictions are ignored too
    rows.sort(key=lambda r: -r[0])
    return rows, n


def score(preds, gts, one_class, min_px, t=0.5):
    if one_class:
        preds = {k: E.nms([(0, s, b) for _, s, b in v], 0.5) for k, v in preds.items()}
        gts = {k: [(0,) + tuple(x[1:]) for x in v] for k, v in gts.items()}
    classes = [0] if one_class else [0, 1]
    ap50, ap, P, R, ngt = [], [], 0, 0, 0
    allrows = []
    for c in classes:
        rows, n = match(preds, gts, c, t, min_px)
        ap50.append(E.ap(rows, n)); allrows += rows; ngt += n
        ap.append(np.mean([E.ap(match(preds, gts, c, t, min_px)[0], n) for t in np.arange(0.5, 0.96, 0.05)]))
    allrows.sort(key=lambda r: -r[0])
    P, R, _, _ = E.pr_at(allrows, ngt, E.SHOW_CONF)
    per = {E.NAMES[c]: a for c, a in zip(classes, ap50)} if not one_class else {}
    return dict(map50=np.nanmean(ap50), map=np.nanmean(ap), P=P, R=R, n=ngt, per=per)


def load_test():
    gts, preds = {}, defaultdict(list)
    for f in sorted((TEST / "images").glob("*.jpg")):
        W, H = Image.open(f).size
        gts[f.stem] = E.ground_truth(TEST / "labels" / f"{f.stem}.txt", W, H)
    for r in csv.DictReader(open(PRED, encoding="utf-8")):
        preds[r["image"]].append((E.NAMES.index(r["class"]), float(r["confidence"]),
                                  tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))))
    return preds, gts


def load_fair():
    gts = {}
    for v in csv.DictReader(open(FAIR, encoding="utf-8")):
        f = OLD / v["hand_split"] / "images" / v["hand_image"]
        W, H = Image.open(f).size
        gts[f.stem] = E.ground_truth(OLD / v["hand_split"] / "labels" / f"{f.stem}.txt", W, H)
    raw = json.loads(FAIR_PRED.read_text())["preds"]
    return {k: [(c, s, tuple(b)) for c, s, b in v] for k, v in raw.items()}, gts


def table(title, preds, gts):
    total = sum(len(g) for g in gts.values())
    lines = [title, f"{'':4}{'classes':<22}{'bolls scored':<26}{'mAP50':>7}{'mAP50-95':>10}{'P':>7}{'R':>7}"
                    f"{'open AP50':>11}{'closed AP50':>13}"]
    for one in (False, True):
        for m in SIZES:
            s = score(preds, gts, one, m)
            size = "all sizes" if m == 0 else f">= {m} px ({total - s['n']} small ignored)"
            cls = "1 class (boll)" if one else "2 (open / closed)"
            o = f"{100 * s['per']['open_boll']:>10.1f}%" if s["per"] else f"{'-':>11}"
            c = f"{100 * s['per']['closed_boll']:>12.1f}%" if s["per"] else f"{'-':>13}"
            lines.append(f"{'':4}{cls:<22}{size:<26}{100 * s['map50']:>6.1f}%{100 * s['map']:>9.1f}%"
                         f"{100 * s['P']:>6.1f}%{100 * s['R']:>6.1f}%{o}{c}")
    return lines


def main():
    mark = ROOT / "agrieye_output" / "boll_detector" / "boll_det" / "trained_on.txt"
    which = "big-set model (25 ep)" if mark.exists() and "big" in mark.read_text() else "small-set model"
    lines = [f"Active local boll detector ({which}) - the SAME predictions, re-scored. No training.",
             "Scoring A (as eval_roboflow_detect.py): AP@50 101-point, duplicates merged at overlap 0.5.", ""]
    lines += table("A) 18 hand-checked TEST views", *load_test()) + [""]
    lines += table("B) 24 FAIR views (hand-checked key; no model trained on them)", *load_fair()) + [""]
    lines += [f"Size = the longer side of the box in the 1440 px view. A ~5 cm boll is 24 px at about 1.5 m, "
              f"32 px at about 1.1 m.", f"P / R at confidence {E.SHOW_CONF}."]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines) + f"\n\nSaved: {OUT}")


if __name__ == "__main__":
    main()
