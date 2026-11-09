"""
Test every trained model on held-out data and write one summary.

  python test_all_models.py
Output: sample_results/model_tests/all_models_summary.csv (+ printed table)

Models (only those that exist are tested):
  disease classifier v1 / v2 / v3   on field test leaves (835, never trained on) and outside test leaves
  leaf-finder (YOLO11n-seg)         on its 144 validation tiles (box + mask precision / recall / mAP)
  boll classifier                   on its validation crops (if trained)
"""
import csv
from pathlib import Path

import pipeline as P

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sample_results" / "model_tests"
DISEASE = {
    "disease classifier v1": P.MODELS / "disease_cls_v1_backup" / "weights" / "best.pt",
    "disease classifier v2 (current)": P.CLS_MODEL,
    "disease classifier v3 (field+outside)": P.MODELS / "disease_cls_v3" / "weights" / "best.pt",
}
BOLL = P.MODELS / "boll_cls" / "weights" / "best.pt"


def cls_sets():
    v3 = P.WORK / "dataset_cls_v3" / "val"
    field = [(f, c) for c in ("healthy", "diseased") for f in (P.WORK / "dataset_cls" / "val" / c).glob("*.png")]
    outside = [(f, c) for c in ("healthy", "diseased") for f in (v3 / c).glob("ext_*.png")] if v3.exists() else []
    return {"field test leaves": field, "outside test leaves": outside}


def score(model, files, pos="diseased"):
    from PIL import Image
    pred = []
    for i in range(0, len(files), 64):
        pred += P.classify(model, [Image.open(f).convert("RGB") for f, _ in files[i:i + 64]])
    y = [t == pos for _, t in files]
    p = [l == pos for l, _ in pred]
    s = [q for _, q in pred]
    tp = sum(a and b for a, b in zip(p, y)); fp = sum(a and not b for a, b in zip(p, y))
    fn = sum((not a) and b for a, b in zip(p, y)); tn = len(y) - tp - fp - fn
    import numpy as np
    o = np.argsort(-np.asarray(s)); yy = np.asarray(y)[o]
    tpr = np.r_[0, np.cumsum(yy) / max(1, yy.sum())]; fpr = np.r_[0, np.cumsum(1 - yy) / max(1, (1 - yy).sum())]
    return dict(n=len(y), accuracy=100 * (tp + tn) / max(1, len(y)), recall=100 * tp / max(1, tp + fn),
                precision=100 * tp / max(1, tp + fp), healthy_ok=100 * tn / max(1, tn + fp),
                auc=float(np.trapezoid(tpr, fpr)))


def main():
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    sets = cls_sets()
    for name, w in DISEASE.items():
        if not w.exists():
            print(f"(skip {name}: not trained)")
            continue
        m = YOLO(str(w))
        for sn, fs in sets.items():
            if not fs:
                continue
            r = score(m, fs)
            rows.append(dict(model=name, test_set=sn, items=r["n"], accuracy=round(r["accuracy"], 1),
                             precision=round(r["precision"], 1), recall=round(r["recall"], 1),
                             healthy_correct=round(r["healthy_ok"], 1), auc_or_map50=round(r["auc"], 3), map50_95=""))
            print(rows[-1])
    if P.LEAF_MODEL.exists():
        m = YOLO(str(P.LEAF_MODEL))
        res = m.val(data=str(P.LEAF_DATA / "data.yaml"), imgsz=P.IMGSZ, batch=8, device="cpu", plots=False,
                    verbose=False, project=str(OUT), name="leaf_finder_val", exist_ok=True)
        for part, k in (("boxes", res.box), ("masks", res.seg)):
            rows.append(dict(model="leaf-finder (YOLO11n-seg)", test_set=f"144 val tiles - {part}",
                             items=4671, accuracy="", precision=round(100 * float(k.mp), 1),
                             recall=round(100 * float(k.mr), 1), healthy_correct="",
                             auc_or_map50=round(float(k.map50), 3), map50_95=round(float(k.map), 3)))
            print(rows[-1])
    if BOLL.exists():
        print("boll classifier: test with its own validation set (not implemented here yet)")
    else:
        print("(skip boll classifier: not trained)")
    with open(OUT / "all_models_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSummary: {OUT / 'all_models_summary.csv'}")


if __name__ == "__main__":
    main()
