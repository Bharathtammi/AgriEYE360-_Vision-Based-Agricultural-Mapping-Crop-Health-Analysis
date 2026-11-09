"""
Experiment: can SAM 3 tell diseased from healthy leaves BY ITSELF (text prompts only)?

Test set = the same 835 hand-labelled leaves the disease model v2 is tested on
(disease_autolabel/dataset_cls/val, 7 photos). SAM 3 sees the same 1280 px tiles of those photos and
is asked for each prompt below. Every hand-labelled leaf is matched to SAM 3's masks (box IoU >= 0.5);
the leaf's score for a prompt = best matching SAM 3 score (0 if SAM 3 did not pick it).

  python sam3_disease_test.py run       # SAM 3 on the tiles (~15-30 min CPU), saves raw output
  python sam3_disease_test.py report    # scores vs your labels, compared with model v2

Output: sample_results/sam3_disease_test/  (raw.jsonl, report.csv, examples.jpg)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import pipeline as P
from sam3_autolabel import box_iou, build_predictor, make_tiles

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sample_results" / "sam3_disease_test"
RAW = OUT / "raw.jsonl"
PROMPTS = ["diseased leaf", "leaf with brown spots", "healthy leaf", "leaf"]
TILE, OVERLAP = 1280, 0.2


def truth():
    lab = {p.name: c for c in ("healthy", "diseased") for p in (P.WORK / "dataset_cls" / "val" / c).glob("*.png")}
    leaves = defaultdict(list)
    for l in P.MANIFEST.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("crop") in lab:
                leaves[r["source"]].append(dict(crop=r["crop"], box=r["box"], truth=lab[r["crop"]]))
    return leaves


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if RAW.exists():
        done = {json.loads(l)["source"] for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()}
    pred, _ = build_predictor(ROOT / "sam3.pt", 0.1, "cpu")
    for src, lv in sorted(truth().items()):
        if src in done:
            continue
        img = Image.open(src).convert("RGB")
        W, H = img.size
        tiles = [t for t in make_tiles(W, H, TILE, OVERLAP)
                 if any(t[0] <= (l["box"][0] + l["box"][2]) / 2 < t[2] and t[1] <= (l["box"][1] + l["box"][3]) / 2 < t[3]
                        for l in lv)]
        dets = []
        for k, (x0, y0, x1, y1) in enumerate(tiles, 1):
            arr = np.ascontiguousarray(np.asarray(img.crop((x0, y0, x1, y1)))[:, :, ::-1])
            pred.set_image(arr)
            for pi, prompt in enumerate(PROMPTS):          # one prompt at a time: independent scores
                res = pred(text=[prompt])
                res = res[0] if isinstance(res, (list, tuple)) else res
                if res.boxes is None:
                    continue
                for b, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                    dets.append(dict(p=pi, s=round(float(s), 4),
                                     box=[float(b[0] + x0), float(b[1] + y0), float(b[2] + x0), float(b[3] + y0)]))
            print(f"  {Path(src).name}: tile {k}/{len(tiles)}", flush=True)
        with RAW.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(source=src, dets=dets)) + "\n")
        print(f"{Path(src).name}: {len(lv)} labelled leaves, {len(tiles)} tiles, {len(dets)} SAM 3 masks")


def auc(y, s):
    order = np.argsort(-np.asarray(s), kind="mergesort")
    y = np.asarray(y)[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    tpr = np.r_[0, tp / max(1, y.sum())]
    fpr = np.r_[0, fp / max(1, (1 - y).sum())]
    return float(np.trapezoid(tpr, fpr))


def report():
    raw = {json.loads(l)["source"]: json.loads(l)["dets"] for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = []
    for src, lv in truth().items():
        if src not in raw:
            continue
        for l in lv:
            sc = [0.0] * len(PROMPTS)
            for d in raw[src]:
                if box_iou(l["box"], d["box"]) >= 0.5:
                    sc[d["p"]] = max(sc[d["p"]], d["s"])
            rows.append(dict(l, source=src, **{PROMPTS[i]: sc[i] for i in range(len(PROMPTS))}))
    y = [r["truth"] == "diseased" for r in rows]
    nd = sum(y)
    print(f"\n{len(rows)} hand-labelled leaves ({nd} diseased, {len(rows) - nd} healthy) from {len(raw)} photos\n")
    print(f"{'SAM 3 rule':48s} {'acc':>6s} {'dis.recall':>10s} {'dis.prec':>8s} {'AUC':>6s}")
    out = []

    def show(name, pred, score=None):
        tp = sum(p and t for p, t in zip(pred, y)); fp = sum(p and not t for p, t in zip(pred, y))
        fn = sum((not p) and t for p, t in zip(pred, y)); tn = len(y) - tp - fp - fn
        a = auc(y, score) if score is not None else float("nan")
        r = dict(rule=name, accuracy=round(100 * (tp + tn) / len(y), 1), diseased_recall=round(100 * tp / max(1, tp + fn), 1),
                 diseased_precision=round(100 * tp / max(1, tp + fp), 1), auc=round(a, 3), tp=tp, fp=fp, fn=fn, tn=tn)
        out.append(r)
        print(f"{name:48s} {r['accuracy']:5.1f}% {r['diseased_recall']:9.1f}% {r['diseased_precision']:7.1f}% {a:6.3f}")

    print("Did SAM 3 even find the leaf?  'leaf' prompt matched: "
          f"{sum(r['leaf'] > 0 for r in rows)} / {len(rows)}\n")
    for pr in ("diseased leaf", "leaf with brown spots"):
        s = [r[pr] for r in rows]
        show(f'"{pr}" picked it (score >= 0.3)', [v >= 0.3 for v in s], s)
        show(f'"{pr}" picked it (score >= 0.1)', [v >= 0.1 for v in s], s)
    s = [max(r["diseased leaf"], r["leaf with brown spots"]) - r["healthy leaf"] for r in rows]
    show('diseased-type score > "healthy leaf" score', [v > 0 for v in s], s)
    show("(reference) always say healthy", [False] * len(y))
    print(f"\n{'Disease model v2 (trained on your labels)':48s} {89.2:5.1f}% {82.7:9.1f}% {85.7:7.1f}% {0.944:6.3f}")
    import csv
    with open(OUT / "report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    with open(OUT / "per_leaf.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["crop", "truth"] + PROMPTS); w.writeheader()
        w.writerows({k: r[k] for k in ["crop", "truth"] + PROMPTS} for r in rows)
    # examples: SAM 3's most "diseased" leaves and the true diseased leaves it scored lowest
    T = 130
    top = sorted(rows, key=lambda r: -r["diseased leaf"])[:24]
    low = sorted([r for r in rows if r["truth"] == "diseased"], key=lambda r: r["diseased leaf"])[:24]
    sheet = Image.new("RGB", (12 * (T + 4), 2 * (2 * (T + 30)) + 60), (235, 235, 235))
    dr = ImageDraw.Draw(sheet)
    for gi, (title, grp) in enumerate([('SAM 3 "diseased leaf" - highest scores (your label underneath)', top),
                                       ("Leaves YOU labelled diseased that SAM 3 scored lowest", low)]):
        y0 = gi * (2 * (T + 30) + 30)
        dr.text((6, y0 + 8), title, fill=(0, 0, 0))
        for i, r in enumerate(grp):
            im = Image.open(next((P.CROPS / c / r["crop"]) for c in ("diseased", "healthy")
                                 if (P.CROPS / c / r["crop"]).exists())).convert("RGB")
            im.thumbnail((T, T))
            x, yy = (i % 12) * (T + 4) + 2, y0 + 28 + (i // 12) * (T + 30)
            sheet.paste(im, (x, yy))
            dr.text((x, yy + T + 2), f"SAM {r['diseased leaf']:.2f} / you: {r['truth']}", fill=(0, 0, 0))
    sheet.save(OUT / "examples.jpg", quality=88)
    print(f"\nReport: {OUT / 'report.csv'}\nExamples: {OUT / 'examples.jpg'}")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
