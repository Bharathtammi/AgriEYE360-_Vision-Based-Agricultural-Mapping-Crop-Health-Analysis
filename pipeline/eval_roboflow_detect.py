"""
Test the Roboflow boll DETECTOR (R2, project "Cotton boll labelling") on the 18 test views and compute the
scores Roboflow's free plan does not show: mAP@50, mAP@50:95, precision and recall per class (open / closed).

  set ROBOFLOW_API_KEY=<your private key>          (cmd; never saved)
  python eval_roboflow_detect.py                   # finds the detection model in your workspace by itself
  python eval_roboflow_detect.py --model cotton-bolls-field-view-trbz1/1   # R2 (140 labelled views)

Two ways are tested:
  full  = the whole 1440 px view sent at once (what the Roboflow test page does)
  tiled = the view cut into 2 x 2 tiles of 720 px, results joined (small bolls look 2x bigger)
Answers are cached (raw_*.json), so a re-run costs nothing. About 90 images are sent in total.

Test data: roboflow_export/UPLOAD_cotton_bolls_labelled/test (your hand-checked boxes, never used in training).
Output: sample_results/roboflow_boll_detect_test/ (report.txt, predictions_*.csv, preview_*.jpg)
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
TEST = ROOT / "roboflow_export" / "UPLOAD_cotton_bolls_labelled" / "test"
OUT = ROOT / "sample_results" / "roboflow_boll_detect_test"
NAMES = ["open_boll", "closed_boll"]
API = "https://api.roboflow.com"
SHOW_CONF = 0.5          # Roboflow's default display threshold, used for precision / recall
OPEN_C, CLOSED_C = (0, 230, 255), (255, 0, 200)


# ---------------------------------------------------------------- model id
def find_detector(key):
    r = requests.get(API + "/", params={"api_key": key}, timeout=30)
    ws = r.json().get("workspace") if r.ok else None
    if not ws:
        sys.exit(f"Roboflow did not accept this key (HTTP {r.status_code}: {r.text[:200]}).\n"
                 f"The key you set is {len(key)} characters long and starts with '{key[:2]}'.\n"
                 "Check: 1) it is the NEW private key (the old one was revoked), 2) no quotes or spaces, "
                 "3) copied from Settings > API Keys > Private API Key.")
    projs = (requests.get(f"{API}/{ws}", params={"api_key": key}, timeout=30).json().get("workspace") or {}).get("projects", [])
    found = []
    for p in projs:
        pid = p.get("id", "").split("/")[-1]
        info = requests.get(f"{API}/{ws}/{pid}", params={"api_key": key}, timeout=30).json()
        typ = (info.get("project") or {}).get("type") or p.get("type", "")
        if "detect" not in str(typ):
            continue
        for v in info.get("versions", []):                  # the API does not always say which versions are trained
            found.append((pid, str(v.get("id", "")).split("/")[-1], (info.get("project") or {}).get("name", pid)))
    if not found:
        sys.exit("No object-detection project with a version found. Give the id: --model cotton-bolls-field-view-trbz1/1")
    boll = [f for f in found if "boll" in f[0].lower() or "boll" in f[2].lower()] or found
    pid, ver, name = sorted(boll, key=lambda f: int(f[1]) if f[1].isdigit() else 0)[-1]
    print("Detection models found: " + ", ".join(f"{p}/{v} ({n})" for p, v, n in found))
    print(f"Using: {pid}/{ver}  ({name})")
    return f"{pid}/{ver}"


# ---------------------------------------------------------------- data
def ground_truth(lab, W, H):
    out = []
    for l in lab.read_text().splitlines():
        if l.strip():
            c, x, y, w, h = l.split()
            x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
            out.append((int(c), (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return out


def parse(resp, dx=0, dy=0):
    preds = resp.get("predictions", []) if isinstance(resp, dict) else []
    out = []
    for p in preds:
        cls = p.get("class")
        if cls not in NAMES:
            continue
        x, y, w, h = p["x"] + dx, p["y"] + dy, p["width"], p["height"]
        out.append((NAMES.index(cls), float(p["confidence"]), (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def nms(preds, thr=0.5):
    keep = []
    for p in sorted(preds, key=lambda p: -p[1]):
        if all(q[0] != p[0] or iou(q[2], p[2]) < thr for q in keep):
            keep.append(p)
    return keep


# ---------------------------------------------------------------- metrics
def match(preds, gts, t):
    """preds: {img: [(c, conf, box)]}, gts: {img: [(c, box)]}. Returns per class sorted (conf, tp) and n_gt."""
    res = {c: [] for c in range(len(NAMES))}
    ngt = {c: sum(1 for g in gts.values() for gc, _ in g if gc == c) for c in res}
    for img in gts:
        for c in res:
            g = [b for gc, b in gts[img] if gc == c]
            used = [False] * len(g)
            for _, conf, box in sorted([p for p in preds.get(img, []) if p[0] == c], key=lambda p: -p[1]):
                best, bi = 0, -1
                for i, gb in enumerate(g):
                    if not used[i]:
                        v = iou(box, gb)
                        if v > best:
                            best, bi = v, i
                if best >= t:
                    used[bi] = True; res[c].append((conf, 1))
                else:
                    res[c].append((conf, 0))
    for c in res:
        res[c].sort(key=lambda r: -r[0])
    return res, ngt


def ap(rows, n):
    if n == 0:
        return float("nan")
    tp = np.cumsum([r[1] for r in rows]); fp = np.cumsum([1 - r[1] for r in rows])
    if len(rows) == 0:
        return 0.0
    rec = tp / n; prec = tp / np.maximum(tp + fp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]                   # precision envelope
    return float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0 for r in np.linspace(0, 1, 101)]))


def pr_at(rows, n, conf):
    sel = [r for r in rows if r[0] >= conf]
    tp = sum(r[1] for r in sel)
    return (tp / len(sel) if sel else 0.0), (tp / n if n else 0.0), tp, len(sel) - tp


def evaluate(preds, gts):
    lines, summ = [], {}
    r50, ngt = match(preds, gts, 0.5)
    aps50 = {c: ap(r50[c], ngt[c]) for c in r50}
    aps = {c: np.mean([ap(match(preds, gts, t)[0][c], ngt[c]) for t in np.arange(0.5, 0.96, 0.05)]) for c in r50}
    for c in r50:
        p, r, tp, fp = pr_at(r50[c], ngt[c], SHOW_CONF)
        best = max(((pr_at(r50[c], ngt[c], t)[:2], t) for t in np.arange(0.05, 0.96, 0.05)),
                   key=lambda x: 2 * x[0][0] * x[0][1] / max(x[0][0] + x[0][1], 1e-9))
        (bp, br), bt = best
        lines.append(f"  {NAMES[c]:12s} boxes {ngt[c]:4d}   AP@50 {100 * aps50[c]:5.1f}%   AP@50:95 {100 * aps[c]:5.1f}%   "
                     f"at conf {SHOW_CONF}: precision {100 * p:5.1f}%  recall {100 * r:5.1f}%  (found {tp}, false {fp})   "
                     f"best conf {bt:.2f}: P {100 * bp:.1f}% R {100 * br:.1f}%")
        summ[NAMES[c]] = dict(ap50=aps50[c], ap=aps[c], p=p, r=r)
    m50, m = np.nanmean(list(aps50.values())), np.nanmean(list(aps.values()))
    allrows = sorted(r50[0] + r50[1], key=lambda r: -r[0])
    P, R, _, _ = pr_at(allrows, sum(ngt.values()), SHOW_CONF)
    head = (f"  mAP@50 {100 * m50:.1f}%   mAP@50:95 {100 * m:.1f}%   "
            f"all bolls at conf {SHOW_CONF}: precision {100 * P:.1f}%  recall {100 * R:.1f}%")
    return [head] + lines, dict(map50=m50, map=m, P=P, R=R, cls=summ)


# ---------------------------------------------------------------- drawing
def dashed(d, b, col, width, dash=12, gap=8):
    x1, y1, x2, y2 = b
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        L = max(abs(bx - ax), abs(by - ay)); n = int(L // (dash + gap)) + 1
        for k in range(n):
            s, e = k * (dash + gap) / max(L, 1), min(1, (k * (dash + gap) + dash) / max(L, 1))
            d.line([(ax + (bx - ax) * s, ay + (by - ay) * s), (ax + (bx - ax) * e, ay + (by - ay) * e)], fill=col, width=width)


def preview(img_path, gt, pred, title, path):
    im = Image.open(img_path).convert("RGB"); d = ImageDraw.Draw(im)
    for _, b in gt:                                                     # your labels: thin white
        d.rectangle(b, outline=(255, 255, 255), width=2)
    for c, conf, b in pred:
        if conf < SHOW_CONF:
            continue
        if c == 0:
            d.rectangle([b[0] - 3, b[1] - 3, b[2] + 3, b[3] + 3], outline=(0, 0, 0), width=8); d.rectangle(b, outline=OPEN_C, width=4)
        else:
            dashed(d, [b[0] - 3, b[1] - 3, b[2] + 3, b[3] + 3], (0, 0, 0), 7); dashed(d, b, CLOSED_C, 4)
    f = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 26)
    d.rectangle([0, 0, im.width, 50], fill=(0, 0, 0))
    d.text((12, 11), title, font=f, fill=(255, 255, 255))
    im.save(path, quality=90)


# ---------------------------------------------------------------- main
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--model", default="auto")
    ap_.add_argument("--api", default="https://serverless.roboflow.com")
    a = ap_.parse_args()
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip().strip('"').strip("'").strip()
    if not key:
        sys.exit("Set your key first (cmd):  set ROBOFLOW_API_KEY=xxxxxxxx")
    if key.startswith("rf_"):
        sys.exit("That is the PUBLISHABLE key (rf_...). Use the PRIVATE key from Settings > API Keys.")
    model = find_detector(key) if a.model == "auto" else a.model
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    client = InferenceHTTPClient(api_url=a.api, api_key=key)
    client.configure(InferenceConfiguration(confidence_threshold=0.01, iou_threshold=0.5, max_detections=1000))
    OUT.mkdir(parents=True, exist_ok=True)
    cache_f = OUT / f"raw_{model.replace('/', '_')}.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}

    def infer(k, image):
        if k not in cache:
            try:
                cache[k] = client.infer(image, model_id=model)
            except Exception as e:
                sys.exit(f"Roboflow could not run model '{model}': {str(e)[:300]}\n"
                         "This usually means that version has NO trained model (or the id is wrong). "
                         "Check Roboflow > project > Models; if no model is listed there, R2 was never trained.")
            cache_f.write_text(json.dumps(cache))
        return cache[k]

    imgs = sorted((TEST / "images").glob("*.jpg"))
    gts, full, tiled = {}, {}, {}
    for n, f in enumerate(imgs, 1):
        im = Image.open(f).convert("RGB"); W, H = im.size
        gts[f.stem] = ground_truth(TEST / "labels" / f"{f.stem}.txt", W, H)
        full[f.stem] = parse(infer(f"full/{f.stem}", str(f)))
        tp = []
        for ty in (0, H // 2):
            for tx in (0, W // 2):
                tile = im.crop((tx, ty, tx + W // 2, ty + H // 2))
                tp += parse(infer(f"tile/{f.stem}/{tx}_{ty}", np.array(tile)[:, :, ::-1].copy()), tx, ty)
        tiled[f.stem] = nms(tp)
        print(f"  {n:2d}/{len(imgs)} {f.stem}: your boxes {len(gts[f.stem])}, model full {sum(p[1] >= SHOW_CONF for p in full[f.stem])}, "
              f"tiled {sum(p[1] >= SHOW_CONF for p in tiled[f.stem])} (conf >= {SHOW_CONF})")

    n_open = sum(c == 0 for g in gts.values() for c, _ in g); n_closed = sum(c == 1 for g in gts.values() for c, _ in g)
    lines = [f"Roboflow boll detector (R2) {model} on {len(imgs)} test views never used in training",
             f"Your hand-checked boxes: open {n_open}, closed {n_closed}.  IoU 0.5 = a box counts as found if it overlaps "
             f"your box by at least 50%.", ""]
    best = None
    for name, preds in (("WHOLE VIEW (as on the Roboflow test page)", full), ("2 x 2 TILES (small bolls look 2x bigger)", tiled)):
        rows, s = evaluate(preds, gts)
        lines += [name] + rows + [""]
        best = best if best and best[1]["map50"] >= s["map50"] else (name, s)
        with open(OUT / f"predictions_{'full' if preds is full else 'tiled'}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2"])
            for img, ps in preds.items():
                for c, conf, b in ps:
                    w.writerow([img, NAMES[c], round(conf, 3)] + [round(v, 1) for v in b])
    lines += [f"Better way: {best[0].split(' (')[0].lower()}  (mAP@50 {100 * best[1]['map50']:.1f}%)",
              "Precision = of the boxes the model drew, how many are real bolls of that class.",
              "Recall    = of your boxes, how many the model found with the right class."]
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    # previews: the 2 views with the most open bolls
    top = sorted(gts, key=lambda k: -sum(c == 0 for c, _ in gts[k]))[:2]
    use = tiled if best[1] is not None and best[0].startswith("2 x 2") else full
    for k in top:
        preview(TEST / "images" / f"{k}.jpg", gts[k], use[k],
                f"{k}   model: cyan = open, magenta dashed = closed   white = your labels", OUT / f"preview_{k}.jpg")
    print(f"\nFiles: {OUT}")


if __name__ == "__main__":
    main()
