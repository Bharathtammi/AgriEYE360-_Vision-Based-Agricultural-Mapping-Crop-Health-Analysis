"""
Step 2b: MISSED-BOLL REVIEW - bolls the automatic labels missed, suggested by the active local detector.

  python build_missed_review.py            # 1) run the model on the training views (cached, can resume) 2) build the page
  python build_missed_review.py write      # add the bolls you confirmed to the labels (backup first)

Suggestions: model boxes with confidence >= CONF on the training views with automatic labels, >= 24 px, that overlap
no labelled boll (IoU < 0.3) and are not a box you already rejected as 'not a boll' in step 2a.
The same boll in near-identical consecutive photos is shown once (twins get the same answer). Open bolls first.
No training. Output: agrieye_output/boll_review/missed/review.html (+ crops/, items.json, predictions.json)
Your answers: 'Download answers' saves boll_missed_answers.csv in your Downloads folder.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

import build_boll_review as B
import eval_roboflow_detect as E

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "agrieye_output" / "boll_detector" / "data_big"
MODEL = ROOT / "agrieye_output" / "boll_detector" / "boll_det" / "weights" / "best.pt"
OUT = ROOT / "agrieye_output" / "boll_review" / "missed"
REJECTED = ROOT / "agrieye_output" / "boll_review" / "open_closed" / "answers.csv"
CONF, N, IMGSZ = 0.5, 400, 1280
NAMES = ["open_boll", "closed_boll"]


def views():
    return [r for r in csv.DictReader(open(DATA / "sources.csv", encoding="utf-8")) if r["labels"].startswith("automatic")]


def predict(vs):
    f = OUT / "predictions.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    todo = [v for v in vs if v["view"] not in cache]
    if todo:
        from ultralytics import YOLO
        m = YOLO(str(MODEL))
        OUT.mkdir(parents=True, exist_ok=True)
        for i, v in enumerate(todo, 1):
            r = m.predict(str(DATA / "train" / "images" / v["view"]), imgsz=IMGSZ, conf=CONF, iou=0.5, max_det=300,
                          device="cpu", verbose=False)[0]
            cache[v["view"]] = [(int(c), round(float(s), 3), [round(x, 1) for x in b]) for b, s, c in
                                zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            if i % 25 == 0 or i == len(todo):
                f.write_text(json.dumps(cache))
                print(f"  predicted {len(cache)}/{len(vs)} views", flush=True)
    return cache


def rejected_boxes():
    """Boxes you called 'not a boll' in step 2a, per view (so they are not suggested again)."""
    if not REJECTED.exists():
        return {}
    sam = {}
    for line in open(B.LAB / "sam3.jsonl", encoding="utf-8"):
        r = json.loads(line); sam[Path(r["view"]).name] = r["bolls"]
    out = defaultdict(list)
    for r in csv.DictReader(open(REJECTED, encoding="utf-8")):
        if r["new"] == "not_boll" and r["view"] in sam:
            out[r["view"]].append(sam[r["view"]][int(r["k"])]["box"])
    return out


def candidates(vs, preds):
    rej = rejected_boxes()
    out = []
    for v in vs:
        img = DATA / "train" / "images" / v["view"]
        lab = E.ground_truth(DATA / "train" / "labels" / f"{Path(v['view']).stem}.txt", 1440, 1440)
        for c, s, b in preds.get(v["view"], []):
            if max(b[2] - b[0], b[3] - b[1]) < 24:
                continue
            if any(E.iou(b, g) >= 0.3 for _, g in lab) or any(E.iou(b, x) >= 0.5 for x in rej.get(v["view"], [])):
                continue
            stem = Path(v["view"]).stem
            out.append(dict(view=str(img), name=v["view"], date=v["date"], direction=stem.split("_")[-1],
                            file=stem.rsplit("_", 1)[0] + ".jpg", k=-1, box=b, label=NAMES[c], conf=s,
                            pts=2 if c == 0 else 1,
                            why=f"model: {'open' if c == 0 else 'closed'} {100 * s:.0f}%, no label here"))
    return out


SKIP = []


def drop_seen(c):
    seen = defaultdict(list)
    for d in SKIP:
        f = ROOT / "agrieye_output" / "boll_review" / d / "items.json"
        for x in json.loads(f.read_text()):
            seen[x["name"]].append(x["box"])
            for t in x["twins"]:
                seen[t[0]].append(t[3])
    return [x for x in c if not any(E.iou(x["box"], b) >= 0.5 for b in seen.get(x["name"], []))]


def pick(c):
    """Open first, most confident first; same boll in near-identical photos grouped as twins."""
    out = []
    for x in sorted(c, key=lambda x: (-x["pts"], -x["conf"])):
        twin = next((o for o in out if B.same_boll(o, x)), None)
        if twin:
            twin["twins"].append(x)
        elif len(out) < N:
            x["twins"] = []; out.append(x)
    return out


def build():
    vs = views()
    preds = predict(vs)
    c = drop_seen(candidates(vs, preds))
    items = pick(c)
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    for i, x in enumerate(items):
        B.crop(x, OUT / "crops" / f"{i}.jpg")
    keep = [dict({k: x[k] for k in ("name", "file", "direction", "k", "label", "conf", "why")}, box=x["box"],
                 twins=[[t["name"], t["file"], t["direction"], t["box"], t["label"]] for t in x["twins"]]) for x in items]
    (OUT / "items.json").write_text(json.dumps(keep), encoding="utf-8")
    page = (B.PAGE.replace("__ITEMS__", json.dumps(keep))
            .replace('KEY="boll_review_v1"', f'KEY="boll_{OUT.name}_v1"')
            .replace("Boll review: is the yellow-boxed boll OPEN or CLOSED?",
                     "Missed bolls: is there a boll in the yellow box? Open or closed?")
            .replace("The current label is already selected.", "The model's guess is already selected.")
            .replace("boll_review_answers.csv", f"boll_{OUT.name}_answers.csv")
            .replace("x.twins.forEach(t=>rows.push([i,t[0],t[1],t[2],t[3],t[4],ans(i),'same boll as #'+(i+1)]));",
                     "x.twins.forEach(t=>rows.push([i,t[0],t[1],t[2],JSON.stringify(t[3]),t[4],ans(i),'same boll as #'+(i+1)]));")
            .replace("rows.push([i,x.name,x.file,x.direction,x.k,x.label,ans(i),x.why]);",
                     "rows.push([i,x.name,x.file,x.direction,JSON.stringify(x.box),x.label,ans(i),x.why]);")
            .replace("['id','view','file','direction','k','old','new','why']",
                     "['id','view','file','direction','box','old','new','why']"))
    (OUT / "review.html").write_text(page, encoding="utf-8")
    n_open = sum(x["label"] == "open_boll" for x in items)
    print(f"Suggested missed bolls: {len(c)} boxes -> {len(items)} cards "
          f"(+ {sum(len(x['twins']) for x in items)} copies in near-identical photos); open {n_open}, closed {len(items) - n_open}")
    print(f"Open in your browser: {OUT / 'review.html'}")


def write():
    import shutil
    from datetime import datetime
    f = OUT / "answers.csv"
    src = Path.home() / "Downloads" / f"boll_{OUT.name}_answers.csv"
    if src.exists() and (not f.exists() or src.stat().st_mtime > f.stat().st_mtime):
        shutil.copy2(src, f)
    rows = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["new"] in NAMES]
    dirs = [B.LAB / "bolls_dataset" / "labels", DATA / "train" / "labels"]
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    for d in dirs:
        shutil.copytree(d, d.parent / f"labels_before_missed_{stamp}")
    by = defaultdict(list)
    for r in rows:
        by[r["view"]].append(r)
    added = defaultdict(int)
    for d in dirs:
        for view, rs in by.items():
            lf = d / f"{Path(view).stem}.txt"
            if not lf.exists():
                continue
            old = [l for l in lf.read_text().splitlines() if l.strip()]
            have = E.ground_truth(lf, 1440, 1440)
            for r in rs:
                x1, y1, x2, y2 = json.loads(r["box"])
                if any(E.iou((x1, y1, x2, y2), g) >= 0.5 for _, g in have):
                    continue                              # already there (e.g. added twice)
                c = NAMES.index(r["new"])
                old.append(f"{c} {(x1 + x2) / 2 / 1440:.5f} {(y1 + y2) / 2 / 1440:.5f} {(x2 - x1) / 1440:.5f} {(y2 - y1) / 1440:.5f}")
                have.append((c, (x1, y1, x2, y2)))
                added[(d.parent.name if d.parent.name != "train" else "data_big", r["new"])] += 1
            lf.write_text(chr(10).join(old) + chr(10), encoding="utf-8")
    for k, v in sorted(added.items()):
        print(f"  {k[0]:<14} added {k[1]:<12}{v:>5}")
    print(f"  backups: labels_before_missed_{stamp} next to each labels folder;  answers kept in {f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="build", choices=["build", "write"])
    ap.add_argument("--model", default="boll_det", help="run folder in agrieye_output/boll_detector")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--out", default="missed", help="output folder in agrieye_output/boll_review")
    ap.add_argument("--skip", nargs="*", default=[], help="earlier review folders: their suggestions are not shown again")
    a = ap.parse_args()
    MODEL = ROOT / "agrieye_output" / "boll_detector" / a.model / "weights" / "best.pt"
    CONF, OUT = a.conf, ROOT / "agrieye_output" / "boll_review" / a.out
    SKIP = a.skip
    write() if a.cmd == "write" else build()
