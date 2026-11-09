"""
Test the boll model trained on Roboflow on the TEST split (images it never saw in training).

  set ROBOFLOW_API_KEY=<your private API key>        (cmd; the key is not saved anywhere)
  python eval_roboflow_boll.py [--model cotton-boll-open-or-close-3frah/3] [--split test]

Sends each crop of roboflow_export/2_cotton_boll_classification/<split>/<class>/ to the model on Roboflow's
hosted API (about 0.06 credits per 1000 images) and writes, in sample_results/roboflow_boll_<split>/:
  predictions.csv      one row per image: true class, predicted class, confidence (resumable cache)
  report.txt           accuracy, confusion matrix, precision / recall per class
  mistakes.jpg         every wrong prediction (true -> predicted)
"""
import argparse
import csv
import os
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "roboflow_export" / "2_cotton_boll_classification"
CLASSES = ["open_boll", "closed_boll", "not_boll"]


def top_class(res):
    """Roboflow classification answer -> (class, confidence) for single- or multi-label responses."""
    if isinstance(res, list):
        res = res[0]
    if res.get("top"):
        return res["top"], float(res.get("confidence", 0))
    p = res.get("predictions")
    if isinstance(p, list) and p:
        b = max(p, key=lambda x: x.get("confidence", 0))
        return b.get("class"), float(b.get("confidence", 0))
    if isinstance(p, dict) and p:
        k = max(p, key=lambda k: p[k].get("confidence", 0))
        return k, float(p[k]["confidence"])
    return "?", 0.0


def find_models(key):
    """Ask Roboflow which workspace this key belongs to and list the boll projects + versions (ids to use)."""
    import requests
    r = requests.get("https://api.roboflow.com/", params={"api_key": key}, timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"Roboflow did not accept this key ({r.status_code}): {r.text[:300]}\n"
                         "Copy the PRIVATE API key again from https://app.roboflow.com/settings/api")
    ws = r.json().get("workspace")
    print(f"Key OK - workspace: {ws}")
    j = requests.get(f"https://api.roboflow.com/{ws}", params={"api_key": key}, timeout=30).json()
    projs = (j.get("workspace") or {}).get("projects", [])
    for p in projs:
        pid = p.get("id", "").split("/")[-1]
        try:
            pv = requests.get(f"https://api.roboflow.com/{ws}/{pid}", params={"api_key": key}, timeout=30).json()
            vers = [(str(v.get("id", "")).split("/")[-1], bool(v.get("model"))) for v in pv.get("versions", [])]
        except Exception:
            vers = []
        print(f"  project {pid}: versions " + ", ".join(f"{v}{' (has model)' if m else ''}" for v, m in vers))
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto", help="Roboflow model id; 'auto' tries the known forms")
    ap.add_argument("--split", default="test")
    ap.add_argument("--api", default="https://serverless.roboflow.com")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick check)")
    args = ap.parse_args()
    key = (os.environ.get("ROBOFLOW_API_KEY") or "").strip().strip('"')
    if not key:
        raise SystemExit("Set your key first (cmd):  set ROBOFLOW_API_KEY=xxxxxxxx")
    print(f"Key in use: {key[:2]}...{key[-2:]}  ({len(key)} characters)")
    if key.startswith("rf_"):
        print("(Warning: keys starting with rf_ are usually the PUBLISHABLE key, which cannot run models.\n"
              " Checking with Roboflow anyway ...)")
    find_models(key)
    from inference_sdk import InferenceHTTPClient
    client = InferenceHTTPClient(api_url=args.api, api_key=key)

    out = ROOT / "sample_results" / f"roboflow_boll_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "predictions.csv"
    done = {}
    if cache.exists():
        done = {r["file"]: r for r in csv.DictReader(open(cache, encoding="utf-8"))}
    files = [(f, c) for c in CLASSES for f in sorted((DATA / args.split / c).glob("*.jpg"))]
    if args.limit:
        files = files[:args.limit]

    # Roboflow accepts different id forms depending on the model type: try them on one image
    cands = [args.model] if args.model != "auto" else [
        "cotton-boll-open-or-close-3frah/3",
        "cotton-boll-open-or-close-3frah-3-vit-base-patch16-224-in21k-t1",
        "bharaths-workspace-3fhlw/cotton-boll-open-or-close-3frah-3-vit-base-patch16-224-in21k-t1"]
    for m in cands:
        try:
            print(f"trying model id {m} ... {top_class(client.infer(str(files[0][0]), model_id=m))}")
            args.model = m
            break
        except Exception as e:
            print(f"  not this one: {str(e).split('?api_key')[0][:400]}")
    else:
        raise SystemExit("None of the model ids worked - send the messages above (without your key).")
    new = not cache.exists()
    with open(cache, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "true", "pred", "conf"])
        if new:
            w.writeheader()
        for k, (f, c) in enumerate(files, 1):
            rel = f"{c}/{f.name}"
            if rel in done:
                continue
            pred, conf = top_class(client.infer(str(f), model_id=args.model))
            row = dict(file=rel, true=c, pred=pred, conf=round(conf, 4))
            w.writerow(row); fh.flush(); done[rel] = row
            if k % 25 == 0 or k == len(files):
                print(f"  {k}/{len(files)}", flush=True)

    rows = [done[f"{c}/{f.name}"] for f, c in files]
    n = len(rows)
    cm = Counter((r["true"], r["pred"]) for r in rows)
    acc = sum(r["true"] == r["pred"] for r in rows) / max(1, n)
    lines = [f"Roboflow model {args.model} on the {args.split} split: {n} images (never used in training)",
             f"Accuracy: {100 * acc:.1f}%", "", "Confusion matrix (rows = true, columns = predicted):",
             f"{'':14s}" + "".join(f"{c:>13s}" for c in CLASSES)]
    for t in CLASSES:
        lines.append(f"{t:14s}" + "".join(f"{cm[(t, p)]:13d}" for p in CLASSES))
    lines += ["", f"{'class':14s}{'precision':>10s}{'recall':>10s}{'n':>6s}"]
    for c in CLASSES:
        tp = cm[(c, c)]
        npred = sum(cm[(t, c)] for t in CLASSES)
        ntrue = sum(cm[(c, p)] for p in CLASSES + ["?"])
        lines.append(f"{c:14s}{100 * tp / max(1, npred):9.1f}%{100 * tp / max(1, ntrue):9.1f}%{ntrue:6d}")
    sky = [r for r in rows if Path(r["file"]).name.startswith("sky_")]
    if sky:
        lines.append(f"\nsky / cloud patches called not_boll: {sum(r['pred'] == 'not_boll' for r in sky)} / {len(sky)}")
    report = "\n".join(lines)
    (out / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    wrong = [r for r in rows if r["true"] != r["pred"]]
    if wrong:
        T = 150
        sh = Image.new("RGB", (10 * T, ((len(wrong) + 9) // 10) * (T + 30)), (30, 30, 30))
        d = ImageDraw.Draw(sh)
        for i, r in enumerate(wrong):
            im = Image.open(DATA / args.split / r["file"]).convert("RGB"); im.thumbnail((T - 4, T - 4))
            x, y = (i % 10) * T, (i // 10) * (T + 30)
            sh.paste(im, (x + 2, y + 2))
            d.text((x + 3, y + T), f"{r['true'][:6]} -> {r['pred'][:6]}", fill=(255, 220, 0))
            d.text((x + 3, y + T + 13), f"{float(r['conf']):.2f}", fill=(200, 200, 200))
        sh.save(out / "mistakes.jpg", quality=88)
    print(f"\nFiles: {out}")


if __name__ == "__main__":
    main()
