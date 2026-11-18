"""
R1 (your Roboflow boll classifier, 97.7%) judges every SAM 3 boll box of ALL views: open / closed / not a boll.
Step 2 of method A, after label_all_povs.py. Run it yourself in cmd (it needs your private key):

  set ROBOFLOW_API_KEY=<your private API key>          (cmd; the key is never saved)
  python label_all_bolls_r1.py [--model cotton-boll-open-or-close-3frah/3]

Crop = the same cut as R1's training data (box + 15% margin, from the 4-direction view).
Exact copies are not sent again (they reuse the original's answers). Resumable: run again to continue.
Input : agrieye_output/pov_all_labels/sam3.jsonl
Output: agrieye_output/pov_all_labels/boll_r1.csv   file, direction, k, pred, conf
Cost  : about 0.06 Roboflow credits per 1,000 crops.
"""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

from eval_roboflow_boll import top_class

ROOT = Path(__file__).resolve().parent
LAB = ROOT / "agrieye_output" / "pov_all_labels"
OUT = LAB / "boll_r1.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cotton-boll-open-or-close-3frah/3")
    ap.add_argument("--api", default="https://serverless.roboflow.com")
    a = ap.parse_args()
    key = (os.environ.get("ROBOFLOW_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise SystemExit("Set your key first (cmd):  set ROBOFLOW_API_KEY=xxxxxxxx")
    from inference_sdk import InferenceHTTPClient
    client = InferenceHTTPClient(api_url=a.api, api_key=key)
    recs = [json.loads(l) for l in (LAB / "sam3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    originals = [r for r in recs if not r.get("copy_of")]
    done = {}
    if OUT.exists():
        done = {(r["file"], r["direction"], int(r["k"])): r for r in csv.DictReader(open(OUT, encoding="utf-8"))}
    todo = [(r, k, d) for r in originals for k, d in enumerate(r["bolls"]) if (r["file"], r["direction"], k) not in done]
    print(f"{len(done)} boll crops already judged, {len(todo)} to go ({len(originals)} original views; model {a.model})")
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "direction", "k", "pred", "conf"])
        if new:
            w.writeheader()
        cur, im = None, None
        for n, (r, k, d) in enumerate(todo, 1):
            if r["view"] != cur:
                cur, im = r["view"], Image.open(r["view"]).convert("RGB")
            x1, y1, x2, y2 = d["box"]
            m = 0.15 * max(x2 - x1, y2 - y1)
            box = (int(max(0, x1 - m)), int(max(0, y1 - m)), int(min(im.width, x2 + m)), int(min(im.height, y2 + m)))
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            try:
                pred, conf = top_class(client.infer(im.crop(box), model_id=a.model))
            except Exception as e:
                print(f"  stopped at {n}/{len(todo)}: {str(e).split('?api_key')[0][:300]}\n  run again to continue.")
                break
            row = dict(file=r["file"], direction=r["direction"], k=k, pred=pred, conf=round(conf, 4))
            w.writerow(row); fh.flush(); done[(r["file"], r["direction"], k)] = row
            if n % 200 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)
    cnt = Counter(r["pred"] for r in done.values())
    low = sum(float(r["conf"]) < 0.6 for r in done.values())
    print(f"\nJudged boll crops: {len(done)}   {dict(cnt)}   low confidence (<0.6): {low}\nSaved: {OUT}")


if __name__ == "__main__":
    main()
