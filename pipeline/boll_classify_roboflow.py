"""
Classify every SAM 3 boll box (open / closed / not a boll) with YOUR boll model trained on Roboflow.

  set ROBOFLOW_API_KEY=<your private API key>          (cmd; the key is not saved anywhere)
  python boll_classify_roboflow.py [--model cotton-boll-open-or-close-3frah/3]

Crop = the same cut as the training data (box + 15% margin, from the 4-POV view).
Input : agrieye_output/objects/sam3_labels.jsonl (+ sam3_labels_extra.jsonl)
Output: agrieye_output/objects/boll_cls_roboflow.csv   crop, pred, conf   (resumable: done crops are skipped)
        agrieye_output/objects/boll_cls_vs_hand.txt    how often the model agrees with your hand sorting
'agrieye.py analyze' then uses for each boll: your hand label > this model > SAM 3's own label.
Cost: about 0.06 Roboflow credits per 1000 crops (~0.2 for the whole field).
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
OBJ = ROOT / "agrieye_output" / "objects"
CROPS = ROOT / "agrieye_output" / "boll_crops"
OUT = OBJ / "boll_cls_roboflow.csv"


def find_pov(r):
    p = Path(r["pov"])
    if p.exists():
        return p
    hits = list((ROOT / "agrieye_output" / "povs").glob(f"*/{r['date']}/{p.name}"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cotton-boll-open-or-close-3frah/3")
    ap.add_argument("--api", default="https://serverless.roboflow.com")
    args = ap.parse_args()
    key = (os.environ.get("ROBOFLOW_API_KEY") or "").strip().strip('"')
    if not key:
        raise SystemExit("Set your key first (cmd):  set ROBOFLOW_API_KEY=xxxxxxxx")
    from inference_sdk import InferenceHTTPClient
    client = InferenceHTTPClient(api_url=args.api, api_key=key)

    recs = []
    for f in ("sam3_labels.jsonl", "sam3_labels_extra.jsonl"):
        if (OBJ / f).exists():
            recs += [json.loads(l) for l in (OBJ / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {}
    if OUT.exists():
        done = {r["crop"]: r for r in csv.DictReader(open(OUT, encoding="utf-8"))}
    todo = [(r, k, d) for r in recs for k, d in enumerate(r["dets"])
            if f"{r['photo_id']}_{r['direction']}_{k:03d}.jpg" not in done]
    print(f"{len(done)} crops already classified, {len(todo)} to go (model {args.model})")
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["crop", "pred", "conf"])
        if new:
            w.writeheader()
        cur, im = None, None
        for n, (r, k, d) in enumerate(todo, 1):
            name = f"{r['photo_id']}_{r['direction']}_{k:03d}.jpg"
            p = find_pov(r)
            if p is None:
                continue
            if p != cur:
                cur, im = p, Image.open(p).convert("RGB")
            x1, y1, x2, y2 = d["box"]
            m = 0.15 * max(x2 - x1, y2 - y1)
            box = (int(max(0, x1 - m)), int(max(0, y1 - m)), int(min(im.width, x2 + m)), int(min(im.height, y2 + m)))
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            try:
                pred, conf = top_class(client.infer(im.crop(box), model_id=args.model))
            except Exception as e:
                print(f"  stopped at {n}/{len(todo)}: {str(e).split('?api_key')[0][:300]}\n  run again to continue.")
                break
            row = dict(crop=name, pred=pred, conf=round(conf, 4))
            w.writerow(row); fh.flush(); done[name] = row
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)

    # agreement with your hand sorting (note: many of these crops were in the TRAINING data, so this is
    # a consistency check, not a test score - the test score is eval_roboflow_boll.py)
    hand = {f.name: {"open": "open_boll", "closed": "closed_boll", "not_boll": "not_boll"}[lab]
            for lab in ("open", "closed", "not_boll") for f in (CROPS / lab).glob("*.jpg")}
    both = [(hand[c], done[c]["pred"]) for c in hand if c in done]
    cnt = Counter(p for p in (r["pred"] for r in done.values()))
    low = sum(float(r["conf"]) < 0.6 for r in done.values())
    lines = [f"Classified crops: {len(done)}   predicted: {dict(cnt)}   low confidence (<0.6): {low}"]
    if both:
        agree = sum(a == b for a, b in both)
        lines.append(f"Agreement with your hand labels: {agree}/{len(both)} = {100 * agree / len(both):.1f}% "
                     "(includes training crops - consistency check only)")
    (OBJ / "boll_cls_vs_hand.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines) + f"\n\nSaved: {OUT}")


if __name__ == "__main__":
    main()
