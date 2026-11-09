"""
Picture sheets to judge SAM 3's "flower" and "closed boll" labels.

  python check_flowers.py
Writes agrieye_output/flower_check/
  flowers_by_score.jpg   random "flower" detections in 3 score bands (high / mid / low)
  closed_bolls.jpg       random "closed boll" detections, for comparison
  open_bolls.jpg         random "open boll" detections
Each tile = the detection box (green) with some context, labelled with its score.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "agrieye_output"
LABELS = OUT / "objects" / "sam3_labels.jsonl"
DST = OUT / "flower_check"
TILE, PER_ROW = 150, 12
BANDS = [("score >= 0.5", 0.5, 9), ("0.4 - 0.5", 0.4, 0.5), ("0.3 - 0.4", 0.0, 0.4)]
NAMES = {0: "open boll", 1: "closed boll", 2: "flower"}


def find_pov(r):
    p = Path(r["pov"])
    if p.exists():
        return p
    hits = list((OUT / "povs").glob(f"*/{r['date']}/{p.name}"))   # photo moved to another plant ID
    return hits[0] if hits else None


def tiles_for(items, cache):
    out = []
    for r, d in items:
        if r["pov"] not in cache:
            p = find_pov(r)
            cache[r["pov"]] = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) if p else None
        img = cache[r["pov"]]
        if img is None:
            continue
        x1, y1, x2, y2 = d["box"]
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.6 + 20
        a, b = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        c, e = int(min(img.shape[1], cx + s / 2)), int(min(img.shape[0], cy + s / 2))
        crop = img[b:e, a:c].copy()
        cv2.rectangle(crop, (int(x1 - a), int(y1 - b)), (int(x2 - a), int(y2 - b)), (60, 220, 60), 2)
        h, w = crop.shape[:2]
        k = TILE / max(h, w)
        crop = cv2.resize(crop, (max(1, int(w * k)), max(1, int(h * k))))
        t = np.full((TILE + 18, TILE, 3), 255, np.uint8)
        t[:crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(t, f"{d['score']:.2f} {r['station']} {r['date'][5:]}", (3, TILE + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
        out.append(t)
    return out


def grid(tiles, title):
    rows = []
    for i in range(0, len(tiles), PER_ROW):
        row = tiles[i:i + PER_ROW]
        row += [np.full_like(tiles[0], 235)] * (PER_ROW - len(row))
        rows.append(np.hstack([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=235) for t in row]))
    body = np.vstack(rows) if rows else np.full((40, PER_ROW * (TILE + 4), 3), 235, np.uint8)
    bar = np.full((34, body.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.vstack([bar, body])


def save(img, name):
    DST.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(DST / name))
    print(DST / name)


def main():
    recs = [json.loads(l) for l in LABELS.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cls = {k: [(r, d) for r in recs for d in r["dets"] if d["cls"] == k] for k in NAMES}
    for k, v in by_cls.items():
        print(f"{NAMES[k]}: {len(v)} detections")
    rng = random.Random(0)
    cache = {}
    flowers = by_cls[2]
    parts = []
    for lab, lo, hi in BANDS:
        band = [x for x in flowers if lo <= x[1]["score"] < hi]
        pick = rng.sample(band, min(36, len(band)))
        pick.sort(key=lambda x: -x[1]["score"])
        parts.append(grid(tiles_for(pick, cache), f'"flower"  {lab}   ({len(band)} detections, showing {len(pick)})'))
    save(np.vstack(parts), "flowers_by_score.jpg")
    for k, name in ((1, "closed_bolls.jpg"), (0, "open_bolls.jpg")):
        pick = rng.sample(by_cls[k], min(48, len(by_cls[k])))
        pick.sort(key=lambda x: -x[1]["score"])
        save(grid(tiles_for(pick, cache), f'"{NAMES[k]}"  random {len(pick)} of {len(by_cls[k])}'), name)


if __name__ == "__main__":
    main()
