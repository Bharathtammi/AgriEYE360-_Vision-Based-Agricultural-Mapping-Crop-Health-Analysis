"""
Experiment 2: can SAM 3 ALONE judge leaf disease by segmenting the DISEASE SPOTS (not the whole leaf)?

Experiment 1 (sam3_disease_test.py) asked SAM 3 for "diseased leaf" on big photo tiles: AUC 0.45-0.75, far
below the trained classifier v2 (AUC 0.944). There the spots were only a few pixels wide. Here every leaf is
ENLARGED: the 835 hand-labelled test leaves (disease_autolabel/dataset_cls/val: 289 diseased, 546 healthy,
the same leaves v2 is tested on) are packed 2 x 2 into 644 px pictures (each leaf ~320 px), and SAM 3 is asked
to segment spot-type prompts. A leaf's disease score = share of the leaf covered by those spot masks.

  python sam3_spot_test.py run [--limit N]    # SAM 3 on the leaf pictures (resumable; ~1 s per picture per prompt on CPU)
  python sam3_spot_test.py report             # AUC / accuracy vs your labels, next to v2

Output: sample_results/sam3_spot_test/  (raw.jsonl, report.txt, report.csv, examples.jpg)
No training: SAM 3 is used as it comes.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
VAL = ROOT / "disease_autolabel" / "dataset_cls" / "val"
OUT = ROOT / "sample_results" / "sam3_spot_test"
RAW = OUT / "raw.jsonl"
PROMPTS = ["brown spot", "leaf spot", "hole in leaf", "yellow patch"]
SIDE, CELL, GRID = 644, 322, 2          # 644 = SAM 3 input size (multiple of 14); 2 x 2 leaves per picture
SMALL = 0.5                             # a "spot" mask may cover at most this share of the leaf (else it is the leaf)
V2 = dict(accuracy=89.2, auc=0.944, recall=82.7, precision=85.7)   # disease classifier v2 on the same 835 leaves


def leaves():
    return [(p, c) for c in ("diseased", "healthy") for p in sorted((VAL / c).glob("*.png"))]


def mosaic(group):
    """2 x 2 picture; returns image, per-cell leaf masks (bool, picture coords) and cell boxes."""
    im = Image.new("RGB", (SIDE, SIDE), (0, 0, 0))
    cells = []
    for k, (p, c) in enumerate(group):
        leaf = Image.open(p).convert("RGB")
        s = (CELL - 8) / max(leaf.size)
        leaf = leaf.resize((max(1, int(leaf.width * s)), max(1, int(leaf.height * s))), Image.LANCZOS)
        x0 = (k % GRID) * CELL + (CELL - leaf.width) // 2
        y0 = (k // GRID) * CELL + (CELL - leaf.height) // 2
        im.paste(leaf, (x0, y0))
        m = np.zeros((SIDE, SIDE), bool)
        m[y0:y0 + leaf.height, x0:x0 + leaf.width] = np.asarray(leaf).max(axis=2) > 12      # leaf pixels (not black)
        cells.append(dict(crop=p.name, truth=c, mask=m, box=((k % GRID) * CELL, (k // GRID) * CELL)))
    return im, cells


def run(limit):
    from sam3_autolabel import build_predictor
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if RAW.exists():
        done = {json.loads(l)["crop"] for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [x for x in leaves() if x[0].name not in done]
    rng = np.random.default_rng(0)                         # mix diseased / healthy in every picture
    todo = [todo[i] for i in rng.permutation(len(todo))]
    if limit:
        todo = todo[:limit]
    pred, _ = build_predictor(ROOT / "sam3.pt", 0.1, "cpu")
    t0 = time.time()
    groups = [todo[i:i + GRID * GRID] for i in range(0, len(todo), GRID * GRID)]
    for gi, group in enumerate(groups, 1):
        im, cells = mosaic(group)
        pred.set_image(np.ascontiguousarray(np.asarray(im)[:, :, ::-1]))
        res_cells = [dict(crop=c["crop"], truth=c["truth"], leaf_px=int(c["mask"].sum()), p={}) for c in cells]
        for prompt in PROMPTS:
            res = pred(text=[prompt]); res = res[0] if isinstance(res, (list, tuple)) else res
            union = [np.zeros((SIDE, SIDE), bool) for _ in cells]
            small = [np.zeros((SIDE, SIDE), bool) for _ in cells]
            maxs, n = [0.0] * len(cells), [0] * len(cells)
            if res.masks is not None:
                ms = res.masks.data.cpu().numpy() > 0.5
                if ms.shape[1:] != (SIDE, SIDE):
                    ms = np.stack([np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((SIDE, SIDE))) > 127 for m in ms])
                for m, s in zip(ms, res.boxes.conf.cpu().numpy()):
                    for k, c in enumerate(cells):
                        inside = m & c["mask"]
                        if inside.sum() < 4 or inside.sum() < 0.5 * m.sum():   # mask belongs to another cell
                            continue
                        union[k] |= inside
                        if inside.sum() <= SMALL * max(1, c["mask"].sum()):
                            small[k] |= inside
                        maxs[k] = max(maxs[k], float(s)); n[k] += 1
            for k, c in enumerate(cells):
                lp = max(1, c["mask"].sum())
                res_cells[k]["p"][prompt] = dict(frac=round(float(union[k].sum() / lp), 4),
                                                 frac_small=round(float(small[k].sum() / lp), 4),
                                                 max=round(maxs[k], 4), n=n[k])
        with RAW.open("a", encoding="utf-8") as fh:
            for r in res_cells:
                fh.write(json.dumps(r) + "\n")
        el = time.time() - t0
        print(f"  picture {gi}/{len(groups)}  ({len(done) + gi * GRID * GRID} leaves)  "
              f"{el / gi:.1f} s per picture, about {el / gi * (len(groups) - gi) / 60:.0f} min left", flush=True)


def auc(y, s):
    order = np.argsort(-np.asarray(s, float), kind="mergesort")
    y = np.asarray(y, float)[order]
    tp, fp = np.r_[0, np.cumsum(y)], np.r_[0, np.cumsum(1 - y)]
    return float(np.trapezoid(tp / max(1, y.sum()), fp / max(1, (1 - y).sum())))


def best_split(y, s):
    """Threshold that gives the best accuracy (optimistic: chosen on the test set itself)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    best = (0, 0)
    for t in np.unique(s):
        p = s >= t
        acc = (p == y).mean()
        if acc > best[0]:
            best = (acc, t)
    p = s >= best[1]
    tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
    return best[0], best[1], tp / max(1, tp + fn), tp / max(1, tp + fp)


def report():
    rows = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    y = [r["truth"] == "diseased" for r in rows]
    lines = [f"SAM 3 ALONE, spot segmentation, on {len(rows)} hand-labelled test leaves "
             f"({sum(y)} diseased, {len(y) - sum(y)} healthy); each leaf enlarged to ~320 px",
             "score = share of the leaf covered by SAM 3's spot masks (small = masks smaller than half the leaf)", "",
             f"{'SAM 3 score':40s} {'AUC':>6s} {'best acc':>9s} {'dis. recall':>12s} {'dis. prec':>10s}"]
    out = []
    scores = {}
    for pr in PROMPTS:
        for key, name in (("frac_small", "spot area"), ("max", "spot confidence")):
            scores[f'"{pr}" {name}'] = [r["p"][pr][key] for r in rows]
    scores["all spot prompts, area summed"] = [sum(r["p"][pr]["frac_small"] for pr in PROMPTS) for r in rows]
    scores["all spot prompts, best confidence"] = [max(r["p"][pr]["max"] for pr in PROMPTS) for r in rows]
    for name, s in scores.items():
        a = auc(y, s); acc, t, rec, prec = best_split(y, s)
        out.append(dict(score=name, auc=round(a, 3), best_accuracy=round(100 * acc, 1), threshold=round(float(t), 4),
                        diseased_recall=round(100 * rec, 1), diseased_precision=round(100 * prec, 1)))
        lines.append(f"{name:40s} {a:6.3f} {100 * acc:8.1f}% {100 * rec:11.1f}% {100 * prec:9.1f}%")
    base = 100 * (1 - np.mean(y))
    lines += ["", f"{'(reference) always say healthy':40s} {'':6s} {base:8.1f}%",
              f"{'Disease classifier v2 (trained, L2)':40s} {V2['auc']:6.3f} {V2['accuracy']:8.1f}% "
              f"{V2['recall']:11.1f}% {V2['precision']:9.1f}%", "",
              "'best acc' uses the best threshold chosen on these same leaves, so it flatters SAM 3 a little."]
    best = max(out, key=lambda r: r["auc"])
    verdict = ("SAM 3 alone is CLOSE to the trained classifier" if best["auc"] >= 0.9 else
               "SAM 3 alone is clearly WORSE than the trained classifier" if best["auc"] < 0.85 else
               "SAM 3 alone is somewhat worse than the trained classifier")
    lines += [f"Best SAM 3 score: {best['score']} (AUC {best['auc']}) -> {verdict} (v2 AUC {V2['auc']})."]
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(OUT / "report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("\n".join(lines))
    examples(rows, scores[best["score"]], best["score"])
    print(f"\nFiles: {OUT}")


def examples(rows, s, name):
    """Highest-scoring leaves and the diseased leaves SAM 3 scored lowest, with your label."""
    T = 140
    order = np.argsort(-np.asarray(s))
    top = [rows[i] for i in order[:12]]
    low = [rows[i] for i in order[::-1] if rows[i]["truth"] == "diseased"][:12]
    sheet = Image.new("RGB", (6 * (T + 6), 4 * (T + 22) + 70), (235, 235, 235)); d = ImageDraw.Draw(sheet)
    for gi, (title, grp) in enumerate([(f"Highest SAM 3 score ({name})", top), ("Diseased (your label) but lowest SAM 3 score", low)]):
        y0 = gi * (2 * (T + 22) + 35)
        d.text((6, y0 + 8), title, fill=(0, 0, 0))
        for i, r in enumerate(grp):
            im = Image.open(VAL / r["truth"] / r["crop"]).convert("RGB"); im.thumbnail((T, T))
            x, yy = (i % 6) * (T + 6) + 3, y0 + 26 + (i // 6) * (T + 22)
            sheet.paste(im, (x, yy))
            d.text((x, yy + T + 3), f"you: {r['truth']}", fill=(160, 0, 0) if r["truth"] == "diseased" else (0, 110, 0))
    sheet.save(OUT / "examples.jpg", quality=88)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a.limit) if a.cmd == "run" else report()
