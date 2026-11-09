"""
Confusion matrices for both Roboflow boll models, from answers already saved on this laptop
(no API key, no credits). Roboflow's free plan hides its own confusion matrix.

  python confusion_matrix_bolls.py

R1 boll classifier  (cotton-boll-open-or-close-3frah/3): sample_results/roboflow_boll_test/predictions.csv (355 test crops)
R2 boll detector    (cotton-bolls-field-view-trbz1/1):   sample_results/roboflow_boll_detect_test/raw_*.json (18 test views)
  R2 matching: a model box at confidence >= 0.5 counts for one of your boxes if they overlap by IoU >= 0.5
  (any class), so a found boll can still have the wrong class. Your boxes with no model box = "missed";
  model boxes with no boll of yours = "false box" (background).

Output: sample_results/confusion_matrices/  R1_confusion_matrix.png, R2_confusion_matrix.png, confusion_matrices.txt
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import eval_roboflow_detect as E

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sample_results" / "confusion_matrices"
CONF, IOU = 0.5, 0.5


def r1_matrix():
    names = ["open_boll", "closed_boll", "not_boll"]
    m = np.zeros((3, 3), int)
    with open(ROOT / "sample_results" / "roboflow_boll_test" / "predictions.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["true"] in names and r["pred"] in names:
                m[names.index(r["true"]), names.index(r["pred"])] += 1
    return names, names, m


def r2_matrix():
    raw = next((ROOT / "sample_results" / "roboflow_boll_detect_test").glob("raw_*.json"))
    cache = json.loads(raw.read_text())
    rows, cols = ["open_boll", "closed_boll", "background\n(no boll)"], ["open_boll", "closed_boll", "missed\n(no box)"]
    m = np.zeros((3, 3), int)
    for f in sorted((E.TEST / "images").glob("*.jpg")):
        W, H = Image.open(f).size
        gt = E.ground_truth(E.TEST / "labels" / f"{f.stem}.txt", W, H)
        pred = sorted([p for p in E.parse(cache[f"full/{f.stem}"]) if p[1] >= CONF], key=lambda p: -p[1])
        used = [False] * len(gt)
        for c, _, box in pred:                                  # greedy: most confident box first
            best, bi = 0, -1
            for i, (_, gb) in enumerate(gt):
                if not used[i]:
                    v = E.iou(box, gb)
                    if v > best:
                        best, bi = v, i
            if best >= IOU:
                used[bi] = True; m[gt[bi][0], c] += 1
            else:
                m[2, c] += 1                                    # false box
        for i, (gc, _) in enumerate(gt):
            if not used[i]:
                m[gc, 2] += 1                                   # missed
    m[2, 2] = 0
    return rows, cols, m


def draw(rows, cols, m, title, sub, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    pct = m / np.maximum(m.sum(1, keepdims=True), 1)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i == 2 and j == 2 and "missed" in cols[2]:
                ax.text(j, i, "-", ha="center", va="center", color="#888", fontsize=13); continue
            dark = pct[i, j] > 0.55
            ax.text(j, i, f"{m[i, j]}\n{100 * pct[i, j]:.0f}%", ha="center", va="center", fontsize=12,
                    color="white" if dark else "#1f2328", fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("Model said", fontsize=11); ax.set_ylabel("Your label (truth)", fontsize=11)
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left")
    fig.text(0.01, 0.01, sub, fontsize=8.5, color="#59636e")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(path); plt.close(fig)


def table(rows, cols, m):
    w = 14
    lines = [" " * w + "".join(f"{c.replace(chr(10), ' '):>{w + 6}}" for c in cols)]
    for i, r in enumerate(rows):
        lines.append(f"{r.replace(chr(10), ' '):<{w}}" + "".join(f"{m[i, j]:>{w + 6}}" for j in range(len(cols))))
    return lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    rows, cols, m = r1_matrix()
    acc = np.trace(m) / m.sum()
    draw(rows, cols, m, f"R1 boll classifier: {100 * acc:.1f}% correct on {m.sum()} test crops",
         "cotton-boll-open-or-close-3frah/3 · test crops never used in training · % = share of each true row",
         OUT / "R1_confusion_matrix.png")
    out += [f"R1 boll classifier (cotton-boll-open-or-close-3frah/3), {m.sum()} test crops, accuracy {100 * acc:.1f}%",
            "rows = your label, columns = model"] + table(rows, cols, m) + [""]
    rows, cols, m = r2_matrix()
    found = m[0, 0] + m[0, 1] + m[1, 0] + m[1, 1]
    draw(rows, cols, m, f"R2 boll detector: right class on {m[0, 0] + m[1, 1]} of {m[:2].sum()} bolls",
         f"cotton-bolls-field-view-trbz1/1 · 18 test views · model boxes at confidence >= {CONF}, IoU >= {IOU}",
         OUT / "R2_confusion_matrix.png")
    out += [f"R2 boll detector (cotton-bolls-field-view-trbz1/1), 18 test views, conf >= {CONF}, IoU >= {IOU}",
            "rows = your label (last row = no boll there), columns = model (last column = no box)"] + table(rows, cols, m) + [
            f"bolls found (any class): {found} of {m[:2].sum()}; right class: {m[0, 0] + m[1, 1]}; "
            f"open called closed: {m[0, 1]}; closed called open: {m[1, 0]}; false boxes: {m[2, 0] + m[2, 1]}"]
    (OUT / "confusion_matrices.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out)); print(f"\nFiles: {OUT}")


if __name__ == "__main__":
    main()
