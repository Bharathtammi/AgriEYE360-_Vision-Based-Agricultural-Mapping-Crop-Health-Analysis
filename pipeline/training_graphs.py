"""
Roboflow-style training graphs for the two models trained on this laptop, from their saved training logs
(results.csv written by Ultralytics after every epoch). Nothing is trained or re-run.

  python training_graphs.py

  L1 leaf finder (YOLO11n-seg):        disease_autolabel/models/leaf_seg/results.csv
  L2 disease classifier v2 (YOLO11n-cls): disease_autolabel/models/disease_cls/results.csv

Layout (like Roboflow): "Model performance" over epochs on top, three loss panels below.
Added: the best epoch (= the saved best.pt) and validation loss next to training loss (shows overfitting).
Output: sample_results/training_graphs/  L1_leaf_finder_training.png, L2_disease_classifier_training.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "disease_autolabel" / "models"
OUT = ROOT / "sample_results" / "training_graphs"
ACCENT, LIGHT, GREY, INK, QUIET, GRID = "#2a78d6", "#7fb0ea", "#8a949e", "#1f2328", "#59636e", "#e6e9ec"


def load(name):
    with open(MODELS / name / "results.csv", encoding="utf-8") as fh:
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    return rows


def style(ax, title, ylabel=None, pct=False):
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c4cad0")
    ax.tick_params(colors=QUIET, labelsize=9)
    ax.set_xlabel("Epoch", color=QUIET, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=QUIET, fontsize=9.5)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))


def best_line(ax, ep, label=True):
    ax.axvline(ep, color=QUIET, linestyle=":", linewidth=1.2)
    if label:
        ax.text(ep, ax.get_ylim()[1], f"best epoch {ep:.0f} \n(saved as best.pt) ", va="top", ha="right",
                fontsize=8.5, color=QUIET)


def loss_panel(ax, rows, train_key, val_key, title, best):
    e = [r["epoch"] for r in rows]
    ax.plot(e, [r[train_key] for r in rows], color=ACCENT, linewidth=2, label="training")
    if val_key:
        ax.plot(e, [r[val_key] for r in rows], color=GREY, linewidth=1.8, linestyle="--", label="validation")
    style(ax, title)
    best_line(ax, best, label=False)
    if val_key:                                   # one line needs no legend: the title names it
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")


def figure(title, sub, path):
    fig = plt.figure(figsize=(14, 8.2), dpi=140)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.42, wspace=0.22)
    fig.text(0.06, 0.965, title, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.937, sub, fontsize=10, color=QUIET)
    return fig, fig.add_subplot(gs[0, :]), [fig.add_subplot(gs[1, i]) for i in range(3)], path


def save(fig, path, note):
    fig.text(0.06, 0.012, note, fontsize=8.5, color=QUIET, va="bottom")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.1)
    fig.savefig(path); plt.close(fig)
    print(path)


def l1():
    rows = load("leaf_seg")
    e = [r["epoch"] for r in rows]
    m50, m5095 = [100 * r["metrics/mAP50(M)"] for r in rows], [100 * r["metrics/mAP50-95(M)"] for r in rows]
    best = rows[max(range(len(rows)), key=lambda i: rows[i]["metrics/mAP50(M)"])]
    fig, top, bottom, path = figure(
        f"L1 leaf finder (YOLO11n-seg): mask mAP50 reached {max(m50):.1f}% after {len(rows)} epochs",
        f"Validation on 144 tiles · trained on 832 tiles · batch 8 · image size 640 · {rows[-1]['time'] / 3600:.1f} h on CPU",
        OUT / "L1_leaf_finder_training.png")
    top.plot(e, m50, color=ACCENT, linewidth=2.4, label="mask mAP50")
    top.plot(e, m5095, color=LIGHT, linewidth=2, linestyle="--", label="mask mAP50-95")
    bi = rows.index(best)
    top.annotate(f"{m50[bi]:.1f}%", (e[bi], m50[bi]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    top.annotate(f"{m5095[-1]:.1f}%", (e[-1], m5095[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK)
    style(top, "Model performance (validation)", pct=True); top.set_ylim(0, max(m50) * 1.32)
    best_line(top, best["epoch"]); top.legend(frameon=False, fontsize=9.5, loc="lower right")
    loss_panel(bottom[0], rows, "train/box_loss", "val/box_loss", "Box loss", best["epoch"])
    loss_panel(bottom[1], rows, "train/seg_loss", "val/seg_loss", "Segmentation (mask) loss", best["epoch"])
    loss_panel(bottom[2], rows, "train/cls_loss", "val/cls_loss", "Class loss", best["epoch"])
    save(fig, path, "Source: disease_autolabel/models/leaf_seg/results.csv (Ultralytics log, one row per epoch). "
                    "Still rising at the last epoch: a longer run could gain a little.")


def l2():
    rows = load("disease_cls")
    e = [r["epoch"] for r in rows]
    acc = [100 * r["metrics/accuracy_top1"] for r in rows]
    bi = max(range(len(rows)), key=lambda i: acc[i]); best = rows[bi]
    fig, top, bottom, path = figure(
        f"L2 disease classifier v2 (YOLO11n-cls): best accuracy {acc[bi]:.1f}% at epoch {best['epoch']:.0f} of {len(rows)}",
        f"Validation on 835 field leaves · trained on 2,924 leaves (970 diseased, 1,954 healthy) · batch 32 · image size 224 · "
        f"{rows[-1]['time'] / 60:.0f} min on CPU · early stopping (patience 10)",
        OUT / "L2_disease_classifier_training.png")
    top.plot(e, acc, color=ACCENT, linewidth=2.4, marker="o", markersize=4, label="validation accuracy")
    top.annotate(f"{acc[bi]:.1f}%", (e[bi], acc[bi]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    top.annotate(f"{acc[-1]:.1f}%", (e[-1], acc[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK)
    style(top, "Model performance (validation)", pct=True)
    top.set_ylim(min(acc) - 3, max(acc) + 3.5); top.set_xticks(e)
    best_line(top, best["epoch"]); top.legend(frameon=False, fontsize=9.5, loc="lower right")
    for ax in bottom:
        ax.set_xticks(e[::2])
    loss_panel(bottom[0], rows, "train/loss", None, "Training loss", best["epoch"])
    loss_panel(bottom[1], rows, "val/loss", None, "Validation loss", best["epoch"])
    bottom[1].lines[0].set_color(GREY)
    lr = bottom[2]
    lr.plot(e, [1000 * r["lr/pg0"] for r in rows], color=ACCENT, linewidth=2)
    style(lr, "Learning rate (x 0.001)"); best_line(lr, best["epoch"], label=False)
    save(fig, path, "Source: disease_autolabel/models/disease_cls/results.csv (Ultralytics log, one row per epoch). "
                    "Training loss keeps falling after the best epoch\n"
                    "while validation accuracy and loss do not improve: the start of overfitting, so the epoch-9 weights are kept.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    l1(); l2()
    print(f"\nFiles: {OUT}")
