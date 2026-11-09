"""
Performance charts for the healthy/diseased leaf classifier.

Evaluates  disease_autolabel\\models\\disease_cls\\weights\\best.pt  on the validation set
(disease_autolabel\\dataset_cls\\val: photos never used for training) and writes to
disease_autolabel\\metrics\\ :

  00_results_grid.png        2x5 per-epoch grid (Roboflow/Ultralytics style): losses, precision,
                             recall, F1, accuracy, ROC AUC, AP - results + smoothed
  epoch_metrics.csv          the per-epoch numbers (cached; needs weights\\epoch*.pt)
  01_training_curves.png     loss and accuracy per epoch
  02_confusion_matrix.png    counts and % per class
  03_class_metrics.png       precision / recall / F1 per class
  04_roc_curve.png           ROC curve + AUC
  05_precision_recall.png    precision-recall curve + average precision
  06_threshold_tradeoff.png  how the decision threshold trades diseased recall vs healthy accuracy
  dashboard.png              all of the above on one page
  metrics_summary.csv        the numbers behind the charts (table view)

Run:
  "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" "F:\\Project and completed courses\\Agri EYE 360\\model_metrics.py"
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "disease_autolabel"
RUN = WORK / "models" / "disease_cls"
MODEL = RUN / "weights" / "best.pt"
VAL = WORK / "dataset_cls" / "val"
OUT = WORK / "metrics"
OUT.mkdir(exist_ok=True)

# palette (validated: blue/orange pass CVD + contrast on the light surface)
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})

# ------------------------------------------------------------------ predictions
model = YOLO(str(MODEL))
d_idx = [k for k, v in model.names.items() if v == "diseased"][0]
y_true, p_dis = [], []
for cls in ("diseased", "healthy"):
    files = sorted((VAL / cls).glob("*"))
    for i in range(0, len(files), 128):
        for r in model.predict([str(f) for f in files[i:i + 128]], imgsz=224, verbose=False, device="cpu"):
            y_true.append(1 if cls == "diseased" else 0)
            p_dis.append(float(r.probs.data[d_idx]))
y_true, p_dis = np.array(y_true), np.array(p_dis)
THR = 0.5
y_pred = (p_dis >= THR).astype(int)

tp = int(((y_pred == 1) & (y_true == 1)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
tn = int(((y_pred == 0) & (y_true == 0)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())


def prf(tp_, fp_, fn_):
    p = tp_ / max(1, tp_ + fp_); r = tp_ / max(1, tp_ + fn_)
    return p, r, 2 * p * r / max(1e-9, p + r)


m_dis = prf(tp, fp, fn)
m_hea = prf(tn, fn, fp)
acc = (tp + tn) / len(y_true)

# ROC / PR over all thresholds
ths = np.r_[np.inf, np.sort(np.unique(p_dis))[::-1], -np.inf]
tpr = np.array([((p_dis >= t) & (y_true == 1)).sum() / (y_true == 1).sum() for t in ths])
fpr = np.array([((p_dis >= t) & (y_true == 0)).sum() / (y_true == 0).sum() for t in ths])
auc = float(np.trapezoid(tpr, fpr))
prec = np.array([((p_dis >= t) & (y_true == 1)).sum() / max(1, (p_dis >= t).sum()) for t in ths])
prec[0] = 1.0
ap = float(np.sum((tpr[1:] - tpr[:-1]) * prec[1:]))
grid_t = np.linspace(0.05, 0.95, 91)
rec_d = np.array([((p_dis >= t) & (y_true == 1)).sum() / (y_true == 1).sum() for t in grid_t])
acc_h = np.array([((p_dis < t) & (y_true == 0)).sum() / (y_true == 0).sum() for t in grid_t])

# training log
ep, tl, vl, va = [], [], [], []
with (RUN / "results.csv").open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        r = {k.strip(): v for k, v in r.items()}
        ep.append(int(r["epoch"])); tl.append(float(r["train/loss"]))
        vl.append(float(r["val/loss"])); va.append(float(r["metrics/accuracy_top1"]))
best_ep = ep[int(np.argmax(va))]


# ------------------------------------------------------------------ per-epoch metrics
# Needs weights\epoch{N}.pt (training with save_period=1). Cached in metrics\epoch_metrics.csv.
def score_set(yt, p, thr=0.5):
    yp = p >= thr
    tp_ = int((yp & (yt == 1)).sum()); fn_ = int((~yp & (yt == 1)).sum())
    tn_ = int((~yp & (yt == 0)).sum()); fp_ = int((yp & (yt == 0)).sum())
    pd_, rd_, fd_ = prf(tp_, fp_, fn_)
    ph_, rh_, fh_ = prf(tn_, fn_, fp_)
    t_ = np.r_[np.inf, np.sort(np.unique(p))[::-1], -np.inf]
    tpr_ = np.array([((p >= t) & (yt == 1)).sum() / (yt == 1).sum() for t in t_])
    fpr_ = np.array([((p >= t) & (yt == 0)).sum() / (yt == 0).sum() for t in t_])
    pr_ = np.array([((p >= t) & (yt == 1)).sum() / max(1, (p >= t).sum()) for t in t_]); pr_[0] = 1
    return {"precision_diseased": pd_, "recall_diseased": rd_, "f1_diseased": fd_,
            "precision_healthy": ph_, "recall_healthy": rh_, "f1_healthy": fh_,
            "accuracy": (tp_ + tn_) / len(yt), "roc_auc": float(np.trapezoid(tpr_, fpr_)),
            "ap_diseased": float(np.sum((tpr_[1:] - tpr_[:-1]) * pr_[1:]))}


val_files = [f for c in ("diseased", "healthy") for f in sorted((VAL / c).glob("*"))]
epoch_w = sorted((RUN / "weights").glob("epoch*.pt"), key=lambda f: int(f.stem[5:]))
cache_f = OUT / "epoch_metrics.csv"
per_epoch = {}
if cache_f.exists():
    with cache_f.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.pop("weights_mtime") == str(int((RUN / "weights" / f"epoch{r['epoch']}.pt").stat().st_mtime)
                                             if (RUN / "weights" / f"epoch{r['epoch']}.pt").exists() else ""):
                per_epoch[int(r.pop("epoch"))] = {k: float(v) for k, v in r.items()}
for w in epoch_w:
    e = int(w.stem[5:])
    if e in per_epoch:
        continue
    print(f"  scoring epoch {e} ...", end="\r")
    mm = YOLO(str(w)); di = [k for k, v in mm.names.items() if v == "diseased"][0]
    pe = []
    for i in range(0, len(val_files), 128):
        pe += [float(r.probs.data[di]) for r in
               mm.predict([str(f) for f in val_files[i:i + 128]], imgsz=224, verbose=False, device="cpu")]
    per_epoch[e] = score_set(y_true, np.array(pe))
if per_epoch:
    keys = list(next(iter(per_epoch.values())).keys())
    with cache_f.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["epoch", *keys, "weights_mtime"])
        for e in sorted(per_epoch):
            wf = RUN / "weights" / f"epoch{e}.pt"
            wr.writerow([e, *[f"{per_epoch[e][k]:.5f}" for k in keys], int(wf.stat().st_mtime) if wf.exists() else ""])


# ------------------------------------------------------------------ charts
def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=INK, fontsize=9,
                va="center")
    ax.plot([x], [y], "o", ms=8, color=color, mec=SURF, mew=2, zorder=5)


def ch_curves(ax_loss, ax_acc):
    ax_loss.plot(ep, tl, color=BLUE, label="Train loss")
    ax_loss.plot(ep, vl, color=ORANGE, label="Validation loss")
    ax_loss.axvline(best_ep, color=INK2, lw=1, ls="--")
    ax_loss.set(title="Loss per epoch", xlabel="Epoch", ylabel="Loss")
    ax_loss.legend(loc="upper right")
    ax_acc.plot(ep, np.array(va) * 100, color=BLUE)
    b = va[ep.index(best_ep)] * 100
    label_end(ax_acc, best_ep, b, f"Best: {b:.1f}% (epoch {best_ep})", BLUE, dy=10)
    ax_acc.set(title="Validation accuracy per epoch", xlabel="Epoch", ylabel="Accuracy (%)")
    ax_acc.set_ylim(min(va) * 100 - 3, 100)


def ch_confusion(ax):
    cm = np.array([[tp, fn], [fp, tn]])
    rows = cm.sum(axis=1, keepdims=True)
    pct = cm / rows
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=13,
                    color="white" if pct[i, j] > 0.55 else INK, fontweight="bold")
    ax.set_xticks([0, 1], ["Diseased", "Healthy"]); ax.set_yticks([0, 1], ["Diseased", "Healthy"])
    ax.set(title="Confusion matrix (threshold 0.5)", xlabel="Model says", ylabel="You labelled")
    for s in ax.spines.values():
        s.set_visible(False)


def ch_class(ax):
    names = ["Precision", "Recall", "F1"]
    x = np.arange(3); w = 0.36
    for off, vals, col, lab in ((-w / 2 - 0.01, m_dis, ORANGE, "Diseased"), (w / 2 + 0.01, m_hea, BLUE, "Healthy")):
        bars = ax.bar(x + off, np.array(vals) * 100, w, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v:.0%}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, names); ax.set_ylim(0, 108); ax.set_ylabel("%")
    ax.set_title("Per-class precision, recall, F1"); ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)


def ch_roc(ax):
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="Random guess")
    ax.plot(fpr, tpr, color=BLUE, label=f"Model (AUC {auc:.3f})")
    i = int(np.argmin(np.abs(ths[1:-1] - THR))) + 1
    label_end(ax, fpr[i], tpr[i], "threshold 0.5", BLUE, dy=-10)
    ax.set(title="ROC curve", xlabel="Healthy leaves wrongly flagged (FPR)",
           ylabel="Diseased leaves caught (TPR)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")


def ch_pr(ax):
    base = (y_true == 1).mean()
    ax.axhline(base, color=INK2, lw=1, ls="--", label=f"Random guess ({base:.0%})")
    ax.plot(tpr[1:], prec[1:], color=ORANGE, label=f"Model (AP {ap:.3f})")
    ax.set(title="Precision-recall (diseased)", xlabel="Recall: diseased caught",
           ylabel="Precision: flagged that are diseased", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left")


def ch_thr(ax):
    ax.plot(grid_t, rec_d * 100, color=ORANGE, label="Diseased caught (recall)")
    ax.plot(grid_t, acc_h * 100, color=BLUE, label="Healthy correct")
    ax.axvline(THR, color=INK2, lw=1, ls="--")
    j = int(np.argmin(np.abs(grid_t - THR)))
    label_end(ax, grid_t[j], rec_d[j] * 100, f"{rec_d[j]:.0%}", ORANGE)
    label_end(ax, grid_t[j], acc_h[j] * 100, f"{acc_h[j]:.0%}", BLUE)
    ax.set(title="Decision threshold trade-off", xlabel="Threshold on p(diseased)", ylabel="%",
           ylim=(0, 102))
    ax.legend(loc="lower center")


def save(fn, name, size=(6.4, 4.6)):
    fig, ax = plt.subplots(figsize=size)
    fn(ax); fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)


# ---- Roboflow / Ultralytics-style grid: one panel per logged metric, results + smoothed
def smooth(y, f=0.1):
    n = max(1, round(len(y) * f * 2) // 2 + 1)
    p = np.ones(n // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]))
    return np.convolve(yp, np.ones(n) / n, mode="valid")


log = {}
with (RUN / "results.csv").open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        for k, v in r.items():
            log.setdefault(k.strip(), []).append(float(v))
# 2 x 5 grid like Ultralytics/Roboflow results.png. Per-epoch precision/recall/AUC come
# from per_epoch (needs epoch*.pt); without them only the logged columns are drawn.
pe_ep = sorted(e for e in per_epoch if e in log["epoch"])
series = {"train/loss": (log["epoch"], log["train/loss"]), "val/loss": (log["epoch"], log["val/loss"]),
          "metrics/accuracy_top1": (log["epoch"], log["metrics/accuracy_top1"])}
for k, title in (("precision_diseased", "metrics/precision(diseased)"), ("recall_diseased", "metrics/recall(diseased)"),
                 ("f1_diseased", "metrics/F1(diseased)"), ("roc_auc", "metrics/ROC_AUC"),
                 ("precision_healthy", "metrics/precision(healthy)"), ("recall_healthy", "metrics/recall(healthy)"),
                 ("ap_diseased", "metrics/AP(diseased)")):
    if pe_ep:
        series[title] = (pe_ep, [per_epoch[e][k] for e in pe_ep])
layout = [["train/loss", "metrics/precision(diseased)", "metrics/recall(diseased)", "metrics/F1(diseased)",
           "metrics/ROC_AUC"],
          ["val/loss", "metrics/precision(healthy)", "metrics/recall(healthy)", "metrics/accuracy_top1",
           "metrics/AP(diseased)"]]
with plt.style.context("default"):
    fig, axs = plt.subplots(2, 5, figsize=(17, 7))
    for row, names_ in zip(axs, layout):
        for ax, name in zip(row, names_):
            ax.set_title(name, fontsize=13)
            if name not in series:
                ax.text(0.5, 0.5, "needs per-epoch\nweights", ha="center", va="center", transform=ax.transAxes,
                        color="gray")
                continue
            x, y = series[name]
            y = np.array(y, dtype=float)
            ax.plot(x, y, marker="o", ms=4, lw=1.5, color="#1f77b4", label="results")
            ax.plot(x, smooth(y), ":", lw=2, color="#ff7f0e", label="smooth")
    axs[0, 1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "00_results_grid.png", dpi=150)
    plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6)); ch_curves(a1, a2); fig.tight_layout()
fig.savefig(OUT / "01_training_curves.png", dpi=150); plt.close(fig)
save(ch_confusion, "02_confusion_matrix.png", (5.2, 4.6))
save(ch_class, "03_class_metrics.png")
save(ch_roc, "04_roc_curve.png", (5.4, 5))
save(ch_pr, "05_precision_recall.png", (5.4, 5))
save(ch_thr, "06_threshold_tradeoff.png")

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 4)
ch_curves(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]))
ch_confusion(fig.add_subplot(gs[0, 2]))
ch_class(fig.add_subplot(gs[0, 3]))
ch_roc(fig.add_subplot(gs[1, 0]))
ch_pr(fig.add_subplot(gs[1, 1]))
ch_thr(fig.add_subplot(gs[1, 2:]))
fig.suptitle(f"AgriEYE leaf disease classifier: {len(y_true)} test leaves from unseen photos | "
             f"accuracy {acc:.1%} | ROC AUC {auc:.3f}", fontsize=15, fontweight="bold", color=INK, x=0.01,
             ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "dashboard.png", dpi=130); plt.close(fig)

rows = [
    ("test leaves", len(y_true)), ("test diseased", int(y_true.sum())), ("test healthy", int((1 - y_true).sum())),
    ("accuracy", f"{acc:.4f}"), ("roc_auc", f"{auc:.4f}"), ("average_precision", f"{ap:.4f}"),
    ("diseased precision", f"{m_dis[0]:.4f}"), ("diseased recall", f"{m_dis[1]:.4f}"), ("diseased f1", f"{m_dis[2]:.4f}"),
    ("healthy precision", f"{m_hea[0]:.4f}"), ("healthy recall", f"{m_hea[1]:.4f}"), ("healthy f1", f"{m_hea[2]:.4f}"),
    ("TP diseased->diseased", tp), ("FN diseased->healthy", fn), ("FP healthy->diseased", fp), ("TN healthy->healthy", tn),
    ("best epoch", best_ep), ("epochs run", ep[-1]),
]
with (OUT / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as fh:
    csv.writer(fh).writerows([("metric", "value"), *rows])
for k, v in rows:
    print(f"{k:24s} {v}")
print(f"\nCharts: {OUT}")
