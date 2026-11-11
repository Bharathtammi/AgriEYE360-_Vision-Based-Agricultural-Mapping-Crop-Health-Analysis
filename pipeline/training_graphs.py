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
    import time
    for attempt in range(5):                     # Windows: the picture may be open in a viewer for a moment
        try:
            fig.savefig(path); break
        except OSError:
            if attempt == 4:
                print(f"Could not save {path} (is it open in another program?)")
            time.sleep(2)
    plt.close(fig)
    print(path)


def args(name):
    """Training settings Ultralytics saved next to the log (args.yaml: one 'key: value' per line)."""
    out = {}
    f = MODELS / name / "args.yaml"
    if f.exists():
        for l in f.read_text(encoding="utf-8").splitlines():
            if ":" in l and not l.startswith(" "):
                k, v = l.split(":", 1); out[k.strip()] = v.strip().strip("'\"")
    return out


def count(folder):
    return sum(1 for p in Path(folder).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def hours(sec):
    return f"{sec / 3600:.1f} h" if sec >= 5400 else f"{sec / 60:.0f} min"


def l1():
    rows = load("leaf_seg")
    a = args("leaf_seg")
    ds = Path(a.get("data", "")).parent
    e = [r["epoch"] for r in rows]
    m50, m5095 = [100 * r["metrics/mAP50(M)"] for r in rows], [100 * r["metrics/mAP50-95(M)"] for r in rows]
    best = rows[max(range(len(rows)), key=lambda i: rows[i]["metrics/mAP50(M)"])]
    end = ("Still rising near the last epoch: a longer run could gain a little." if best["epoch"] >= len(rows) - 3
           else f"Best at epoch {best['epoch']:.0f}; later epochs did not improve it (best weights kept).")
    fig, top, bottom, path = figure(
        f"L1 leaf finder (YOLO11n-seg): mask mAP50 reached {max(m50):.1f}% after {len(rows)} epochs",
        f"Validation on {count(ds / 'images' / 'val')} tiles · trained on {count(ds / 'images' / 'train')} tiles · "
        f"batch {a.get('batch', '?')} · image size {a.get('imgsz', '?')} · {hours(rows[-1]['time'])} on CPU",
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
    save(fig, path, f"Source: disease_autolabel/models/leaf_seg/results.csv (Ultralytics log, one row per epoch). {end}")


def l2(name="disease_cls", label="L2 disease classifier", file="L2_disease_classifier_training.png"):
    rows = load(name)
    a = args(name)
    ds = Path(a.get("data", ""))
    n = {s: {c: count(ds / s / c) for c in ("diseased", "healthy")} for s in ("train", "val")}
    e = [r["epoch"] for r in rows]
    acc = [100 * r["metrics/accuracy_top1"] for r in rows]
    bi = max(range(len(rows)), key=lambda i: acc[i]); best = rows[bi]
    fig, top, bottom, path = figure(
        f"{label} (YOLO11n-cls): best accuracy {acc[bi]:.1f}% at epoch {best['epoch']:.0f} of {len(rows)}",
        f"Validation on {sum(n['val'].values()):,} leaves · trained on {sum(n['train'].values()):,} leaves "
        f"({n['train']['diseased']:,} diseased, {n['train']['healthy']:,} healthy) · batch {a.get('batch', '?')} · "
        f"image size {a.get('imgsz', '?')} · {hours(rows[-1]['time'])} on CPU · early stopping (patience {a.get('patience', '?')})",
        OUT / file)
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
    end = (f"Training ran {len(rows) - bi - 1} epochs past the best one without improving (overfitting starts), "
           f"so the epoch-{best['epoch']:.0f} weights are kept." if bi < len(rows) - 1 else
           "Best at the last epoch: a longer run could gain a little.")
    save(fig, path, f"Source: disease_autolabel/models/{name}/results.csv (Ultralytics log, one row per epoch).\n{end}")


BOLL = ROOT / "agrieye_output" / "boll_detector"


def boll():
    """Local boll detector (train_boll_detector.py), once trained."""
    global MODELS
    keep, MODELS = MODELS, BOLL
    try:
        rows, a = load("boll_det"), args("boll_det")
    finally:
        MODELS = keep
    e = [r["epoch"] for r in rows]
    m50, m5095 = [100 * r["metrics/mAP50(B)"] for r in rows], [100 * r["metrics/mAP50-95(B)"] for r in rows]
    bi = max(range(len(rows)), key=lambda i: m50[i]); best = rows[bi]
    data = ROOT / "roboflow_export" / "1_cotton_boll_detection"
    fig, top, bottom, path = figure(
        f"Local boll detector (YOLO11n): mAP50 {m50[bi]:.1f}% at epoch {best['epoch']:.0f} of {len(rows)}",
        f"Validation on {count(data / 'valid' / 'images')} views · trained on {count(data / 'train' / 'images')} views "
        f"(open / closed bolls) · batch {a.get('batch', '?')} · image size {a.get('imgsz', '?')} · {hours(rows[-1]['time'])} on CPU",
        OUT / "boll_detector_training.png")
    top.plot(e, m50, color=ACCENT, linewidth=2.4, label="mAP50")
    top.plot(e, m5095, color=LIGHT, linewidth=2, linestyle="--", label="mAP50-95")
    top.annotate(f"{m50[bi]:.1f}%", (e[bi], m50[bi]), xytext=(0, 9), textcoords="offset points", ha="center",
                 fontsize=9.5, color=INK, fontweight="bold")
    top.axhline(43.6, color=GREY, linewidth=1, linestyle="-.")
    top.text(e[0], 43.6, "  Roboflow R2 (test views): 43.6%", va="bottom", fontsize=8.5, color=QUIET)
    style(top, "Model performance (validation)", pct=True); top.set_ylim(0, max(max(m50), 43.6) * 1.32)
    best_line(top, best["epoch"]); top.legend(frameon=False, fontsize=9.5, loc="lower right")
    loss_panel(bottom[0], rows, "train/box_loss", "val/box_loss", "Box loss", best["epoch"])
    loss_panel(bottom[1], rows, "train/cls_loss", "val/cls_loss", "Class loss", best["epoch"])
    loss_panel(bottom[2], rows, "train/dfl_loss", "val/dfl_loss", "Distribution focal loss", best["epoch"])
    save(fig, path, "Source: agrieye_output/boll_detector/boll_det/results.csv (Ultralytics log, one row per epoch). "
                    "The R2 line is its score on the 18 test views (this chart: 28 validation views).")


def boll_compare():
    """All boll-detector runs side by side: validation curves (left) and test scores (right)."""
    import re
    global MODELS
    runs = []
    for d in sorted(BOLL.glob("boll_det*")):
        if not (d / "results.csv").exists() or d.name.endswith("_new"):
            continue
        keep, MODELS = MODELS, BOLL
        try:
            rows, a = load(d.name), args(d.name)
        finally:
            MODELS = keep
        ep = a.get("epochs", "?")
        rep = ROOT / "sample_results" / "local_boll_detect_test" / f"report_{ep}ep.txt"
        ta = tb = None
        if rep.exists():
            t = rep.read_text(encoding="utf-8")
            m = re.search(r"A\).*?\n\s+mAP@50 ([\d.]+)%", t, re.S); ta = float(m.group(1)) if m else None
            m = re.search(r"B\).*?\n\s+mAP@50 ([\d.]+)%", t, re.S); tb = float(m.group(1)) if m else None
        runs.append(dict(ep=int(ep), active=d.name == "boll_det", rows=rows, ta=ta, tb=tb))
    if len(runs) < 2:
        return
    runs.sort(key=lambda r: r["ep"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=140, gridspec_kw=dict(width_ratios=[1.5, 1]))
    act = next(r for r in runs if r["active"])
    fig.text(0.05, 0.95, f"Boll detector: {act['ep']} epochs is best ({act['tb']:.1f}% test mAP50, default scoring)",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.05, 0.905, "YOLO11n, image size 1280, same 140 views and split for every run; "
             "the kept model is chosen on validation only (differences under 1 point = tie)", fontsize=10, color=QUIET)
    shades = ["#b8bec4", "#8a949e", "#5c6670"]
    ends = []
    for i, r in enumerate(runs):
        e = [x["epoch"] for x in r["rows"]]; m = [100 * x["metrics/mAP50(B)"] for x in r["rows"]]
        col, lw = (ACCENT, 2.6) if r["active"] else (shades[i % 3], 1.8)
        ax.plot(e, m, color=col, linewidth=lw, zorder=3 if r["active"] else 2)
        ends.append((m[-1], e[-1], r, col))
    xmax = max(r["ep"] for r in runs)
    lab_x, ys = xmax + 1.5, []                            # labels in a column right of the curves, kept apart
    for v, ex, r, col in sorted(ends, key=lambda t: t[0]):
        y = max(v, ys[-1] + 2.2) if ys else v
        ys.append(y)
        ax.plot([ex, lab_x - 0.3], [v, y], color=col, linewidth=0.8, linestyle=":")
        ax.text(lab_x, y, f"{r['ep']} epochs" + (" (kept)" if r["active"] else ""), va="center", fontsize=9.5,
                color=INK if r["active"] else QUIET, fontweight="bold" if r["active"] else "normal")
    style(ax, "Validation mAP50 per epoch (28 views)", pct=True); ax.set_xlim(0, xmax + 10)
    x = range(len(runs)); w = 0.36
    for k, (key, lab, off) in enumerate((("tb", "Ultralytics default", -w / 2), ("ta", "same scoring as R2", w / 2))):
        vals = [r[key] or 0 for r in runs]
        cols = [(ACCENT if r["active"] else GREY) for r in runs]
        bars = bx.bar([i + off for i in x], vals, width=w, color=cols, alpha=1.0 if k == 0 else 0.45, label=lab)
        for b_, v in zip(bars, vals):
            bx.text(b_.get_x() + b_.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    bx.axhline(43.6, color=GREY, linestyle="-.", linewidth=1, label="Roboflow R2: 43.6% (same scoring as R2)")
    bx.set_xticks(list(x), [f"{r['ep']} epochs" for r in runs])
    style(bx, "Test mAP50 (18 views)", pct=True); bx.set_xlabel(""); bx.set_ylim(0, 65); bx.set_xlim(-0.65, len(runs) - 0.35)
    bx.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.1, wspace=0.18)
    path = OUT / "boll_detector_epochs_compare.png"
    import time
    for attempt in range(5):
        try:
            fig.savefig(path); break
        except OSError:
            time.sleep(2)
    plt.close(fig); print(path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    l1(); l2()
    if (BOLL / "boll_det" / "results.csv").exists():
        boll()
        boll_compare()
    if (MODELS / "disease_cls_v3" / "results.csv").exists():          # v3 (field + public leaves), once trained
        l2("disease_cls_v3", "Disease classifier v3", "L2_v3_disease_classifier_training.png")
    print(f"\nFiles: {OUT}")
