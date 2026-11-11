"""
Keep the BEST model after every training run, and redraw the training graphs.

Used by pipeline.py (traincls, trainleaf) and train_disease_v3.py (train). A training run never writes into the
active model folder any more:

  1. the new model is trained into  <models>/<name>_new/
  2. the NEW and the CURRENT model are both tested on the SAME validation data (Ultralytics val):
       classifier   -> top-1 accuracy        leaf finder -> mask mAP50
  3. the better one becomes  <models>/<name>/   (the folder every script reads)
     the other one is kept, never deleted:
       new is better  -> old model moved to  <models>/<name>_previous_<date-time>/
       old is better  -> new model moved to  <models>/<name>_rejected_<date-time>/
  4. training_graphs.py redraws the graphs of the active models (sample_results/training_graphs/)
  5. the copies in Results/models/ are refreshed (hard links / copies of the active best.pt)
  A line is added to <models>/training_history.csv for every decision.

Manual use (e.g. to re-test after changing the validation set):
  python train_guard.py decide disease_cls      # compares disease_cls_new with disease_cls (if both exist)
"""
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "disease_autolabel" / "models"
HISTORY = MODELS / "training_history.csv"
RESULTS_COPIES = {"disease_cls": ROOT / "Results" / "models" / "disease_classifier_v2_best.pt",
                  "leaf_seg": ROOT / "Results" / "models" / "leaf_finder_best.pt"}


def score(weights, kind, data, imgsz):
    """Test one model on the validation split -> (main score, tie-breaker), both 0-1.
    cls: (accuracy, diseased recall) on <data>/val/<class>/ with the pipeline's own classify() (label at p >= 0.5)
    seg: (mask mAP50, mask mAP50-95) from Ultralytics val on <data> (a data.yaml)."""
    from ultralytics import YOLO
    m = YOLO(str(weights))
    if kind == "cls":
        from PIL import Image
        import pipeline as P
        files = [(f, c) for c in ("diseased", "healthy") for f in sorted((Path(data) / "val" / c).glob("*"))
                 if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        out = []
        for i in range(0, len(files), 64):
            out += P.classify(m, [Image.open(f).convert("RGB") for f, _ in files[i:i + 64]])
        ok = sum(l == c for (l, _), (_, c) in zip(out, files))
        dis = [(l, c) for (l, _), (_, c) in zip(out, files) if c == "diseased"]
        return ok / max(1, len(files)), sum(l == "diseased" for l, _ in dis) / max(1, len(dis))
    r = m.val(data=str(data), split="val", imgsz=imgsz, batch=4 if kind == "det" else 16, device="cpu", plots=False,
              verbose=False, project=str(MODELS / "_val_tmp"), name="val", exist_ok=True)
    if kind == "det":                                    # boll detector: box mAP50, tie-breaker box mAP50-95
        return float(r.box.map50), float(r.box.map)
    return float(r.seg.map50), float(r.seg.map)


def refresh_copy(name):
    dst = RESULTS_COPIES.get(name)
    src = MODELS / name / "weights" / "best.pt"
    if not dst or not dst.parent.exists() or not src.exists():
        return
    if dst.exists():
        dst.unlink()                                   # only the link/copy; the model itself stays
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def decide(name, kind, data, imgsz, models=None):
    """Compare <name>_new with <name>; keep the better as <name>. Returns a short message.
    models: folder holding the model folders (default disease_autolabel/models)."""
    global MODELS, HISTORY
    if models is not None:
        MODELS, HISTORY = Path(models), Path(models) / "training_history.csv"
    new, cur = MODELS / f"{name}_new", MODELS / name
    new_w, cur_w = new / "weights" / "best.pt", cur / "weights" / "best.pt"
    if not new_w.exists():
        raise SystemExit(f"No new model found: {new_w}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    unit, tie = {"cls": ("accuracy", "diseased recall"), "seg": ("mask mAP50", "mask mAP50-95"),
                 "det": ("box mAP50", "box mAP50-95")}[kind]
    print(f"\nChoosing the best model ({unit}, tie-breaker {tie}; same validation data: {data}) ...")
    s_new = score(new_w, kind, data, imgsz)
    s_cur = score(cur_w, kind, data, imgsz) if cur_w.exists() else None
    r4 = lambda s: tuple(round(v, 4) for v in s)         # equal to 0.01 % counts as a tie
    if s_cur is None or r4(s_new) > r4(s_cur):           # full tie -> keep the current model (results stay the same)
        if cur.exists():
            shutil.move(str(cur), str(MODELS / f"{name}_previous_{stamp}"))
        shutil.move(str(new), str(cur))
        kept, other = "NEW", (f"old model kept as {name}_previous_{stamp}" if s_cur is not None else "no old model")
    else:
        shutil.move(str(new), str(MODELS / f"{name}_rejected_{stamp}"))
        kept = "OLD" if r4(s_new) < r4(s_cur) else "OLD (tie)"
        other = f"new model kept as {name}_rejected_{stamp}"
    shutil.rmtree(MODELS / "_val_tmp", ignore_errors=True)
    refresh_copy(name)
    fmt = lambda s: "-" if s is None else f"{100 * s[0]:.1f}% ({tie} {100 * s[1]:.1f}%)"
    msg = (f"{name}: new {fmt(s_new)} vs current {fmt(s_cur)} [{unit}] -> keeping the {kept} model; {other}. "
           f"Active model: {cur_w}")
    first = not HISTORY.exists()
    with open(HISTORY, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if first:
            w.writerow(["date_time", "model", "metric", "new", "current", "kept"])
        w.writerow([stamp, name, f"{unit} / {tie}", fmt(s_new), fmt(s_cur), kept])
    print(msg)
    graphs()
    return msg


def graphs():
    r = subprocess.run([sys.executable, str(ROOT / "training_graphs.py")], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-500:])


if __name__ == "__main__":
    import pipeline as P
    jobs = {"disease_cls": ("cls", P.WORK / "dataset_cls", P.CLS_IMGSZ),
            "leaf_seg": ("seg", P.WORK / "dataset_leaf" / "data.yaml", P.IMGSZ),
            "disease_cls_v3": ("cls", P.WORK / "dataset_cls_v3", P.CLS_IMGSZ)}
    if len(sys.argv) == 3 and sys.argv[1] == "decide" and sys.argv[2] in jobs:
        decide(sys.argv[2], *jobs[sys.argv[2]])
    else:
        print(__doc__)
