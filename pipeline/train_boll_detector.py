"""
Local cotton boll DETECTOR (YOLO11n) trained on this laptop - the local counterpart of Roboflow's R2.

  python train_boll_detector.py train [--epochs 30] [--imgsz 1280] [--batch 4]
  python train_boll_detector.py test          # only test the active model on the 18 test views

Data: roboflow_export/1_cotton_boll_detection (the same 140 hand-checked views and the same split as R2:
train 94 / valid 28 / test 18; classes open_boll, closed_boll).
Training: YOLO11n from COCO weights; no brightness augmentation (white clouds look like open cotton).
After training (train_guard.py): new vs current model on the 28 validation views (box mAP50, tie mAP50-95),
the better one is kept as agrieye_output/boll_detector/boll_det/ ; training graphs redrawn.
Then the active model is tested on the 18 TEST views exactly like R2 (eval_roboflow_detect.py metrics).

The model is stored OUTSIDE agrieye_output/models/objects/, so 'agrieye.py analyze' does NOT start using it:
the field results keep coming from SAM 3 + your labels + R1.
Output: agrieye_output/boll_detector/boll_det/ (model, results.csv), sample_results/local_boll_detect_test/report.txt
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "roboflow_export" / "1_cotton_boll_detection"
HOME = ROOT / "agrieye_output" / "boll_detector"
NAME = "boll_det"
YAML = HOME / "data.yaml"
TEST_OUT = ROOT / "sample_results" / "local_boll_detect_test"
R2 = dict(map50=43.6, map=21.7, open_ap=26.8, closed_ap=60.4, P=73.6, R=39.0)     # Roboflow R2 on the same 18 views


def data_yaml():
    HOME.mkdir(parents=True, exist_ok=True)
    YAML.write_text(f"path: {DATA.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                    "nc: 2\nnames: ['open_boll', 'closed_boll']\n", encoding="utf-8")
    return YAML


def train(a):
    from ultralytics import YOLO
    import train_guard
    data_yaml()
    left = HOME / f"{NAME}_new"
    if left.exists():                                    # interrupted earlier run: set aside, never mixed in
        from datetime import datetime
        import shutil
        shutil.move(str(left), str(HOME / f"{NAME}_interrupted_{datetime.now():%Y%m%d_%H%M}"))
    YOLO("yolo11n.pt").train(
        data=str(YAML), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device="cpu", workers=2,
        patience=a.epochs,                               # run all epochs (the curve is the point of this try)
        hsv_v=0.0, hsv_s=0.3, hsv_h=0.01,                # no brightness change: clouds vs open cotton
        fliplr=0.5, degrees=10, mosaic=1.0,
        project=str(HOME), name=f"{NAME}_new", exist_ok=True, plots=True, verbose=False)
    import time
    t0 = time.time()
    train_guard.decide(NAME, "det", YAML, a.imgsz, models=HOME)
    test(a.imgsz)                                        # the active (kept) model -> report.txt
    rejected = [d for d in HOME.glob(f"{NAME}_rejected_*") if d.stat().st_mtime >= t0 - 5]
    for d in rejected:                                   # the run that was NOT kept is tested too, for the record
        test(a.imgsz, run=d, active=False)


def test(imgsz=1280, run=None, active=True):
    """The 18 test views, two scorings:
    - same as R2 (eval_roboflow_detect.py): duplicate boxes merged at overlap 0.5, our AP formula
    - Ultralytics default val on the test split (overlap 0.7), the same setting as the training stats.
    Writes report_<epochs>ep.txt for every model tested; report.txt = the active model."""
    from ultralytics import YOLO
    from PIL import Image
    import eval_roboflow_detect as E
    run = Path(run) if run else HOME / NAME
    w = run / "weights" / "best.pt"
    if not w.exists():
        raise SystemExit(f"No trained detector yet: {w}")
    model = YOLO(str(w))
    gts, preds = {}, {}
    for f in sorted((DATA / "test" / "images").glob("*.jpg")):
        W, H = Image.open(f).size
        gts[f.stem] = E.ground_truth(DATA / "test" / "labels" / f"{f.stem}.txt", W, H)
        r = model.predict(str(f), imgsz=imgsz, conf=0.01, iou=0.5, max_det=1000, device="cpu", verbose=False)[0]
        preds[f.stem] = [(int(c), float(s), tuple(float(v) for v in b))
                         for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    rows, s = E.evaluate(preds, gts)
    a = {}
    argf = run / "args.yaml"
    if argf.exists():
        a = {l.split(":", 1)[0]: l.split(":", 1)[1].strip() for l in argf.read_text().splitlines() if ":" in l and not l.startswith(" ")}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:          # Ultralytics' own scoring, default settings
        u = model.val(data=str(data_yaml()), split="test", imgsz=imgsz, batch=4, device="cpu", plots=False,
                      verbose=False, project=tmp, name="t", exist_ok=True)
    ub = u.box
    ep = a.get("epochs", "?")
    lines = [f"Local YOLO11n boll detector ({ep} epochs, image size {a.get('imgsz', imgsz)}) on the 18 test views "
             f"never used in training   [{'ACTIVE model' if active else 'not kept (' + run.name + ')'}]", "",
             "A) Same scoring as Roboflow R2 (duplicate boxes merged at overlap 0.5):"] + rows + ["",
             "B) Ultralytics default scoring (overlap 0.7, as in the training stats):",
             f"  mAP@50 {100 * ub.map50:.1f}%   mAP@50:95 {100 * ub.map:.1f}%   precision {100 * ub.mp:.1f}%   recall {100 * ub.mr:.1f}%   "
             f"open AP@50 {100 * ub.ap50[0]:.1f}%   closed AP@50 {100 * ub.ap50[1]:.1f}%", "",
             f"Roboflow R2 on the same views, scoring A: mAP@50 {R2['map50']}%  mAP@50:95 {R2['map']}%  "
             f"open AP {R2['open_ap']}%  closed AP {R2['closed_ap']}%  precision {R2['P']}%  recall {R2['R']}%",
             "Two-step pipeline (SAM 3 finds, R1 judges): 97.7% open/closed on boll crops (a different task)."]
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    (TEST_OUT / f"report_{ep}ep.txt").write_text("\n".join(lines), encoding="utf-8")
    if not active:
        print("\n".join(lines)); return
    (TEST_OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(TEST_OUT / "predictions.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2"])
        for img, ps in preds.items():
            for c, conf, b in ps:
                wr.writerow([img, E.NAMES[c], round(conf, 3)] + [round(v, 1) for v in b])
    print("\n".join(lines)); print(f"\nFiles: {TEST_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else test(a.imgsz)
