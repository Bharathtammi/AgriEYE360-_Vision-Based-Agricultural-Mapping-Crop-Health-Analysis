"""
Improve the boll detector on ONE visit (build_visit_split.py), one experiment at a time; then test on the other visits.

  python train_visit.py run --name base                       # train one experiment (old settings: 30 ep, 1280 px, YOLO11n)
  python train_visit.py run --name ep80 --epochs 80           # ... other settings: --imgsz --model yolo11s.pt --single
  python train_visit.py run --name small24 --drop-small 24   # bolls < 24 px greyed out + unlabelled in train/valid
  python train_visit.py run --name tile --epochs 80 --tile  # train on 2x2 overlapping full-resolution 800 px tiles
  python train_visit.py score                                 # table: every experiment on the visit's TEST photos (+ valid)
  python train_visit.py final --name ep80                     # ONLY at the end: the chosen experiment on every other visit

Scores: 2 classes (open / closed) and 1 class (boll) - all sizes and with bolls < 24 px ignored (rescore_boll_detector).
The active model (boll_det) is not touched. Output: agrieye_output/boll_detector/visit_<date>/runs/<name>/, scores.txt
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

import eval_roboflow_detect as E
import rescore_boll_detector as R

ROOT = Path(__file__).resolve().parent
HOME = ROOT / "agrieye_output" / "boll_detector"


TILE, STARTS = 800, (0, 640)                             # 2 x 2 tiles of 800 px over 1440 px, 160 px overlap


def make_tiles(V, oversample=0):
    """Train + valid cut into 4 overlapping full-resolution tiles each. A boll goes into a tile if at least half of
    it is inside (box clipped to the tile). oversample = extra copies of every TRAIN tile that has an open boll."""
    D = V / "variants" / "tiles"
    if D.exists():
        shutil.rmtree(D)
    n = 0
    for s in ("train", "valid"):
        for sub in ("images", "labels"):
            (D / s / sub).mkdir(parents=True, exist_ok=True)
        for f in sorted((V / s / "images").glob("*.jpg")):
            im = Image.open(f).convert("RGB")
            g = E.ground_truth(V / s / "labels" / f"{f.stem}.txt", 1440, 1440)
            for ty in STARTS:
                for tx in STARTS:
                    lines = []
                    for c, (x1, y1, x2, y2) in g:
                        a1, b1, a2, b2 = max(x1, tx), max(y1, ty), min(x2, tx + TILE), min(y2, ty + TILE)
                        if a2 <= a1 or b2 <= b1 or (a2 - a1) * (b2 - b1) < 0.5 * (x2 - x1) * (y2 - y1):
                            continue
                        lines.append(f"{c} {((a1 + a2) / 2 - tx) / TILE:.5f} {((b1 + b2) / 2 - ty) / TILE:.5f} "
                                     f"{(a2 - a1) / TILE:.5f} {(b2 - b1) / TILE:.5f}")
                    name = f"{f.stem}_t{tx}_{ty}"
                    im.crop((tx, ty, tx + TILE, ty + TILE)).save(D / s / "images" / f"{name}.jpg", quality=95)
                    (D / s / "labels" / f"{name}.txt").write_text(chr(10).join(lines) + (chr(10) if lines else ""), encoding="utf-8")
                    n += 1
                    if s == "train" and oversample and any(l.startswith("0 ") for l in lines):
                        for k in range(oversample):
                            shutil.copy2(D / s / "images" / f"{name}.jpg", D / s / "images" / f"{name}_x{k}.jpg")
                            shutil.copy2(D / s / "labels" / f"{name}.txt", D / s / "labels" / f"{name}_x{k}.txt")
                            n += 1
    NL = chr(10)
    (D / "data.yaml").write_text(f"path: {D.as_posix()}{NL}train: train/images{NL}val: valid/images{NL}"
                                 f"nc: 2{NL}names: ['open_boll', 'closed_boll']{NL}", encoding="utf-8")
    print(f"{n} tiles of {TILE} px -> {D}")
    return D / "data.yaml"


def predict_tiled(m, f, single):
    """Predict the 4 tiles, shift back, drop boxes cut by an inner tile edge, merge duplicates (overlap 0.5)."""
    im = Image.open(f).convert("RGB"); out = []
    for ty in STARTS:
        for tx in STARTS:
            r = m.predict(im.crop((tx, ty, tx + TILE, ty + TILE)), imgsz=TILE, conf=0.01, iou=0.5, max_det=1000,
                          device="cpu", verbose=False)[0]
            for b, sc, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                x1, y1, x2, y2 = b
                if (x1 < 2 and tx > 0) or (y1 < 2 and ty > 0) or (x2 > TILE - 2 and tx + TILE < 1440) or                         (y2 > TILE - 2 and ty + TILE < 1440):
                    continue                             # cut by the tile edge: the neighbouring tile has it whole
                out.append((1 if single else int(c), float(sc), (x1 + tx, y1 + ty, x2 + tx, y2 + ty)))
    return [(c, s, list(b)) for c, s, b in E.nms(out, 0.5)]


def predict(run, folder, imgsz, single, tile=False):
    """Predictions of one experiment on one folder of images (cached in the run folder)."""
    cf = run / f"pred_{folder.parent.name}.json"         # test / valid / 20260731 ...
    w = run / "weights" / "best.pt"
    if cf.exists() and cf.stat().st_mtime > w.stat().st_mtime:
        return {k: [(c, s, tuple(b)) for c, s, b in v] for k, v in json.loads(cf.read_text()).items()}
    from ultralytics import YOLO
    m = YOLO(str(w))
    out = {}
    for f in sorted(folder.glob("*.jpg")):
        if tile:
            out[f.stem] = predict_tiled(m, f, single); continue
        r = m.predict(str(f), imgsz=imgsz, conf=0.01, iou=0.5, max_det=1000, device="cpu", verbose=False)[0]
        out[f.stem] = [(1 if single else int(c), float(s), [float(v) for v in b]) for b, s, c in
                       zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    cf.write_text(json.dumps(out))
    return {k: [(c, s, tuple(b)) for c, s, b in v] for k, v in out.items()}


def drop_small(V, px):
    """Copy of train + valid where every boll smaller than px (longer side) is painted grey and its label removed,
    so the model is neither taught it is a boll nor that it is background. Test pictures are not changed
    (the '>=24px' score columns ignore small bolls there)."""
    from PIL import ImageDraw
    D = V / "variants" / f"small{px}"
    n = 0
    for s in ("train", "valid"):
        for sub in ("images", "labels"):
            (D / s / sub).mkdir(parents=True, exist_ok=True)
        for f in sorted((V / s / "images").glob("*.jpg")):
            im = Image.open(f).convert("RGB"); d = ImageDraw.Draw(im); keep = []
            for l in (V / s / "labels" / f"{f.stem}.txt").read_text().splitlines():
                if not l.strip():
                    continue
                c, x, y, w, h = map(float, l.split())
                if max(w, h) * 1440 < px:
                    d.rectangle(((x - w / 2) * 1440 - 2, (y - h / 2) * 1440 - 2, (x + w / 2) * 1440 + 2,
                                 (y + h / 2) * 1440 + 2), fill=(114, 114, 114)); n += 1
                else:
                    keep.append(l)
            im.save(D / s / "images" / f.name, quality=95)
            (D / s / "labels" / f"{f.stem}.txt").write_text(chr(10).join(keep) + (chr(10) if keep else ""), encoding="utf-8")
    NL = chr(10)
    (D / "data.yaml").write_text(f"path: {D.as_posix()}{NL}train: train/images{NL}val: valid/images{NL}"
                                 f"nc: 2{NL}names: ['open_boll', 'closed_boll']{NL}", encoding="utf-8")
    print(f"greyed out {n} bolls < {px} px in train + valid -> {D}")
    return D / "data.yaml"


def gts(folder):
    """Hand boxes; plus IGNORE ZONES (hidden / unclear bolls from build_label_review.py) as (cls, box, True)."""
    out = {}
    for f in folder.glob("*.jpg"):
        g = E.ground_truth(folder.parent / "labels" / f"{f.stem}.txt", 1440, 1440)
        ig = folder.parent / "ignore" / f"{f.stem}.txt"
        if ig.exists():
            g += [(c, b, True) for c, b in E.ground_truth(ig, 1440, 1440)]
        out[f.stem] = g
    return out


IOU = 0.5


def row(name, p, g, single):
    a = R.score(p, g, False, 0, IOU); a24 = R.score(p, g, False, 24, IOU)
    b = R.score(p, g, True, 0, IOU); b24 = R.score(p, g, True, 24, IOU)
    two = (f"{100 * a['map50']:>6.1f}%{100 * a24['map50']:>7.1f}%"
           f"{100 * a['per']['open_boll']:>7.1f}%{100 * a['per']['closed_boll']:>7.1f}%") if not single else f"{'(trained as 1 class)':>29}"
    return (f"{name:<22}{two}{100 * b['map50']:>8.1f}%{100 * b24['map50']:>7.1f}%"
            f"{100 * b24['P']:>7.1f}%{100 * b24['R']:>7.1f}%")


HEAD = (f"{'':<22}{'-- 2 classes (open/closed) --':>29}{'---- 1 class (boll) ----':>31}\n"
        f"{'experiment':<22}{'mAP50':>7}{'>=24px':>7}{'open':>8}{'closed':>8}{'mAP50':>9}{'>=24px':>7}{'P':>8}{'R':>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score", "final"])
    ap.add_argument("--visit", default="20260804")
    ap.add_argument("--name")
    ap.add_argument("--folder", help="dataset folder in agrieye_output/boll_detector instead of visit_<date>, e.g. data_photo_split")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--single", action="store_true", help="train one class 'boll' (open + closed together)")
    ap.add_argument("--tile", action="store_true", help="train and predict on 2x2 overlapping 800 px tiles")
    ap.add_argument("--cls-pw", type=float, default=0.0, help="Ultralytics class weights power (0 off, 1 = inverse frequency)")
    ap.add_argument("--oversample-open", type=int, default=0, help="with --tile: extra copies of train tiles with an open boll")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run (same --name and other options)")
    ap.add_argument("--iou", type=float, default=0.5, help="score: overlap needed to count a boll as found (0.5 standard)")
    ap.add_argument("--drop-small", type=int, default=0, help="grey out + unlabel bolls smaller than this (px) in train/valid")
    a = ap.parse_args()
    global IOU
    IOU = a.iou
    tag = "" if a.iou == 0.5 else f"_iou{int(round(100 * a.iou))}"
    V = HOME / (a.folder or f"visit_{a.visit}")
    runs = V / "runs"
    if a.cmd == "run":
        from ultralytics import YOLO
        out = runs / a.name
        cfg = dict(epochs=a.epochs, imgsz=a.imgsz, model=a.model, batch=a.batch, single=a.single, drop_small=a.drop_small, tile=a.tile,
                   cls_pw=a.cls_pw, oversample_open=a.oversample_open)
        if a.tile:
            a.imgsz = cfg["imgsz"] = TILE
        if a.resume:                                     # interrupted: continue from the last saved epoch
            YOLO(str(out / "weights" / "last.pt")).train(resume=True)
        else:
          y = make_tiles(V, a.oversample_open) if a.tile else drop_small(V, a.drop_small) if a.drop_small else V / "data.yaml"
          YOLO(a.model).train(
            data=str(y), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device="cpu", workers=2,
            patience=a.epochs, hsv_v=0.0, hsv_s=0.3, hsv_h=0.01, fliplr=0.5, degrees=10, mosaic=1.0, seed=0,
            single_cls=a.single, cls_pw=a.cls_pw, project=str(runs), name=a.name, exist_ok=True, plots=True, verbose=False)
        (out / "experiment.json").write_text(json.dumps(cfg))
        a.cmd = "score"
    if a.cmd == "score":
        lines = [f"One-visit development: visit {a.visit} - scored on its TEST photos (valid in brackets below)", HEAD]
        tg, vg = gts(V / "test" / "images"), gts(V / "valid" / "images")
        val = []
        for run in sorted(runs.glob("*/experiment.json"), key=lambda f: f.stat().st_mtime):
            cfg = json.loads(run.read_text()); d = run.parent
            lines.append(row(d.name, predict(d, V / "test" / "images", cfg["imgsz"], cfg["single"], cfg.get("tile", False)), tg, cfg["single"]))
            val.append(row(d.name, predict(d, V / "valid" / "images", cfg["imgsz"], cfg["single"], cfg.get("tile", False)), vg, cfg["single"]))
        lines += ["", "VALID photo (used by training to pick the best epoch):", HEAD] + val
        lines += ["", f"P / R at confidence {E.SHOW_CONF}, bolls >= 24 px. Test = {len(tg)} pictures, valid = {len(vg)}."]
        lines += [f"A boll counts as found at overlap (IoU) >= {IOU}. Hidden/unclear bolls (ignore/ folder) count neither way."]
        (V / f"scores{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines) + f"\n\nSaved: {V / f'scores{tag}.txt'}")
    if a.cmd == "final":
        d = runs / a.name; cfg = json.loads((d / "experiment.json").read_text())
        lines = [f"FINAL: experiment '{a.name}' (developed on visit {a.visit}) on every OTHER visit", HEAD]
        sets = [("this visit, test", V / "test" / "images")] + [
            (f"visit {o.name}", o / "images") for o in sorted((V / "other_visits").iterdir())]
        for label, folder in sets:
            lines.append(row(label, predict(d, folder, cfg["imgsz"], cfg["single"], cfg.get("tile", False)), gts(folder), cfg["single"]))
        lines += ["", "21 Jul (and half of 22 Jul) have automatic labels only - not hand-checked."]
        (d / "final.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines) + f"\n\nSaved: {d / 'final.txt'}")


if __name__ == "__main__":
    main()
