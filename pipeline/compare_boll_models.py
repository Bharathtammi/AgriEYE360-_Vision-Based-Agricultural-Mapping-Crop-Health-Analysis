"""
FAIR comparison of boll detectors on the views that NO model trained on, graded against TWO answer keys.

  python compare_boll_models.py                                   # local models only
  $env:ROBOFLOW_API_KEY="..."   (PowerShell; never saved)
  python compare_boll_models.py --rf cotton-bolls-all-povs/1      # + your Roboflow model

Views : agrieye_output/boll_detector/fair_views.csv - 24 hand-checked views (6 photos) that are in Roboflow's
        VALID split (not trained on) and not in the local big training set.
Keys  : HAND = your hand-checked boxes (incl. the small bolls added by hand)
        AUTO = the automatic boxes (SAM 3 + R1) of the same views - the kind of labels Roboflow scores itself on
Models: --local boll_det [more run folders in agrieye_output/boll_detector]   (default: the active model)
Score : the same as eval_roboflow_detect.py (scoring A: duplicates merged at overlap 0.5).
Output: sample_results/boll_model_comparison/report.txt (+ raw Roboflow answers cached, so a re-run costs nothing)
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

from PIL import Image

import eval_roboflow_detect as E

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "roboflow_export" / "1_cotton_boll_detection"
AUTO = ROOT / "agrieye_output" / "pov_all_labels" / "bolls_dataset" / "labels"
HOME = ROOT / "agrieye_output" / "boll_detector"
OUT = ROOT / "sample_results" / "boll_model_comparison"


def find_preds(obj):
    """Roboflow Workflow outputs are named by the workflow; find the first detections block inside, whatever its name."""
    if isinstance(obj, dict):
        if isinstance(obj.get("predictions"), list):
            return obj
        vals = obj.values()
    elif isinstance(obj, list):
        vals = obj
    else:
        return None
    for v in vals:
        r = find_preds(v)
        if r is not None:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", nargs="*", default=["boll_det"])
    ap.add_argument("--rf", help="Roboflow model id, e.g. cotton-bolls-all-povs/1")
    ap.add_argument("--workflow", nargs="?", const="bharaths-workspace-3fhlw/bollsdatasetdraftsam3only-1-yolo26n-t1-vbollsdatasetdraftsam3only-1-yolo26n-t1-logic",
                    help="run the model through a Roboflow Workflow (workspace/workflow_id) instead of model_id")
    ap.add_argument("--tile", type=int, default=1, help="send each view as N x N tiles (use if the Roboflow "
                    "dataset version was tiled, e.g. --tile 2 for 2x2)")
    ap.add_argument("--api", default="https://serverless.roboflow.com")
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()
    views = list(csv.DictReader(open(HOME / "fair_views.csv", encoding="utf-8")))
    imgs, keys = {}, {"HAND": {}, "AUTO": {}}
    for v in views:
        f = OLD / v["hand_split"] / "images" / v["hand_image"]
        W, H = Image.open(f).size
        k = f.stem; imgs[k] = f
        keys["HAND"][k] = E.ground_truth(OLD / v["hand_split"] / "labels" / f"{k}.txt", W, H)
        keys["AUTO"][k] = E.ground_truth(AUTO / f"{Path(v['pov_view']).stem}.txt", W, H)
    preds = {}
    for name in a.local:
        w = HOME / name / "weights" / "best.pt"
        if not w.exists():
            print(f"skip {name}: {w} not found"); continue
        cf = OUT / f"raw_local_{name}.json"             # cached: a re-run needs no torch (Application Control
        stamp = f"{w.stat().st_mtime:.0f}_{a.imgsz}"      # can block torch DLLs in a normal PowerShell window)
        c = json.loads(cf.read_text()) if cf.exists() else {}
        if c.get("stamp") != stamp or set(c.get("preds", {})) != set(imgs):
            from ultralytics import YOLO
            m = YOLO(str(w))
            c = {"stamp": stamp, "preds": {k: [(int(cl), float(s), list(map(float, b))) for b, s, cl in
                                               zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
                                           for k, f in imgs.items()
                                           for r in [m.predict(str(f), imgsz=a.imgsz, conf=0.01, iou=0.5,
                                                               max_det=1000, device="cpu", verbose=False)[0]]}}
            OUT.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps(c))
        preds[f"local {name}"] = {k: [(cl, s, tuple(b)) for cl, s, b in v] for k, v in c["preds"].items()}
    if a.workflow and not a.rf:
        a.rf = "workflow"
    if a.rf:
        OUT.mkdir(parents=True, exist_ok=True)
        T = a.tile
        tag = ("wf_" + a.workflow.split("/")[-1][:40]) if a.workflow else a.rf.replace("/", "_")
        cf = OUT / f"raw_{tag}{f'_tile{T}' if T > 1 else ''}.json"

        def ask(x):                                      # one image (path or PIL) -> {"predictions": [...]}
            if not a.workflow:
                return cl.infer(x, model_id=a.rf)
            ws, wf = a.workflow.split("/", 1)
            return find_preds(cl.run_workflow(workspace_name=ws, workflow_id=wf, images={"image": x})) or {}
        cache = json.loads(cf.read_text()) if cf.exists() else {}
        cl = None
        for k, f in imgs.items():
            if k not in cache:
                if cl is None:                               # the key is only needed for views not cached yet
                    key = os.environ.get("ROBOFLOW_API_KEY", "").strip().strip('"').strip("'")
                    if not key:
                        sys.exit('Set your key first (PowerShell):  $env:ROBOFLOW_API_KEY="xxxxxxxx"')
                    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
                    cl = InferenceHTTPClient(api_url=a.api, api_key=key)
                    cl.configure(InferenceConfiguration(confidence_threshold=0.01, iou_threshold=0.5,
                                                        max_detections=1000, api_key_transport="header"))
                try:
                    if T == 1:
                        cache[k] = ask(str(f))
                    else:                                # N x N tiles; answers kept per tile with their offset
                        im = Image.open(f).convert("RGB"); W, H = im.size; tw, th = W // T, H // T
                        cache[k] = [dict(dx=i * tw, dy=j * th, r=ask(im.crop((i * tw, j * th, (i + 1) * tw,
                                                                                    (j + 1) * th))))
                                    for j in range(T) for i in range(T)]
                except Exception as e:
                    sys.exit(f"Roboflow could not run '{a.rf}': {str(e).split('?api_key')[0][:300]}")
                cf.write_text(json.dumps(cache))
        def merged(c):                                   # tiles -> one view; boxes cut at tile edges stay split
            return E.parse(c) if isinstance(c, dict) else [b for t in c for b in E.parse(t["r"], t["dx"], t["dy"])]
        preds[f"Roboflow{' workflow' if a.workflow else ''}{f' tiles {T}x{T}' if T > 1 else ''}"] = {k: merged(cache[k]) for k in imgs}
    n_open = {kk: sum(c == 0 for g in keys[kk].values() for c, _ in g) for kk in keys}
    n_cl = {kk: sum(c == 1 for g in keys[kk].values() for c, _ in g) for kk in keys}
    lines = [f"FAIR boll detector comparison: {len(imgs)} views no model trained on "
             f"(6 photos; in Roboflow's validation split, not in the local training set)",
             f"Answer keys: HAND = hand-checked (open {n_open['HAND']}, closed {n_cl['HAND']})   "
             f"AUTO = SAM 3 + R1 (open {n_open['AUTO']}, closed {n_cl['AUTO']})",
             "Scoring A (duplicates merged at overlap 0.5; the same as the R2 / local test reports)", "",
             f"{'model':<38}{'key':<6}{'mAP50':>7}{'mAP50-95':>10}{'open AP50':>11}{'closed AP50':>13}{'P':>7}{'R':>7}"]
    for name, p in preds.items():
        for kk in ("HAND", "AUTO"):
            _, s = E.evaluate(p, keys[kk])
            c = s["cls"]
            lines.append(f"{name:<38}{kk:<6}{100 * s['map50']:>6.1f}%{100 * s['map']:>9.1f}%"
                         f"{100 * c['open_boll']['ap50']:>10.1f}%{100 * c['closed_boll']['ap50']:>12.1f}%"
                         f"{100 * s['P']:>6.1f}%{100 * s['R']:>6.1f}%")
    seen = sum(v["hand_split"] == "train" for v in views)
    for name in a.local:
        mark = HOME / name / "trained_on.txt"              # written by train_boll_detector.py ("big" / "small")
        if not (mark.exists() and "big" in mark.read_text()):
            lines.append(f"!! local {name} was trained on the SMALL hand-checked set: it saw {seen} of these "
                         f"{len(views)} views in training -> its row is NOT a fair score.")
    lines += ["", f"P / R at confidence {E.SHOW_CONF}. HAND vs AUTO for the same model shows how much of a score "
                  "comes from the answer key, not the model."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines) + f"\n\nSaved: {OUT / 'report.txt'}")


if __name__ == "__main__":
    main()
