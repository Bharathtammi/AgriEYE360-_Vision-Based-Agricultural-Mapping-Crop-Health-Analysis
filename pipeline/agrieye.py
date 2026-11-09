"""
AgriEYE 360 field pipeline. All paths/settings live in agrieye_config.json.

  import       find panoramas in input_dirs, drop duplicates, read date/GPS/heading
  stations     group panoramas into numbered plant stations by GPS -> stations.csv
               (open it, fix any wrong station numbers; your edits are kept)
  slice        cut each panorama into 4 compass-aligned POVs (N, E, S, W)
  autolabel    SAM 3 labels open bolls / closed bolls / flowers on the POVs
  traindet     train a fast YOLO detector on those labels
  analyze      per POV: bolls, flowers (+ leaves and % diseased if those models
               exist) -> results\\detections.csv, growth.csv, growth plots
  streetview   export Street View tour JSON (for agrieye-field-viewer.html)
  report       annotated POVs + one summary sheet per station
  all          import -> stations -> slice -> autolabel -> analyze -> report -> streetview

Run with the venv Python:
  "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" "F:\\Project and completed courses\\Agri EYE 360\\agrieye.py" import
Use --config to point at a different config file.
"""

import argparse
import base64
import csv
import io
import json
import math
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sam3_autolabel import content_hash, find_images, make_tiles  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


# ================================================================== config
class Cfg:
    def __init__(self, path):
        self.path = Path(path)
        text = self.path.read_text(encoding="utf-8")
        # accept Windows paths typed with single backslashes (C:\Folder\file)
        text = re.sub(r'(?<!\\)\\(?![\\"])', r"\\\\", text)
        c = json.loads(text)
        self.c = c
        self.inputs = [Path(p) for p in c["input_dirs"]]
        self.out = Path(c["output_dir"])
        self.sam3 = Path(c.get("sam3_model", ROOT / "sam3.pt"))
        self.pov = c["pov"]
        self.radius = float(c.get("station_radius_m", 1.5))
        self.obj = c["objects"]
        self.sv = c["streetview"]
        # derived paths
        self.catalog = self.out / "catalog.csv"
        self.stations = self.out / "stations.csv"
        self.povs = self.out / "povs"
        self.labels = self.out / "objects" / "sam3_labels.jsonl"
        self.obj_previews = self.out / "objects" / "previews"
        self.det_data = self.out / "dataset_objects"
        self.det_model = self.out / "models" / "objects" / "weights" / "best.pt"
        self.results = self.out / "results"
        self.tours = self.out / "streetview"

    @property
    def classes(self):
        return list(self.obj["prompts"].keys())


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(p, rows, fields):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ============================================================ 1. import
def read_meta(path):
    """date/time, GPS and compass heading from EXIF + GPano XMP."""
    im = Image.open(path)
    W, H = im.size
    ex = im.getexif()
    dt = None
    raw = ex.get(306) or ex.get_ifd(0x8769).get(36867)
    if raw:
        try:
            dt = datetime.strptime(str(raw).strip("\x00")[:19], "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    if dt is None:  # fall back to a timestamp in the file name
        m = re.search(r"(20\d{6})_(\d{6})", path.stem)
        if m:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    lat = lon = None
    g = ex.get_ifd(0x8825)
    if g and 2 in g and 4 in g:
        dms = lambda t: float(t[0]) + float(t[1]) / 60 + float(t[2]) / 3600
        lat = dms(g[2]) * (1 if g.get(1, "N") == "N" else -1)
        lon = dms(g[4]) * (1 if g.get(3, "E") == "E" else -1)
    heading = None
    head = path.open("rb").read(600_000)
    m = re.search(rb'PoseHeadingDegrees="([-\d.]+)"', head) or \
        re.search(rb"<GPano:PoseHeadingDegrees>([-\d.]+)<", head)
    if m:
        heading = float(m.group(1))
    is_pano = abs(W / H - 2) < 0.05
    return dict(width=W, height=H, datetime=dt, lat=lat, lon=lon, heading=heading, panorama=is_pano)


def cmd_import(cfg, args):
    print("Input folders:")
    for p in cfg.inputs:
        print(f"  {p}  {'(missing!)' if not p.exists() else ''}")
    rows, seen, dups = [], {}, 0
    used_ids = set()
    files = [f for src in cfg.inputs for f in find_images([src])]
    print(f"Scanning {len(files)} image files ...")
    for f in files:
        h = content_hash(f)
        if h in seen:
            dups += 1
            continue
        seen[h] = f
        m = read_meta(f)
        pid = m["datetime"].strftime("%Y%m%d_%H%M%S") if m["datetime"] else f.stem
        base, k = pid, 2
        while pid in used_ids:
            pid, k = f"{base}_{k}", k + 1
        used_ids.add(pid)
        rows.append(dict(photo_id=pid, path=str(f),
                         date=m["datetime"].strftime("%Y-%m-%d") if m["datetime"] else "",
                         time=m["datetime"].strftime("%H:%M:%S") if m["datetime"] else "",
                         lat=f"{m['lat']:.7f}" if m["lat"] is not None else "",
                         lon=f"{m['lon']:.7f}" if m["lon"] is not None else "",
                         heading=f"{m['heading']:.2f}" if m["heading"] is not None else "",
                         width=m["width"], height=m["height"], panorama=int(m["panorama"])))
    rows.sort(key=lambda r: (r["date"], r["time"], r["photo_id"]))
    write_csv(cfg.catalog, rows, list(rows[0].keys()))
    n_pano = sum(r["panorama"] for r in rows)
    print(f"\n{len(rows)} unique photos ({n_pano} 360 panoramas), {dups} duplicates skipped")
    print(f"Dates: {sorted({r['date'] for r in rows if r['date']})}")
    missing = [r["photo_id"] for r in rows if not r["lat"]]
    if missing:
        print(f"No GPS: {missing}")
    print(f"Catalog: {cfg.catalog}")


def catalog(cfg, panos_only=True):
    if not cfg.catalog.exists():
        raise SystemExit("Run 'import' first.")
    rows = read_csv(cfg.catalog)
    return [r for r in rows if r["panorama"] == "1"] if panos_only else rows


# =========================================================== 2. stations
def local_xy(lat, lon, lat0, lon0):
    return ((lon - lon0) * 111320 * math.cos(math.radians(lat0)), (lat - lat0) * 110540)


def cmd_stations(cfg, args):
    rows = catalog(cfg)
    if cfg.stations.exists() and not args.force:
        print(f"{cfg.stations} already exists - keeping your edits. Use --force to regenerate.")
        return
    geo = [r for r in rows if r["lat"]]
    lat0 = min(float(r["lat"]) for r in geo)
    lon0 = min(float(r["lon"]) for r in geo)
    pts = {r["photo_id"]: local_xy(float(r["lat"]), float(r["lon"]), lat0, lon0) for r in geo}
    # greedy clustering: each photo joins the nearest station centre within radius
    centres = []  # [x, y, n]
    assign = {}
    for r in sorted(geo, key=lambda r: (r["date"], r["time"])):
        x, y = pts[r["photo_id"]]
        best, bd = None, cfg.radius
        for i, (cx, cy, n) in enumerate(centres):
            d = math.hypot(x - cx, y - cy)
            if d <= bd:
                best, bd = i, d
        if best is None:
            centres.append([x, y, 1])
            best = len(centres) - 1
        else:
            cx, cy, n = centres[best]
            centres[best] = [(cx * n + x) / (n + 1), (cy * n + y) / (n + 1), n + 1]
        assign[r["photo_id"]] = best
    # number stations west->east, then south->north (stable, readable order)
    order = sorted(range(len(centres)), key=lambda i: (round(centres[i][0]), centres[i][1]))
    name = {i: f"P{k + 1:02d}" for k, i in enumerate(order)}
    out = []
    for r in rows:
        st = name[assign[r["photo_id"]]] if r["photo_id"] in assign else ""
        x, y = pts.get(r["photo_id"], ("", ""))
        out.append(dict(station=st, photo_id=r["photo_id"], date=r["date"], time=r["time"],
                        x_m=f"{x:.1f}" if x != "" else "", y_m=f"{y:.1f}" if y != "" else "",
                        lat=r["lat"], lon=r["lon"], heading=r["heading"], path=r["path"]))
    write_csv(cfg.stations, out, list(out[0].keys()))
    print(f"{len(centres)} stations from {len(geo)} geotagged panoramas (radius {cfg.radius} m)\n")
    by_st = defaultdict(list)
    for r in out:
        by_st[r["station"] or "(none)"].append(r["date"][5:])
    for st in sorted(by_st):
        print(f"  {st:7s} visits: {', '.join(by_st[st])}")
    print(f"\nCheck and edit the 'station' column in: {cfg.stations}")
    print("(Photos without GPS have an empty station - fill it in or leave empty to skip.)")


def stations(cfg):
    if not cfg.stations.exists():
        raise SystemExit("Run 'stations' first.")
    return [r for r in read_csv(cfg.stations) if r["station"].strip()]


# ============================================================== 3. slice
def imread(path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img, q=92):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    buf.tofile(str(path))


def equirect_to_perspective(img, fov_deg, yaw_deg, pitch_deg, size):
    """Same projection as ss-agent_4pov.py, float32 + square output."""
    import cv2
    h, w = img.shape[:2]
    f = (size / 2) / math.tan(math.radians(fov_deg) / 2)
    xs = np.linspace(-size / 2, size / 2, size, dtype=np.float32)
    x, y = np.meshgrid(xs, xs)
    y = -y
    z = np.full_like(x, f)
    n = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    p, yw = math.radians(pitch_deg), math.radians(yaw_deg)
    y2 = y * math.cos(p) - z * math.sin(p)
    z2 = y * math.sin(p) + z * math.cos(p)
    x3 = x * math.cos(yw) + z2 * math.sin(yw)
    z3 = -x * math.sin(yw) + z2 * math.cos(yw)
    lon = np.arctan2(x3, z3)
    lat = np.arcsin(np.clip(y2, -1, 1))
    u = ((lon + math.pi) / (2 * math.pi) * (w - 1)).astype(np.float32)
    v = ((math.pi / 2 - lat) / math.pi * (h - 1)).astype(np.float32)
    return cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def pov_path(cfg, st_row, d):
    return cfg.povs / st_row["station"] / st_row["date"] / f"{st_row['photo_id']}_{d}.jpg"


def cmd_slice(cfg, args):
    rows = stations(cfg)
    fov, pitch = cfg.pov["fov_deg"], cfg.pov["pitch_deg"]
    dirs = cfg.pov["directions"]
    for k, r in enumerate(rows, 1):
        outs = {d: pov_path(cfg, r, d) for d in dirs}
        if all(p.exists() for p in outs.values()) and not args.force:
            continue
        t = time.time()
        img = imread(r["path"])
        h, w = img.shape[:2]
        # native resolution: a 90-degree view spans a quarter of the panorama width
        size = int(cfg.pov["size"]) if str(cfg.pov["size"]).isdigit() else int(round(w * fov / 360))
        heading = float(r["heading"]) if r["heading"] else 0.0
        for d, compass in dirs.items():
            yaw = ((compass - heading + 180) % 360) - 180   # compass bearing -> image yaw
            imwrite(outs[d], equirect_to_perspective(img, fov, yaw, pitch, size))
        print(f"[{k}/{len(rows)}] {r['station']} {r['date']} {r['photo_id']}  {size}px  {time.time() - t:.1f}s")
    print(f"\nPOVs in {cfg.povs}  (folder per station / date; N E S W = compass direction)")


def all_povs(cfg):
    out = []
    for r in stations(cfg):
        for d in cfg.pov["directions"]:
            p = pov_path(cfg, r, d)
            if p.exists():
                out.append((r, d, p))
    return out


# ======================================================= 4. autolabel (SAM 3)
def merge(dets, thr=0.6):
    from sam3_autolabel import box_iou
    dets = sorted(dets, key=lambda d: -d["score"])
    kept = []
    for d in dets:
        if all(k["cls"] != d["cls"] or box_iou(d["box"], k["box"]) < thr for k in kept):
            kept.append(d)
    return kept


def refine(img, dets, names, cross_thr=0.5):
    """Fix SAM 3's overlapping prompts:
    1. open vs closed boll decided by white-fibre fraction inside the box
    2. one label per object: overlapping boxes of different classes -> keep best score"""
    from sam3_autolabel import box_iou
    arr = np.asarray(img)
    i_open = names.index("open_boll") if "open_boll" in names else None
    i_closed = names.index("closed_boll") if "closed_boll" in names else None
    for d in dets:
        # measure only the central half of the box: a closed boll seen against bright
        # sky has white corners, but its centre is purple; an open boll's centre is fibre
        bx1, by1, bx2, by2 = d["box"]
        cw, ch = (bx2 - bx1) / 4, (by2 - by1) / 4
        x1, y1, x2, y2 = int(bx1 + cw), int(by1 + ch), int(np.ceil(bx2 - cw)), int(np.ceil(by2 - ch))
        patch = arr[max(0, y1):y2, max(0, x1):x2]
        if patch.size == 0:
            d["white"] = 0.0
            continue
        pi = patch.astype(int)
        mx, mn = pi.max(axis=2), pi.min(axis=2)
        # white OR shaded-grey fibre: bright-ish, nearly colourless, and not sky-blue
        fibre = (mn > 125) & (mx - mn < 40) & (pi[..., 2] - pi[..., 0] < 18)
        d["white"] = round(float(fibre.mean()), 3)
        # fibre has texture; sky and the blurred camera-mount patch are smooth
        d["texture"] = round(float(patch.mean(axis=2).std()), 1)
        is_fibre = d["white"] >= 0.2 and d["texture"] >= 14
        if i_open is not None and i_closed is not None:
            if d["cls"] == i_open and not is_fibre:
                d["cls"] = i_closed if d["white"] < 0.05 else -1   # -1 = drop (sky / blur)
            elif d["cls"] == i_closed and is_fibre:
                d["cls"] = i_open
            elif d["cls"] not in (i_open, i_closed) and is_fibre and d["white"] >= 0.35:
                d["cls"] = i_open        # e.g. "flower" box that is really white fibre
    dets = [d for d in dets if d["cls"] >= 0]
    dets = merge(dets)                   # same-class duplicates created by relabelling
    kept = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(box_iou(d["box"], k["box"]) < cross_thr for k in kept):
            kept.append(d)
    if i_open is not None:               # one open boll = its touching fibre pieces combined
        kept = union_touching(kept, i_open)
    return kept


def union_touching(dets, cls, gap=0.15):
    """Merge boxes of one class that overlap or nearly touch (a fluffy open boll is often
    detected as 2-4 separate locks). gap = allowed spacing as a fraction of box size."""
    group = [d for d in dets if d["cls"] == cls]
    rest = [d for d in dets if d["cls"] != cls]
    changed = True
    while changed:
        changed = False
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i]["box"], group[j]["box"]
                g = gap * min(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
                if a[0] - g < b[2] and b[0] - g < a[2] and a[1] - g < b[3] and b[1] - g < a[3]:
                    group[i] = dict(group[i], box=[min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
                                    score=max(group[i]["score"], group[j]["score"]))
                    del group[j]
                    changed = True
                    break
            if changed:
                break
    return rest + group


def tiled_detect(img, tile, fn, names=None):
    """fn(tile_rgb_array) -> list of (box_xyxy, score, cls). Returns merged dets in image coords."""
    W, H = img.size
    dets = []
    for x0, y0, x1, y1 in make_tiles(W, H, tile, 0.2):
        arr = np.ascontiguousarray(np.asarray(img.crop((x0, y0, x1, y1)))[:, :, ::-1])
        for box, sc, c in fn(arr):
            b = [float(box[0] + x0), float(box[1] + y0), float(box[2] + x0), float(box[3] + y0)]
            dets.append(dict(box=b, score=float(sc), cls=int(c)))
    dets = merge(dets)
    return refine(img, dets, names) if names else dets


def draw_dets(img, dets, names, path):
    from PIL import ImageDraw
    cols = [(255, 255, 255), (230, 60, 60), (255, 120, 220), (60, 160, 255)]
    im = img.copy()
    dr = ImageDraw.Draw(im)
    for d in dets:
        c = cols[d["cls"] % len(cols)]
        dr.rectangle(d["box"], outline=c, width=3)
        dr.text((d["box"][0] + 3, d["box"][1] + 2), names[d["cls"]], fill=c)
    im.thumbnail((1400, 1400))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=85)


def cmd_autolabel(cfg, args):
    from sam3_autolabel import build_predictor
    names = cfg.classes
    prompts = [cfg.obj["prompts"][n] for n in names]
    done = set()
    if cfg.labels.exists():
        done = {json.loads(l)["pov"] for l in cfg.labels.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [(r, d, p) for r, d, p in all_povs(cfg) if str(p) not in done]
    if args.station:
        todo = [t for t in todo if t[0]["station"] in args.station]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} POVs done, {len(todo)} to label. Prompts: {prompts}")
    if not todo:
        return
    pred, _ = build_predictor(cfg.sam3, cfg.obj["conf"], args.device)

    def fn(arr):
        pred.set_image(arr)
        res = pred(text=prompts)
        res = res[0] if isinstance(res, (list, tuple)) else res
        if res.boxes is None or len(res.boxes) == 0:
            return []
        return zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                   res.boxes.cls.cpu().numpy().astype(int))

    cfg.labels.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with cfg.labels.open("a", encoding="utf-8") as fh:
        for k, (r, d, p) in enumerate(todo, 1):
            t = time.time()
            img = Image.open(p).convert("RGB")
            raw = tiled_detect(img, int(cfg.obj["tile"]), fn)          # SAM 3 as-is
            dets = refine(img, [dict(x) for x in raw], names)         # cleaned labels
            fh.write(json.dumps(dict(pov=str(p), station=r["station"], date=r["date"], photo_id=r["photo_id"],
                                     direction=d, width=img.width, height=img.height,
                                     dets=dets, raw=raw)) + "\n")
            fh.flush()
            draw_dets(img, dets, names, cfg.obj_previews / f"{r['station']}_{r['photo_id']}_{d}.jpg")
            cnt = {n: sum(x["cls"] == i for x in dets) for i, n in enumerate(names)}
            eta = (time.time() - t0) / k * (len(todo) - k) / 60
            print(f"[{k}/{len(todo)}] {r['station']} {r['date']} {d}  {cnt}  {time.time() - t:.0f}s  ETA {eta:.0f} min")
    print(f"\nLabels: {cfg.labels}\nPreviews: {cfg.obj_previews}")


def cmd_refine(cfg, args):
    """Re-apply the cleaning rules (refine) to the saved raw SAM 3 output - no SAM 3 re-run."""
    if not cfg.labels.exists():
        raise SystemExit("Run 'autolabel' first.")
    names = cfg.classes
    out = []
    for line in cfg.labels.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if "raw" in j:
            img = Image.open(j["pov"]).convert("RGB")
            j["dets"] = refine(img, [dict(x) for x in j["raw"]], names)
            draw_dets(img, j["dets"], names, cfg.obj_previews / f"{j['station']}_{j['photo_id']}_{j['direction']}.jpg")
        out.append(json.dumps(j))
    cfg.labels.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Re-refined {len(out)} POVs -> {cfg.labels}")


def sam3_labels(cfg):
    if not cfg.labels.exists():
        return {}
    return {(j := json.loads(l))["pov"]: j for l in cfg.labels.read_text(encoding="utf-8").splitlines() if l.strip()}


# ============================================================ 5. traindet
def cmd_traindet(cfg, args):
    labs = sam3_labels(cfg)
    if not labs:
        raise SystemExit("Run 'autolabel' first.")
    tile = int(cfg.obj["tile"])
    photos = sorted({j["photo_id"] for j in labs.values()})
    random.Random(0).shuffle(photos)
    val = set(photos[:max(1, round(len(photos) * 0.15))])
    shutil.rmtree(cfg.det_data, ignore_errors=True)
    n = defaultdict(int)
    for j in labs.values():
        split = "val" if j["photo_id"] in val else "train"
        img = Image.open(j["pov"]).convert("RGB")
        for x0, y0, x1, y1 in make_tiles(img.width, img.height, tile, 0.2):
            tw, th = x1 - x0, y1 - y0
            lines = []
            for d in j["dets"]:
                bx1, by1, bx2, by2 = d["box"]
                cx1, cy1, cx2, cy2 = max(bx1, x0), max(by1, y0), min(bx2, x1), min(by2, y1)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                if (cx2 - cx1) * (cy2 - cy1) < 0.5 * (bx2 - bx1) * (by2 - by1):
                    continue  # mostly outside this tile
                lines.append(f"{d['cls']} {((cx1 + cx2) / 2 - x0) / tw:.5f} {((cy1 + cy2) / 2 - y0) / th:.5f} "
                             f"{(cx2 - cx1) / tw:.5f} {(cy2 - cy1) / th:.5f}")
            stem = f"{j['photo_id']}_{j['direction']}_{x0}_{y0}"
            for sub in ("images", "labels"):
                (cfg.det_data / sub / split).mkdir(parents=True, exist_ok=True)
            img.crop((x0, y0, x1, y1)).save(cfg.det_data / "images" / split / f"{stem}.jpg", quality=92)
            (cfg.det_data / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            n[split] += 1
    (cfg.det_data / "data.yaml").write_text(
        f"path: {cfg.det_data.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(cfg.classes)}\nnames: {cfg.classes}\n")
    print(f"Object dataset: {dict(n)} tiles -> {cfg.det_data}")
    from ultralytics import YOLO
    YOLO(args.base).train(data=str(cfg.det_data / "data.yaml"), epochs=args.epochs, imgsz=640,
                          batch=args.batch, device=args.device, workers=2, patience=15,
                          project=str(cfg.det_model.parents[2]), name="objects", exist_ok=True)
    print(f"\nDetector: {cfg.det_model}")


# ============================================================= 6. analyze
def boll_hand_labels(cfg):
    """crop name -> 'open' / 'closed' / 'not_boll' from sort_bolls.py (boll_crops folders)."""
    base = cfg.out / "boll_crops"
    return {f.name: lab for lab in ("open", "closed", "not_boll") for f in (base / lab).glob("*.jpg")}


BOLL_CHECK_CONF = 0.6     # boll model answers below this confidence are counted as "bolls to check"


def boll_model_labels(cfg):
    """crop name -> ('open' / 'closed' / 'not_boll', confidence) from boll_classify_roboflow.py."""
    f = cfg.out / "objects" / "boll_cls_roboflow.csv"
    if not f.exists():
        return {}
    short = {"open_boll": "open", "closed_boll": "closed", "not_boll": "not_boll"}
    return {r["crop"]: (short.get(r["pred"], ""), float(r["conf"])) for r in read_csv(f) if short.get(r["pred"])}


def covered(a, b):
    """Fraction of box a that lies inside box b."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def draw_leaf_health(img, polys, labels, needs_check, row, path):
    """Leaf outlines: green = healthy, red = diseased, yellow = needs checking."""
    from PIL import ImageDraw
    im = img.copy()
    dr = ImageDraw.Draw(im, "RGBA")
    for poly, (lab, p) in zip(polys, labels):
        col = (255, 200, 0) if needs_check(p) else (230, 40, 40) if lab == "diseased" else (40, 200, 60)
        pts = [tuple(map(float, xy)) for xy in poly]
        if len(pts) >= 3:
            dr.polygon(pts, fill=col + (70,), outline=col + (255,), width=3)
    txt = (f"{row['station']} {row['date']} {row['direction']}   leaves {row['leaves']}   "
           f"diseased {row['diseased_leaves']} ({row['pct_diseased']}%)   check {row['check_leaves']}")
    dr.rectangle([0, 0, im.width, 34], fill=(0, 0, 0, 170))
    dr.text((10, 9), txt, fill=(255, 255, 255))
    im.thumbnail((1200, 1200))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=85)


def cmd_analyze(cfg, args):
    import pipeline as P
    names = cfg.classes
    labs = sam3_labels(cfg)
    det = None
    if cfg.det_model.exists() and not args.use_sam3_labels:
        from ultralytics import YOLO
        det = YOLO(str(cfg.det_model))
        print(f"Objects: trained detector {cfg.det_model}")
    elif labs:
        print("Objects: SAM 3 labels (no trained detector yet)")
    else:
        print("Objects: none available (run autolabel) - counts left empty")
    have_leaf = P.LEAF_MODEL.exists() and P.CLS_MODEL.exists()
    if have_leaf:
        from ultralytics import YOLO
        leaf_run = P.make_leaf_runner(False, 0.3, args.device)
        cls_model = YOLO(str(P.CLS_MODEL))
        print(f"Leaves/health: {P.LEAF_MODEL.name} + {P.CLS_MODEL.name}")
    else:
        print("Leaves/health: models not trained yet (pipeline.py trainleaf / traincls) - left empty")

    hand = boll_hand_labels(cfg)
    bmodel = boll_model_labels(cfg)
    if hand or bmodel:
        print(f"Bolls: label = your hand sorting ({len(hand)}) > boll model ({len(bmodel)}) > SAM 3's own label")
    src = defaultdict(int)
    leaf_dir = cfg.results / "leaf_health"
    rows = []
    povs = all_povs(cfg)
    for k, (r, d, p) in enumerate(povs, 1):
        img = Image.open(p).convert("RGB")
        row = dict(station=r["station"], date=r["date"], photo_id=r["photo_id"], direction=d, pov=str(p))
        dets = None
        if det is not None:
            fn = lambda arr: ((b, s, c) for res in det.predict(arr, imgsz=640, conf=0.25, verbose=False, device=args.device)
                              for b, s, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                                 res.boxes.cls.cpu().numpy().astype(int)))
            dets = tiled_detect(img, int(cfg.obj["tile"]), fn, names)
        elif str(p) in labs:
            dets, row["check_bolls"] = [], 0
            for i_, x in enumerate(labs[str(p)]["dets"]):
                name = f"{r['photo_id']}_{d}_{i_:03d}.jpg"
                lab = hand.get(name)
                if lab:
                    src["hand"] += 1
                elif name in bmodel:
                    lab, conf = bmodel[name]
                    src["model"] += 1
                    if conf < BOLL_CHECK_CONF and lab != "not_boll":
                        row["check_bolls"] += 1
                else:
                    src["SAM 3"] += 1
                if lab == "not_boll":
                    continue
                if lab in ("open", "closed"):
                    x = dict(x, cls=names.index(f"{lab}_boll"))
                dets.append(x)
        for i, n in enumerate(names):
            row[n] = sum(x["cls"] == i for x in dets) if dets is not None else ""
        if have_leaf:
            leaves = P.find_leaves(img, leaf_run)
            crops, polys, area, on_boll = [], [], 0.0, 0
            boll_boxes = [x["box"] for x in (dets or [])]
            for box, sc, poly in leaves:
                b = [max(0, int(box[0])), max(0, int(box[1])), min(img.width, int(box[2])), min(img.height, int(box[3]))]
                if (b[2] - b[0]) * (b[3] - b[1]) < 40 * 40:
                    continue
                if any(covered(b, bb) > 0.5 for bb in boll_boxes):
                    on_boll += 1          # red closed bolls get outlined as "leaves": not a leaf
                    continue
                crops.append(P.masked_crop(img, poly, b))
                polys.append(poly)
                xs, ys = poly[:, 0], poly[:, 1]
                area += 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
            labels = P.classify(cls_model, crops) if crops else []
            row["leaves"] = len(labels)
            row["diseased_leaves"] = sum(l == "diseased" for l, _ in labels)
            row["pct_diseased"] = round(100 * row["diseased_leaves"] / len(labels), 2) if labels else 0
            row["check_leaves"] = sum(P.needs_check(p) for _, p in labels)  # unsure: worth a look
            row["canopy_pct"] = round(100 * min(1.0, area / (img.width * img.height)), 2)
            row["dropped_on_boll"] = on_boll
            draw_leaf_health(img, polys, labels, P.needs_check, row,
                             leaf_dir / r["station"] / r["date"] / f"{r['photo_id']}_{d}.jpg")
        rows.append(row)
        if k % 20 == 0 or k == len(povs):
            print(f"  {k}/{len(povs)} POVs")

    if src:
        print(f"Boll labels used: {dict(src)}")
    metrics = names + (["check_bolls"] if bmodel else []) + \
        (["leaves", "diseased_leaves", "check_leaves", "pct_diseased", "canopy_pct"] if have_leaf else [])
    write_csv(cfg.results / "detections.csv", rows, ["station", "date", "photo_id", "direction"] + metrics +
              (["dropped_on_boll"] if have_leaf else []) + ["pov"])

    # growth: per station per date, sum over the 4 directions, mean over repeat photos
    per_photo = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = (row["station"], row["date"], row["photo_id"])
        for m in metrics:
            if row.get(m, "") != "":
                per_photo[key][m] += float(row[m])
    for key, v in per_photo.items():  # percentages: average over directions, not sum
        for m in ("pct_diseased", "canopy_pct"):
            if m in v:
                v[m] /= len(cfg.pov["directions"])
    growth = defaultdict(list)
    for (st, dt, _), v in per_photo.items():
        growth[(st, dt)].append(v)
    grows = []
    for (st, dt), vs in sorted(growth.items()):
        g = dict(station=st, date=dt, photos=len(vs))
        for m in metrics:
            vals = [v[m] for v in vs if m in v]
            g[m] = round(sum(vals) / len(vals), 2) if vals else ""
        grows.append(g)
    write_csv(cfg.results / "growth.csv", grows, ["station", "date", "photos"] + metrics)
    plot_growth(cfg, grows, metrics)
    print(f"\nPer-POV results: {cfg.results / 'detections.csv'}")
    print(f"Growth table   : {cfg.results / 'growth.csv'}")
    print(f"Growth plots   : {cfg.results / 'plots'}")


def plot_growth(cfg, grows, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = cfg.results / "plots"
    out.mkdir(parents=True, exist_ok=True)
    sts = sorted({g["station"] for g in grows})
    for m in metrics:
        if not any(g[m] != "" for g in grows):
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, st in enumerate(sts):
            pts = [(datetime.strptime(g["date"], "%Y-%m-%d"), float(g[m])) for g in grows if g["station"] == st and g[m] != ""]
            if pts:     # 10 colours repeat after P10: later plants get dashed lines + square markers
                ax.plot(*zip(*sorted(pts)), color=f"C{i % 10}", linestyle="-" if i < 10 else "--",
                        marker="o" if i < 10 else "s", linewidth=2, label=st)
        ax.set_title(f"{m.replace('_', ' ')} per station over time")
        ax.set_xlabel("visit date")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out / f"{m}.png", dpi=120)
        plt.close(fig)


# ============================================================== report
CLASS_COLOURS = {"open_boll": (255, 255, 255), "closed_boll": (235, 50, 50), "flower": (255, 110, 220)}
LABELS = {"open_boll": "open bolls", "closed_boll": "closed bolls", "flower": "flowers"}


def _font(size):
    from PIL import ImageFont
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate_pov(img, dets, names, title, counts):
    """Full-size POV with coloured boxes, a title bar and a count legend."""
    from PIL import ImageDraw
    im = img.copy()
    dr = ImageDraw.Draw(im)
    for d in dets:
        n = names[d["cls"]]
        dr.rectangle(d["box"], outline=CLASS_COLOURS.get(n, (60, 160, 255)), width=3)
    W = im.width
    bar = 54 + 36 * len(names)
    dr.rectangle((0, 0, W, bar), fill=(0, 0, 0))
    dr.text((14, 10), title, fill=(255, 255, 255), font=_font(30))
    for i, n in enumerate(names):
        y = 56 + 36 * i
        dr.rectangle((16, y, 44, y + 24), outline=CLASS_COLOURS.get(n), width=4)
        v = counts.get(n, "")
        dr.text((58, y - 2), f"{LABELS.get(n, n)}: {v if v != '' else '-'}", fill=(255, 255, 255), font=_font(26))
    return im


def _num(v):
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:.1f}"
    except (TypeError, ValueError):
        return v


def cmd_report(cfg, args):
    names = cfg.classes
    labs = sam3_labels(cfg)
    det_csv = cfg.results / "detections.csv"
    if not det_csv.exists():
        raise SystemExit("Run 'analyze' first.")
    rows = read_csv(det_csv)
    out_pov = cfg.results / "povs"
    out_sheet = cfg.results / "station_sheets"
    shutil.rmtree(out_pov, ignore_errors=True)
    dirs = list(cfg.pov["directions"])
    by_station = defaultdict(lambda: defaultdict(dict))  # station -> date -> dir -> (thumb, counts)
    for k, r in enumerate(rows, 1):
        p = Path(r["pov"])
        img = Image.open(p).convert("RGB")
        dets = labs.get(str(p), {}).get("dets", [])
        counts = {n: r.get(n, "") for n in names}
        title = f"{r['station']}   {r['date']}   facing {r['direction']}"
        im = annotate_pov(img, dets, names, title, counts)
        dst = out_pov / r["station"] / r["date"] / f"{r['photo_id']}_{r['direction']}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=88)
        if r["direction"] not in by_station[r["station"]][r["date"]]:  # first photo of the day
            t = im.copy()
            t.thumbnail((480, 480))
            by_station[r["station"]][r["date"]][r["direction"]] = t
        if k % 40 == 0:
            print(f"  {k}/{len(rows)} POVs")

    # one sheet per station: rows = visit dates, columns = N E S W, plus a totals column
    from PIL import ImageDraw
    out_sheet.mkdir(parents=True, exist_ok=True)
    growth = {(g["station"], g["date"]): g for g in read_csv(cfg.results / "growth.csv")}
    S, L, T = 480, 170, 50
    for st, dates in sorted(by_station.items()):
        ds = sorted(dates)
        sheet = Image.new("RGB", (L + S * len(dirs), T + S * len(ds)), (25, 25, 25))
        dr = ImageDraw.Draw(sheet)
        dr.text((10, 10), f"Station {st}", fill=(255, 255, 255), font=_font(30))
        for j, d in enumerate(dirs):
            dr.text((L + j * S + S // 2 - 10, 10), d, fill=(255, 220, 0), font=_font(30))
        for i, dt in enumerate(ds):
            y = T + i * S
            g = growth.get((st, dt), {})
            dr.text((10, y + 10), dt, fill=(255, 255, 255), font=_font(24))
            for m, n in enumerate(names):
                dr.text((10, y + 50 + 32 * m), f"{n.split('_')[0]}: {_num(g.get(n, '-'))}",
                        fill=CLASS_COLOURS.get(n), font=_font(20))
            for j, d in enumerate(dirs):
                t = dates[dt].get(d)
                if t:
                    sheet.paste(t, (L + j * S, y))
        sheet.save(out_sheet / f"{st}.jpg", quality=88)
    print(f"\nAnnotated POVs : {out_pov}  (station \\ date \\ photo_direction.jpg)")
    print(f"Station sheets : {out_sheet}  (one image per station: dates x N E S W)")


# ========================================================== 7. streetview
def bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def scene_bolls(cfg, r, labs, hand, bmodel):
    """Every open / closed boll of one panorama as a viewer position (yaw, pitch in radians, 0 = camera forward).
    Same labels as 'analyze': your hand label > boll model (R1) > SAM 3's own label; 'not a boll' is dropped.
    Inverse of equirect_to_perspective: a POV pixel -> longitude/latitude in the panorama."""
    names, size = cfg.classes, int(cfg.pov["size"])
    f = (size / 2) / math.tan(math.radians(cfg.pov["fov_deg"]) / 2)
    heading = float(r["heading"]) if r["heading"] else 0.0
    out = []
    for d, compass in cfg.pov["directions"].items():
        j = labs.get(str(pov_path(cfg, r, d)))
        if not j:
            continue
        view_yaw = math.radians(((compass - heading + 180) % 360) - 180)
        for i_, x in enumerate(j["dets"]):
            name = f"{r['photo_id']}_{d}_{i_:03d}.jpg"
            lab = hand.get(name) or (bmodel[name][0] if name in bmodel else None)
            if lab == "not_boll":
                continue
            cls = names.index(f"{lab}_boll") if lab in ("open", "closed") else x["cls"]
            if names[cls] not in ("open_boll", "closed_boll"):
                continue
            b = x["box"]
            cx, cy = (b[0] + b[2]) / 2 - size / 2, (b[1] + b[3]) / 2 - size / 2      # right / down from view centre
            lon = view_yaw + math.atan2(cx, f)
            lat = math.atan2(-cy, math.hypot(cx, f))
            out.append(dict(yaw=round((lon + math.pi) % (2 * math.pi) - math.pi, 4), pitch=round(-lat, 4),
                            open=names[cls] == "open_boll", dir=d,
                            src="hand" if name in hand else ("model" if name in bmodel else "sam3")))
    return out


def cmd_streetview(cfg, args):
    rows = [r for r in stations(cfg) if r["lat"]]
    labs, hand, bmodel = sam3_labels(cfg), boll_hand_labels(cfg), boll_model_labels(cfg)
    growth = {}
    if (cfg.results / "growth.csv").exists():
        growth = {(g["station"], g["date"]): g for g in read_csv(cfg.results / "growth.csv")}
    by_date = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)
    if args.date:
        by_date = {d: v for d, v in by_date.items() if d in args.date}
    cfg.tours.mkdir(parents=True, exist_ok=True)
    width, q = int(cfg.sv["image_width"]), int(cfg.sv["jpeg_quality"])

    for date, rs in sorted(by_date.items()):
        # one scene per station on that date (first photo if repeated)
        seen, scenes_src = set(), []
        for r in sorted(rs, key=lambda r: r["time"]):
            if r["station"] not in seen:
                seen.add(r["station"])
                scenes_src.append(r)
        scenes = []
        for r in scenes_src:
            im = Image.open(r["path"]).convert("RGB")
            im = im.resize((width, width // 2), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=q)
            g = growth.get((r["station"], r["date"]), {})
            parts = [f"Station {r['station']}", f"{r['date']} {r['time']}"]
            for k_, lab in [("open_boll", "open bolls"), ("closed_boll", "closed bolls"), ("flower", "flowers"),
                            ("leaves", "leaves"), ("pct_diseased", "% diseased leaves")]:
                if g.get(k_, "") != "":
                    parts.append(f"{lab}: {g[k_]}")
            scenes.append(dict(
                id=f"{r['station']}_{r['photo_id']}", name=f"{r['station']} - {r['date']}",
                fileName=Path(r["path"]).name,
                imageUrl="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                coordinates=dict(lat=float(r["lat"]), lng=float(r["lon"])),
                heading=float(r["heading"] or 0), description=" | ".join(parts),
                station=r["station"], date=r["date"],
                analysis={k_: g[k_] for k_ in g if k_ not in ("station", "date")},
                bolls=scene_bolls(cfg, r, labs, hand, bmodel),
                links=[]))
        # links to nearest neighbours; yaw = compass bearing relative to the camera heading
        for a in scenes:
            la, lo = a["coordinates"]["lat"], a["coordinates"]["lng"]
            cand = []
            for b in scenes:
                if b is a:
                    continue
                lb, lob = b["coordinates"]["lat"], b["coordinates"]["lng"]
                dx, dy = local_xy(lb, lob, la, lo)
                dist = math.hypot(dx, dy)
                if dist <= float(cfg.sv["link_max_distance_m"]):
                    cand.append((dist, b, bearing(la, lo, lb, lob)))
            for dist, b, brg in sorted(cand, key=lambda c: c[0])[: int(cfg.sv["links_per_scene"])]:
                yaw = ((brg - a["heading"] + 180) % 360) - 180
                a["links"].append(dict(targetId=b["id"], yaw=round(math.radians(yaw), 4), pitch=-0.12,
                                       distance_m=round(dist, 1)))
        tour = dict(name=f"AgriEYE 360 field tour {date}", createdAt=datetime.now().isoformat(timespec="seconds"),
                    scenes=scenes)
        out = cfg.tours / f"tour_{date}.json"
        out.write_text(json.dumps(tour), encoding="utf-8")
        print(f"{out}  {len(scenes)} scenes, {sum(len(s['links']) for s in scenes)} links, "
              f"{out.stat().st_size / 1e6:.1f} MB")
    print("\nOpen agrieye-field-viewer.html in a browser and choose one of these JSON files.")


# ================================================================== main
def cmd_all(cfg, args):
    import copy
    keep = copy.copy(args)
    keep.force = False  # never overwrite the user's station edits in 'all'
    for fn in (cmd_import, cmd_stations, cmd_slice, cmd_autolabel, cmd_analyze, cmd_report, cmd_streetview):
        print(f"\n===== {fn.__name__[4:]} =====")
        fn(cfg, keep if fn is cmd_stations else args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "agrieye_config.json"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("import").set_defaults(fn=cmd_import)
    p = sub.add_parser("stations")
    p.add_argument("--force", action="store_true", help="regenerate even if you edited stations.csv")
    p.set_defaults(fn=cmd_stations)
    p = sub.add_parser("slice")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_slice)
    p = sub.add_parser("autolabel")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--station", nargs="*", help="only these stations, e.g. P02 P06")
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_autolabel)
    sub.add_parser("refine", help="re-apply cleaning rules to saved SAM 3 output").set_defaults(fn=cmd_refine)
    p = sub.add_parser("traindet")
    p.add_argument("--base", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_traindet)
    p = sub.add_parser("analyze")
    p.add_argument("--device", default="cpu")
    p.add_argument("--use-sam3-labels", action="store_true", help="ignore the trained detector")
    p.set_defaults(fn=cmd_analyze)
    sub.add_parser("report", help="annotated POVs + one summary sheet per station").set_defaults(fn=cmd_report)
    p = sub.add_parser("streetview")
    p.add_argument("--date", nargs="*", help="only these dates, e.g. 2026-08-20")
    p.set_defaults(fn=cmd_streetview)
    p = sub.add_parser("all")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--station", nargs="*")
    p.add_argument("--device", default="cpu")
    p.add_argument("--use-sam3-labels", action="store_true")
    p.add_argument("--date", nargs="*")
    p.set_defaults(fn=cmd_all)
    args = ap.parse_args()
    args.fn(Cfg(args.config), args)


if __name__ == "__main__":
    main()
