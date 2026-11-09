"""
Visual check of plant IDs (stations) in agrieye_output/stations.csv.

Makes two images in agrieye_output/station_check/:
  stations_sheet.jpg  one row per station, one panorama thumbnail per photo (date order).
                      Every panorama is rotated so NORTH is in the middle, so the same spot
                      should look alike across visits. The bottom strip shows the ground/plant
                      under the camera.
  stations_map.png    GPS position of every photo, coloured by station.
and prints warnings (same-day photos at one station, near-duplicate shots, far-apart photos).

Fix IDs by editing the 'station' column of stations.csv (empty = skip that photo), then re-run.

  python check_stations.py
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "agrieye_output"
CSV = OUT / "stations.csv"
DST = OUT / "station_check"
TW, TH = 520, 260          # thumbnail size (2:1 equirect)


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def thumb(path, heading):
    img = imread(path)
    if img is None:
        t = np.full((TH, TW, 3), 60, np.uint8)
        cv2.putText(t, "missing", (10, TH // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return t
    img = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
    shift = int(round(float(heading or 0) / 360 * TW))
    img = np.roll(img, shift, axis=1)            # north to the centre column
    for frac, lab in ((0.25, "W"), (0.5, "N"), (0.75, "E")):
        x = int(frac * TW)
        cv2.line(img, (x, 0), (x, 14), (255, 255, 255), 2)
        cv2.putText(img, lab, (x + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img


def label(img, text, sub):
    bar = np.full((44, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(bar, sub, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    return np.vstack([bar, img])


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    DST.mkdir(parents=True, exist_ok=True)
    by_st = defaultdict(list)
    for r in rows:
        by_st[r["station"].strip() or "(none)"].append(r)

    # ---------- warnings ----------
    print(f"{len(rows)} photos, {len(by_st)} stations\n")
    for st in sorted(by_st):
        rs = sorted(by_st[st], key=lambda r: (r["date"], r["time"]))
        dates = [r["date"][5:] for r in rs]
        print(f"{st}: {len(rs)} photos  visits {', '.join(dates)}")
        pts = [(float(r["x_m"]), float(r["y_m"])) for r in rs if r["x_m"]]
        if len(pts) > 1:
            spread = max(math.dist(a, b) for a in pts for b in pts)
            if spread > 2.0:
                print(f"   ! photos up to {spread:.1f} m apart - maybe two plants")
        same = defaultdict(list)
        for r in rs:
            same[r["date"]].append(r)
        for d, g in same.items():
            if len(g) > 1:
                gap = [f"{a['time']}->{b['time']}" for a, b in zip(g, g[1:])]
                dup = all(a["x_m"] == b["x_m"] and a["y_m"] == b["y_m"] for a, b in zip(g, g[1:]))
                kind = "same GPS = likely repeat shot" if dup else "different GPS = maybe 2 plants"
                print(f"   ! {len(g)} photos on {d} ({', '.join(gap)}) - {kind}")

    # ---------- picture sheet ----------
    max_n = max(len(v) for v in by_st.values())
    sheet_rows = []
    for st in sorted(by_st):
        rs = sorted(by_st[st], key=lambda r: (r["date"], r["time"]))
        cells = [label(thumb(r["path"], r["heading"]), f"{st}  {r['date'][5:]} {r['time'][:5]}",
                       f"{r['photo_id']}  x={r['x_m']} y={r['y_m']}") for r in rs]
        blank = np.full_like(cells[0], 235)
        cells += [blank] * (max_n - len(cells))
        row = np.hstack([np.pad(c, ((4, 4), (4, 4), (0, 0)), constant_values=235) for c in cells])
        head = np.full((row.shape[0], 90, 3), 40, np.uint8)
        cv2.putText(head, st, (8, row.shape[0] // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        sheet_rows.append(np.hstack([head, row]))
    sheet = np.vstack(sheet_rows)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    buf.tofile(str(DST / "stations_sheet.jpg"))

    # ---------- map ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    cmap = plt.get_cmap("tab20")
    for k, st in enumerate(sorted(by_st)):
        rs = [r for r in by_st[st] if r["x_m"]]
        xs = [float(r["x_m"]) for r in rs]; ys = [float(r["y_m"]) for r in rs]
        ax.scatter(xs, ys, s=70, color=cmap(k % 20), edgecolor="white", linewidth=1.5, zorder=3, label=st)
        for r, x, y in zip(rs, xs, ys):
            ax.annotate(r["date"][5:], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color="#555")
        if xs:
            ax.text(np.mean(xs), np.mean(ys) - 0.35, st, ha="center", fontsize=10, weight="bold")
    ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)"); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.set_title("Photo positions by station (GPS is only ~2-3 m accurate)")
    ax.legend(fontsize=7, ncol=2, loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(DST / "stations_map.png"); plt.close(fig)
    print(f"\nWrote:\n  {DST / 'stations_sheet.jpg'}\n  {DST / 'stations_map.png'}")


if __name__ == "__main__":
    main()
