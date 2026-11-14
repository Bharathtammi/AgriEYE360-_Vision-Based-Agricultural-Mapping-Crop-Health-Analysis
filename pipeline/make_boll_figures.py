"""
Clear open / closed boll figures (for the thesis): every boll box drawn so the two classes cannot be confused.

  OPEN   = cyan,    SOLID thick box,  tag "OPEN"
  CLOSED = magenta, DASHED box,       tag "CLOSED"
  every box has a black outline so it stands out on sky, cotton and leaves; legend + counts in a top bar.

  python make_boll_figures.py                 # the 4 views with the most open bolls (20 Aug) + a 2x2 sheet
  python make_boll_figures.py --all           # every view (140)
  python make_boll_figures.py --plant P13     # one plant: all visits side by side (bolls opening over time)

Labels: roboflow_export/1_cotton_boll_detection (your hand labels, then the Roboflow boll model).
Output: sample_results/boll_figures/
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "roboflow_export" / "1_cotton_boll_detection"
OUT = ROOT / "sample_results" / "boll_figures"
OPEN_C, CLOSED_C = (0, 230, 255), (255, 0, 200)
F = r"C:\Windows\Fonts\arialbd.ttf"


def labels(lab):
    W = H = 1440
    for l in lab.read_text().splitlines():
        if l.strip():
            c, x, y, w, h = l.split()
            x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
            yield c == "0", (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def dashed_rect(d, b, col, width, dash=14, gap=9):
    x1, y1, x2, y2 = b
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        L = max(abs(bx - ax), abs(by - ay)); n = int(L // (dash + gap)) + 1
        for k in range(n):
            s, e = k * (dash + gap) / max(L, 1), min(1, (k * (dash + gap) + dash) / max(L, 1))
            d.line([(ax + (bx - ax) * s, ay + (by - ay) * s), (ax + (bx - ax) * e, ay + (by - ay) * e)], fill=col, width=width)


def draw(lab):
    img = Image.open(lab.parent.parent / "images" / f"{lab.stem}.jpg").convert("RGB")
    d = ImageDraw.Draw(img)
    tag = ImageFont.truetype(F, 20)
    no = nc = 0
    boxes = sorted(labels(lab), key=lambda t: -(t[1][2] - t[1][0]) * (t[1][3] - t[1][1]))   # big first, small on top
    for is_open, b in boxes:
        col = OPEN_C if is_open else CLOSED_C
        g = [b[0] - 3, b[1] - 3, b[2] + 3, b[3] + 3]
        if is_open:
            d.rectangle(g, outline=(0, 0, 0), width=9)                 # black halo
            d.rectangle(b, outline=col, width=5)
            no += 1
        else:
            dashed_rect(d, g, (0, 0, 0), 8)
            dashed_rect(d, b, col, 4)
            nc += 1
        if b[2] - b[0] >= 45:                                          # tag on boxes big enough to carry one
            t = "OPEN" if is_open else "CLOSED"
            tw = d.textlength(t, font=tag)
            ty = b[1] - 26 if b[1] > 90 else b[3] + 2
            d.rectangle([b[0] - 3, ty, b[0] + tw + 8, ty + 24], fill=col, outline=(0, 0, 0), width=2)
            d.text((b[0] + 2, ty + 1), t, font=tag, fill=(0, 0, 0))
    # top bar: name, legend, counts
    bar = ImageFont.truetype(F, 30)
    d.rectangle([0, 0, img.width, 58], fill=(0, 0, 0))
    name = lab.stem.split("_")
    d.text((14, 13), f"{name[0]}  {name[1]}  {name[-1]} view", font=bar, fill=(255, 255, 255))
    x = 520
    d.rectangle([x, 14, x + 44, 44], outline=OPEN_C, width=5)
    d.text((x + 56, 13), f"OPEN  {no}", font=bar, fill=OPEN_C)
    x = 860
    dashed_rect(d, (x, 14, x + 44, 44), CLOSED_C, 4, dash=8, gap=5)
    d.text((x + 56, 13), f"CLOSED  {nc}", font=bar, fill=CLOSED_C)
    return img, no, nc


def sheet(imgs, cols, path, T=720):
    rows = (len(imgs) + cols - 1) // cols
    s = Image.new("RGB", (cols * T, rows * T), (0, 0, 0))
    for i, im in enumerate(imgs):
        im = im.copy(); im.thumbnail((T, T)); s.paste(im, ((i % cols) * T, (i // cols) * T))
    s.save(path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--plant")
    ap.add_argument("--pov4", action="store_true", help="one sheet per photo: its 4 views N E S W, all labels drawn")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    labs = sorted(DATA.glob("*/labels/*.txt"))
    if a.pov4:
        order = {"N": 0, "E": 1, "S": 2, "W": 3}
        photos = {}
        for l in labs:
            photos.setdefault(l.stem.rsplit("_", 1)[0], []).append(l)
        dst = OUT / "pov4"
        dst.mkdir(parents=True, exist_ok=True)
        tot_o = tot_c = 0
        for key, ls in sorted(photos.items()):
            ls.sort(key=lambda l: order.get(l.stem[-1], 9))
            drawn = [draw(l) for l in ls]
            no, nc = sum(d[1] for d in drawn), sum(d[2] for d in drawn)
            tot_o += no; tot_c += nc
            sheet([d[0] for d in drawn], 2, dst / f"{key}_4pov.jpg", T=900)
            print(f"{key}: open {no}, closed {nc}  ({ls[0].parent.parent.name})")
        print(f"\n{len(photos)} photos, open {tot_o}, closed {tot_c}\nFiles: {dst}  (one sheet per photo: N E / S W)")
        return
    if a.plant:
        labs = [l for l in labs if l.stem.startswith(a.plant + "_")]
        # one row per visit, the 4 views N E S W
        order = {"N": 0, "E": 1, "S": 2, "W": 3}
        labs.sort(key=lambda l: (l.stem.split("_")[1], order.get(l.stem[-1], 9)))
        imgs = [draw(l)[0] for l in labs]
        p = OUT / f"{a.plant}_all_visits.jpg"
        sheet(imgs, 4, p, T=600)
        print(f"{p}  ({len(imgs)} views; rows = visits, columns = N E S W)")
        return
    if not a.all:
        cnt = [(sum(o for o, _ in labels(l)), l) for l in labs]
        labs = [l for _, l in sorted(cnt, key=lambda t: -t[0])[:4]]
    imgs = []
    for l in labs:
        img, no, nc = draw(l)
        img.save(OUT / f"{l.stem}_boxes.jpg", quality=92)
        imgs.append(img)
        print(f"{l.stem}: open {no}, closed {nc}")
    if not a.all:
        sheet(imgs, 2, OUT / "00_open_vs_closed.jpg")
    print(f"\nFiles: {OUT}")


if __name__ == "__main__":
    main()
