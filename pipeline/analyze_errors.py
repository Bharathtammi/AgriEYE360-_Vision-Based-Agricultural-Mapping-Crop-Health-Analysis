"""
WHAT EXACTLY GOES WRONG: sort every mistake of one experiment into types, with picture sheets to look at.

  python analyze_errors.py visit_20260804 tile          # dataset folder (in agrieye_output/boll_detector), run name
  python analyze_errors.py data_photo_split tile80

Uses the saved test predictions (runs/<name>/pred_test.json) - no training, no new predictions.
Each hand-labelled boll (>= 24 px) is one of:
  found, right class | found, WRONG class (open <-> closed) | found only below the cut-off (low confidence) |
  box in the wrong place (overlap 0.1-0.5) | not found at all
Each prediction above the cut-off that is not a boll is one of:
  duplicate (second box on a found boll) | badly placed box on a boll | on nothing labelled (background / missing label?)
Output: <run>/errors.txt + errors_missed.jpg, errors_false.jpg, errors_confused.jpg (crops: yellow = label, cyan/pink = model)
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import eval_roboflow_detect as E

ROOT = Path(__file__).resolve().parent
DS = ROOT / "agrieye_output" / "boll_detector" / sys.argv[1]
RUN = DS / "runs" / sys.argv[2]
CUT, MIN_PX = 0.25, 24
NAMES = ["open", "closed"]


def side(b):
    return max(b[2] - b[0], b[3] - b[1])


def size_bin(b):
    s = side(b)
    return "24-40 px" if s < 40 else "40-64 px" if s < 64 else "64+ px"


def main():
    preds = {k: [(c, s, tuple(b)) for c, s, b in v] for k, v in json.loads((RUN / "pred_test.json").read_text()).items()}
    imgs = {f.stem: f for f in (DS / "test" / "images").glob("*.jpg")}
    gts = {k: E.ground_truth(DS / "test" / "labels" / f"{k}.txt", 1440, 1440) for k in imgs}
    gt_kind, fp_kind = Counter(), Counter()
    by_size, by_cls, by_day = Counter(), Counter(), Counter()
    sheets = {"missed": [], "false": [], "confused": []}
    low_conf = []
    for k, g in gts.items():
        p_all = sorted(preds.get(k, []), key=lambda p: -p[1])
        p_hi = [p for p in p_all if p[1] >= CUT]
        used = set()
        day = k.split("_")[1] if k.startswith("P") else k.split("_")[1][:8]
        for gc, gb in g:
            if side(gb) < MIN_PX:
                continue
            # best confident prediction (any class) not used yet
            best, bi = 0, -1
            for i, (pc, ps, pb) in enumerate(p_hi):
                if i not in used:
                    v = E.iou(gb, pb)
                    if v > best:
                        best, bi = v, i
            if best >= 0.5:
                used.add(bi)
                kind = "found, right class" if p_hi[bi][0] == gc else "found, WRONG class"
                if kind.endswith("WRONG class"):
                    sheets["confused"].append((k, gb, gc, p_hi[bi]))
            elif best >= 0.1:
                kind = "box in the wrong place"; used.add(bi)
                sheets["missed"].append((k, gb, gc, p_hi[bi]))
            else:
                lo = max((ps for pc, ps, pb in p_all if ps < CUT and E.iou(gb, pb) >= 0.5), default=0)
                kind = "found only below the cut-off" if lo >= 0.05 else "not found at all"
                if lo >= 0.05:
                    low_conf.append(lo)
                sheets["missed"].append((k, gb, gc, None))
            gt_kind[kind] += 1
            ok = kind.startswith("found, ")
            by_size[(size_bin(gb), ok)] += 1; by_cls[(NAMES[gc], kind)] += 1; by_day[(day, ok)] += 1
        for i, (pc, ps, pb) in enumerate(p_hi):
            if i in used or side(pb) < MIN_PX:
                continue
            ov = max((E.iou(pb, gb) for _, gb in g), default=0)
            dup = any(E.iou(pb, p_hi[j][2]) >= 0.5 for j in used)
            kind = "duplicate box" if dup else "badly placed box on a boll" if ov >= 0.1 else "on nothing labelled"
            fp_kind[kind] += 1
            if kind == "on nothing labelled":
                sheets["false"].append((k, None, None, (pc, ps, pb)))
    n = sum(gt_kind.values())
    L = [f"Error analysis: {RUN.relative_to(ROOT)} on {len(imgs)} test pictures, bolls >= {MIN_PX} px, "
         f"confidence cut-off {CUT}", "", f"Hand-labelled bolls: {n}"]
    for kind in ("found, right class", "found, WRONG class", "found only below the cut-off", "box in the wrong place",
                 "not found at all"):
        L.append(f"  {kind:<32}{gt_kind[kind]:>6}  {100 * gt_kind[kind] / max(n, 1):5.1f}%")
    m = sum(fp_kind.values())
    L += ["", f"Model boxes that are not a (new) boll: {m}"]
    for kind in ("on nothing labelled", "duplicate box", "badly placed box on a boll"):
        L.append(f"  {kind:<32}{fp_kind[kind]:>6}")
    L += ["", "Found (any class) by size:"]
    for s in ("24-40 px", "40-64 px", "64+ px"):
        t = by_size[(s, True)] + by_size[(s, False)]
        L.append(f"  {s:<10}{by_size[(s, True)]:>5} / {t:<5} {100 * by_size[(s, True)] / max(t, 1):5.1f}%")
    L += ["", "By class (what happened to each labelled boll):"]
    for c in NAMES:
        t = sum(v for (cc, _), v in by_cls.items() if cc == c)
        parts = ", ".join(f"{kk} {v}" for (cc, kk), v in sorted(by_cls.items()) if cc == c)
        L.append(f"  {c:<7}{t:>5}: {parts}")
    L += ["", "Found (any class) by visit:"]
    for d in sorted({d for d, _ in by_day}):
        t = by_day[(d, True)] + by_day[(d, False)]
        L.append(f"  {d}  {by_day[(d, True)]:>4} / {t:<4} {100 * by_day[(d, True)] / max(t, 1):5.1f}%")
    if low_conf:
        L += ["", f"Bolls found only below {CUT}: median confidence {np.median(low_conf):.2f}"]
    # picture sheets
    random.seed(0)
    for name, items in sheets.items():
        items = random.sample(items, min(60, len(items)))
        if not items:
            continue
        cols = 10; out = Image.new("RGB", (cols * 132, ((len(items) + cols - 1) // cols) * 132), (25, 25, 25))
        for i, (k, gb, gc, p) in enumerate(items):
            im = Image.open(imgs[k]).convert("RGB")
            ref = gb or p[2]; cx, cy = (ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2; s = max(side(ref) * 1.6, 48)
            cr = im.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s))).resize((128, 128)); kk = 128 / (2 * s)
            d = ImageDraw.Draw(cr)
            box = lambda b: ((b[0] - cx + s) * kk, (b[1] - cy + s) * kk, (b[2] - cx + s) * kk, (b[3] - cy + s) * kk)
            if gb:
                d.rectangle(box(gb), outline=(255, 230, 0), width=2)
            if p:
                d.rectangle(box(p[2]), outline=(47, 208, 255) if p[0] == 1 else (255, 79, 216), width=2)
            txt = (f"label {NAMES[gc]}" if gb else "") + (f" model {NAMES[p[0]]} {p[1]:.2f}" if p else "")
            d.text((2, 2), txt.strip(), fill=(255, 255, 255))
            out.paste(cr, (i % cols * 132, i // cols * 132))
        out.save(RUN / f"errors_{name}.jpg", quality=88)
    L += ["", f"Sheets: {RUN / 'errors_missed.jpg'}, errors_false.jpg, errors_confused.jpg",
          "  yellow = hand label, cyan = model closed, pink = model open"]
    (RUN / "errors.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
