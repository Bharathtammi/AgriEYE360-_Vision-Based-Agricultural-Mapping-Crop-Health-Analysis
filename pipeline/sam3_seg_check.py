"""
How reliable is SAM 3's leaf segmentation? You judge every SAM 3 outline on a few random views.

  python sam3_seg_check.py              # check views (10 by default, spread over the visit dates)
  python sam3_seg_check.py --recall     # recall round: only mark leaves WITHOUT an outline (own answers file)
  python sam3_seg_check.py --report     # precision / recall from your answers

Every outline starts GREEN (= good leaf). You only fix the others:
  Left click on an outline   = good -> partial -> wrong -> good ...
        good    (green)  : one real leaf, outline fits well
        partial (yellow) : a real leaf but the outline is clearly off (half missing, two leaves merged, spills over)
        wrong   (red)    : not a leaf (boll, stem, sky, soil, flower, blur)
  Right click on a LEAF WITH NO OUTLINE = mark a missed leaf (blue dot); right click the dot again to remove it
      (count clearly visible leaves only - not tiny far-away specks)
  Enter = save this view, next      B = back to previous view      Esc = quit (finished views are saved)

Outlines come from agrieye_output/sam3_yolo_disease (SAM 3, prompt "leaf", conf 0.3 - the same outlines as
agrieye_output/sam3_segmentation). Answers: sample_results/sam3_seg_check/answers.json (resumable).
Report: sample_results/sam3_seg_check/report.txt
"""
import argparse
import csv
import json
import random
import tkinter as tk
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "agrieye_output" / "sam3_yolo_disease"
POVS = ROOT / "agrieye_output" / "povs"
OUT = ROOT / "sample_results" / "sam3_seg_check"
ANS = OUT / "answers.json"
ANS_R = OUT / "answers_recall.json"     # recall round (labels start from answers.json)
SHOW = 900                              # set to fit the screen in main
RECALL = False
COL = {"good": (40, 220, 60), "partial": (255, 200, 0), "wrong": (240, 40, 40)}
CYCLE = ["good", "partial", "wrong"]


def pick_views(n, seed=0):
    allv = sorted(SRC.glob("*/*/*_polygons.json"))
    by_date = defaultdict(list)
    for p in allv:
        by_date[p.parent.name].append(p)
    rng = random.Random(seed)
    for v in by_date.values():
        rng.shuffle(v)
    out, dates = [], sorted(by_date)
    while len(out) < n and any(by_date.values()):          # round-robin over dates
        for d in dates:
            if by_date[d] and len(out) < n:
                out.append(by_date[d].pop())
    return out


def inside(pt, poly):
    x, y = pt; c = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1:
            c = not c
    return c


def area(poly):
    return 0.5 * abs(sum(poly[i][0] * poly[i - 1][1] - poly[i - 1][0] * poly[i][1] for i in range(len(poly))))


class Checker:
    def __init__(self, root, views):
        self.root, self.views = root, views
        self.file = ANS_R if RECALL else ANS
        self.ans = json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {}
        self.base = json.loads(ANS.read_text(encoding="utf-8")) if RECALL and ANS.exists() else {}
        # recall round always starts at view 1 and shows what was saved (so marks can be added to every view)
        self.i = 0 if RECALL else next((k for k, v in enumerate(views) if self.key(v) not in self.ans), len(views))
        root.title("SAM 3 leaf check - RECALL ROUND" if RECALL else "SAM 3 leaf segmentation check"); root.configure(bg="#1e1e1e")
        self.head = tk.Label(root, bg="#1e1e1e", fg="white", font=("Segoe UI", 11), justify="left", anchor="w")
        self.head.pack(fill="x", padx=8, pady=4)
        self.cv = tk.Canvas(root, width=SHOW, height=SHOW, bg="black", highlightthickness=0); self.cv.pack()
        self.cv.bind("<Button-1>", self.left); self.cv.bind("<Button-3>", self.right)
        self.cv.bind("<Control-Button-1>", self.right); self.cv.bind("<Shift-Button-1>", self.right)   # touchpad: Ctrl/Shift + click = missed
        root.bind("<Return>", self.save_next); root.bind("<KP_Enter>", self.save_next)
        root.bind("b", self.back); root.bind("B", self.back); root.bind("<Escape>", lambda e: root.destroy())
        self.load()

    def key(self, p):
        return str(p.relative_to(SRC))

    def load(self):
        if self.i >= len(self.views):
            self.head.configure(text=f"All {len(self.views)} views done. Close the window and run with --report.")
            self.cv.delete("all"); return
        p = self.views[self.i]
        stem = p.name.replace("_polygons.json", "")
        self.img = Image.open(POVS / p.parent.parent.name / p.parent.name / f"{stem}.jpg").convert("RGB")
        self.k = SHOW / max(self.img.size)
        self.polys = [o["polygon"] for o in json.loads(p.read_text(encoding="utf-8")) if o.get("prompt", "leaf") == "leaf"]
        prev = self.ans.get(self.key(p))
        base = self.base.get(self.key(p))
        self.lab = prev["labels"] if prev else (list(base["labels"]) if base else ["good"] * len(self.polys))
        self.missed = prev["missed"] if prev else []
        self.draw()

    def draw(self):
        im = self.img.copy(); d = ImageDraw.Draw(im, "RGBA")
        for poly, lab in zip(self.polys, self.lab):
            c = COL[lab]
            d.polygon([tuple(q) for q in poly], fill=c + (120 if RECALL else 45,), outline=c + (255,), width=3 if lab == "good" else 6)
        for x, y in self.missed:
            d.ellipse([x - 14, y - 14, x + 14, y + 14], fill=(40, 120, 255, 230), outline=(255, 255, 255, 255), width=3)
        im = im.resize((int(im.width * self.k), int(im.height * self.k)))
        self.ph = ImageTk.PhotoImage(im); self.cv.delete("all"); self.cv.create_image(0, 0, anchor="nw", image=self.ph)
        n = {c: self.lab.count(c) for c in CYCLE}
        p = self.views[self.i]
        self.head.configure(text=(
            f"View {self.i + 1}/{len(self.views)}  {p.parent.parent.name} {p.parent.name} {p.name.replace('_polygons.json', '')}   "
            f"outlines {len(self.polys)}: good {n['good']}  partial {n['partial']}  wrong {n['wrong']}   missed {len(self.missed)}\n"
            + ("RECALL ROUND: RIGHT-CLICK (or Ctrl+click) every clear cotton leaf with NO green fill (blue dot).  "
               "Left click a green leaf only if it is grass / weed / not a leaf.\n"
               "Enter = save + next   B = back   Esc = quit" if RECALL else
               "Left click outline = good > partial > wrong   Right click a leaf without outline = missed   "
               "Enter = save + next   B = back   Esc = quit")))

    def at(self, e):
        return e.x / self.k, e.y / self.k

    def left(self, e):
        pt = self.at(e)
        hits = [i for i, poly in enumerate(self.polys) if inside(pt, poly)]
        if hits:
            i = min(hits, key=lambda i: area(self.polys[i]))      # smallest outline under the click
            self.lab[i] = CYCLE[(CYCLE.index(self.lab[i]) + 1) % 3]; self.draw()

    def right(self, e):
        x, y = self.at(e)
        near = [m for m in self.missed if (m[0] - x) ** 2 + (m[1] - y) ** 2 < 20 ** 2]
        if near:
            self.missed.remove(near[0])
        else:
            self.missed.append([round(x, 1), round(y, 1)])
        self.draw()

    def save_next(self, e=None):
        if self.i >= len(self.views):
            return
        self.ans[self.key(self.views[self.i])] = dict(labels=self.lab, missed=self.missed)
        OUT.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.ans), encoding="utf-8")
        self.i += 1; self.load()

    def back(self, e=None):
        if self.i > 0:
            self.i -= 1; self.load()


def report():
    ans = json.loads(ANS.read_text(encoding="utf-8"))
    if ANS_R.exists():                                   # recall round overrides labels + missed per view
        ans.update(json.loads(ANS_R.read_text(encoding="utf-8")))
    rows, tot = [], defaultdict(int)
    for k, a in sorted(ans.items()):
        n = {c: a["labels"].count(c) for c in CYCLE}
        m = len(a["missed"])
        for c in CYCLE:
            tot[c] += n[c]
        tot["missed"] += m
        allo = sum(n.values())
        rows.append(dict(view=k.replace("_polygons.json", ""), outlines=allo, good=n["good"], partial=n["partial"],
                         wrong=n["wrong"], missed=m,
                         precision=round(100 * (n["good"] + n["partial"]) / max(1, allo), 1),
                         recall=round(100 * (n["good"] + n["partial"]) / max(1, n["good"] + n["partial"] + m), 1)))
    g, pa, w, m = tot["good"], tot["partial"], tot["wrong"], tot["missed"]
    allo = g + pa + w
    leaves = g + pa
    lines = [f"SAM 3 leaf segmentation (prompt 'leaf', conf 0.3) checked by hand on {len(ans)} views",
             f"Outlines: {allo}   good {g}   partial {pa}   wrong (not a leaf) {w}   missed leaves {m}", "",
             f"Precision (outline is a real leaf, good + partial): {100 * leaves / max(1, allo):.1f}%",
             f"Strict precision (good outline only):               {100 * g / max(1, allo):.1f}%",
             (f"Recall (real leaves that got an outline):           {100 * leaves / max(1, leaves + m):.1f}%" if m else
              "Recall: NOT MEASURED (no missed leaves were marked; SAM 3 visibly misses some leaves)"),
             f"Outline quality (good among real leaves):           {100 * g / max(1, leaves):.1f}%",
             "", "Note: recall counts clearly visible leaves only; tiny far-away leaves are not counted as missed."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(OUT / "per_view.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print("\n".join(lines))
    for r in rows:
        print(f"  {r['view']:40s} outlines {r['outlines']:3d}  good {r['good']:3d}  partial {r['partial']:3d}  "
              f"wrong {r['wrong']:3d}  missed {r['missed']:3d}")
    print(f"\nFiles: {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--views", type=int, default=10)
    a = ap.parse_args()
    if a.report:
        report()
    else:
        RECALL = a.recall
        try:                                             # sharp, true-size window on scaled (150%) screens
            import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        r = tk.Tk()
        SHOW = max(500, min(1100, r.winfo_screenheight() - 190))
        Checker(r, pick_views(a.views)); r.mainloop()
