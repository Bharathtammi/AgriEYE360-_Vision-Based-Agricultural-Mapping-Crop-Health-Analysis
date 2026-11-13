"""
FAST grid sorter for boll crops: 30 crops per page, each already filled in with SAM 3's guess.
You only fix the wrong ones, then accept the whole page.

Look at what is inside the GREEN box of each crop:
  open    = white / grey fibre showing (even just cracked)
  closed  = green, red or maroon boll or bud, no fibre
  not a boll = leaf, flower, stem, sky, blur, can't tell

  Left click   = change label: closed -> open -> not a boll -> skip -> closed ...
  Right click  = not a boll
  Enter        = accept page (files are moved into open / closed / not_boll; skip stays unsorted)
  U            = undo last page        Esc = quit (current page is NOT saved)

Pages come in the order: SAM "open" -> SAM "flower" -> SAM "closed".
Run:
  "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" "F:\\Project and completed courses\\Agri EYE 360\\sort_bolls_grid.py"
Missed-boll candidates (find_missed_bolls.py): add  --dir boll_candidates
"""
import argparse
import csv
import shutil
import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", default="boll_crops", help="folder in agrieye_output: boll_crops or boll_candidates")
DIR = Path(__file__).resolve().parent / "agrieye_output" / _ap.parse_args().dir
UNSORTED = DIR / "unsorted"
CANDIDATES = DIR.name == "boll_candidates"
COLS, ROWS, T = 6, 5, 170
CYCLE = ["closed", "open", "not_boll", "skip"]
COLOR = {"open": "#ffffff", "closed": "#ff4fd0", "not_boll": "#ff3b3b", "skip": "#777777"}
TEXT = {"open": "OPEN", "closed": "closed", "not_boll": "NOT A BOLL", "skip": "skip"}
DEFAULT = {"open_boll": "open", "flower": "closed", "closed_boll": "closed"}


class Grid:
    def __init__(self, root):
        self.root = root
        root.title("AgriEYE boll grid sorter - MISSED BOLL CANDIDATES" if CANDIDATES else "AgriEYE boll grid sorter")
        root.configure(bg="#1e1e1e")
        with open(DIR / "manifest.csv", encoding="utf-8") as fh:
            self.meta = {r["crop"]: r for r in csv.DictReader(fh)}
        self.queue = [n for n in self.meta if (UNSORTED / n).exists()]   # manifest is already in sorting order
        self.skipped, self.history, self.last = set(), [], 0.0
        self.head = tk.Label(root, bg="#1e1e1e", fg="white", font=("Segoe UI", 11), justify="left", anchor="w")
        self.head.pack(fill="x", padx=8, pady=4)
        self.frame = tk.Frame(root, bg="#1e1e1e"); self.frame.pack(padx=6, pady=4)
        self.cells = []
        for i in range(COLS * ROWS):
            c = tk.Canvas(self.frame, width=T, height=T + 22, bg="#1e1e1e", highlightthickness=4,
                          highlightbackground="#1e1e1e")
            c.grid(row=i // COLS, column=i % COLS, padx=3, pady=3)
            c.bind("<Button-1>", lambda e, k=i: self.cycle(k))
            c.bind("<Button-3>", lambda e, k=i: self.set(k, "not_boll"))
            self.cells.append(c)
        root.bind("<Return>", self.accept)
        root.bind("<KP_Enter>", self.accept)
        root.bind("u", self.undo); root.bind("U", self.undo)
        root.bind("<Escape>", lambda e: root.destroy())
        self.load()

    def counts(self):
        return {f: len(list((DIR / f).glob("*.jpg"))) for f in ("open", "closed", "not_boll", "unsorted")}

    def load(self):
        todo = [n for n in self.queue if (UNSORTED / n).exists() and n not in self.skipped]
        self.page = todo[:COLS * ROWS]
        self.labels = [DEFAULT[self.meta[n]["sam_label"]] for n in self.page]
        self.imgs = []
        for i, c in enumerate(self.cells):
            c.delete("all")
            if i >= len(self.page):
                c.configure(highlightbackground="#1e1e1e"); continue
            im = Image.open(UNSORTED / self.page[i]).convert("RGB"); im.thumbnail((T, T))
            ph = ImageTk.PhotoImage(im); self.imgs.append(ph)
            c.create_image(T // 2, T // 2, image=ph)
            self.draw(i)
        k = self.counts()
        group = ({"open_boll": 'POSSIBLY MISSED boll - detector says "open"', "closed_boll": 'POSSIBLY MISSED boll - detector says "closed"'}
                 if CANDIDATES else
                 {"open_boll": 'SAM said "open"', "flower": 'SAM said "flower"', "closed_boll": 'SAM said "closed"'})
        g = group[self.meta[self.page[0]]["sam_label"]] if self.page else "all done"
        self.head.configure(text=(
            f"{g}  |  left: {len(todo)}   open {k['open']}  closed {k['closed']}  not_boll {k['not_boll']}\n"
            "Left click = change label (closed > open > not a boll > skip)   Right click = not a boll   "
            "Enter = accept page   U = undo page   Esc = quit"))

    def draw(self, i):
        c, lab = self.cells[i], self.labels[i]
        c.configure(highlightbackground=COLOR[lab])
        c.delete("lab")
        c.create_rectangle(0, T, T + 4, T + 22, fill=COLOR[lab], outline="", tags="lab")
        c.create_text(T // 2, T + 11, text=TEXT[lab], fill="black" if lab != "skip" else "white",
                      font=("Segoe UI", 10, "bold"), tags="lab")

    def cycle(self, i):
        if i < len(self.page):
            self.labels[i] = CYCLE[(CYCLE.index(self.labels[i]) + 1) % len(CYCLE)]; self.draw(i)

    def set(self, i, lab):
        if i < len(self.page):
            self.labels[i] = lab; self.draw(i)

    def accept(self, e=None):
        if not self.page or time.time() - self.last < 0.8:       # held-key guard
            return
        self.last = time.time()
        moved = []
        for n, lab in zip(self.page, self.labels):
            if lab == "skip":
                self.skipped.add(n); continue
            shutil.move(str(UNSORTED / n), str(DIR / lab / n)); moved.append((n, lab))
        self.history.append(moved)
        self.load()

    def undo(self, e=None):
        if not self.history:
            return
        for n, lab in self.history.pop():
            if (DIR / lab / n).exists():
                shutil.move(str(DIR / lab / n), str(UNSORTED / n))
        self.load()


if __name__ == "__main__":
    r = tk.Tk(); Grid(r); r.mainloop()
