"""
Fast keyboard sorter for boll detections (made by make_boll_crops.py).

Look at what is inside the GREEN box:
  O  = open boll      (white / grey fibre showing - even a boll that has just cracked open)
  C  = closed boll    (green, red or maroon boll or bud, no fibre showing)
  X  = not a boll     (leaf, sky, stem, blur, can't tell)
  S  = skip (decide later)     U = undo last     Esc = quit

Order: crops SAM 3 called "open" first, then "flower", then "closed".
Every key press moves the file at once, so you can stop any time and continue later.

Run:
  "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" "F:\\Project and completed courses\\Agri EYE 360\\sort_bolls.py"
"""
import csv
import shutil
import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

DIR = Path(__file__).resolve().parent / "agrieye_output" / "boll_crops"
UNSORTED = DIR / "unsorted"
TARGETS = {"o": "open", "c": "closed", "x": "not_boll"}
VIEW = 560
GROUP = {"open_boll": 'SAM 3 said "open"', "flower": 'SAM 3 said "flower"', "closed_boll": 'SAM 3 said "closed"'}


class Sorter:
    def __init__(self, root):
        self.root = root
        root.title("AgriEYE boll sorter")
        root.configure(bg="#1e1e1e")
        with open(DIR / "manifest.csv", encoding="utf-8") as fh:
            self.meta = {r["crop"]: r for r in csv.DictReader(fh)}
        self.files = [UNSORTED / n for n in self.meta if (UNSORTED / n).exists()]
        self.i = 0
        self.history = []

        top = tk.Frame(root, bg="#1e1e1e")
        top.pack(fill="x", padx=12, pady=8)
        self.info = tk.Label(top, fg="white", bg="#1e1e1e", font=("Segoe UI", 12), anchor="w")
        self.info.pack(side="left")
        self.counts = tk.Label(top, fg="#9fd", bg="#1e1e1e", font=("Segoe UI", 12), anchor="e")
        self.counts.pack(side="right")
        body = tk.Frame(root, bg="#1e1e1e")
        body.pack(padx=12)
        self.canvas = tk.Label(body, bg="black", width=VIEW, height=VIEW)
        self.canvas.pack(side="left")
        side = tk.Frame(body, bg="#1e1e1e")
        side.pack(side="left", padx=14, anchor="n")
        tk.Label(side, justify="left", fg="#ddd", bg="#1e1e1e", font=("Segoe UI", 11), text=(
            "Judge only what is inside the\nGREEN box.\n\n"
            "O  OPEN boll\n    white / grey fibre showing,\n    even just cracked open\n\n"
            "C  CLOSED boll or bud\n    green / red / maroon,\n    no fibre showing\n\n"
            "X  NOT a boll\n    leaf, sky, stem, blur,\n    can't tell\n\n"
            "S  skip     U  undo     Esc  quit")).pack(anchor="w")
        self.group = tk.Label(side, fg="#fc6", bg="#1e1e1e", font=("Segoe UI", 11, "bold"), justify="left")
        self.group.pack(anchor="w", pady=16)
        self.status = tk.Label(root, fg="#aaa", bg="#1e1e1e", font=("Segoe UI", 10))
        self.status.pack(fill="x", padx=12, pady=6)

        self.held, self.last_t = set(), 0.0          # ignore a held-down key (auto-repeat)
        root.bind("<KeyRelease>", lambda e: self.held.discard(e.keysym.lower()))
        for k in TARGETS:
            root.bind(k, lambda e, k=k: self.press(k))
            root.bind(k.upper(), lambda e, k=k: self.press(k))
        for k in "sS":
            root.bind(k, lambda e: self.skip())
        for k in "uU":
            root.bind(k, lambda e: self.undo())
        root.bind("<Escape>", lambda e: root.destroy())
        self.show()

    def tally(self):
        return "   ".join(f"{t}: {len(list((DIR / t).glob('*.jpg')))}" for t in TARGETS.values()) + \
               f"   left: {len(list(UNSORTED.glob('*.jpg')))}"

    def show(self):
        while self.i < len(self.files) and not self.files[self.i].exists():
            self.i += 1
        self.counts.config(text=self.tally())
        if self.i >= len(self.files):
            self.canvas.config(image="", text="All done - close the window", fg="white")
            self.info.config(text="Finished")
            return
        f = self.files[self.i]
        m = self.meta[f.name]
        im = Image.open(f).convert("RGB")
        s = VIEW / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(im)
        self.canvas.config(image=self.img)
        self.info.config(text=f"{self.i + 1} / {len(self.files)}")
        self.group.config(text=f"{GROUP[m['sam_label']]}\n(score {float(m['score']):.2f})")
        self.status.config(text=f"{m['station']}  {m['date']}  {m['direction']}   {f}")

    def press(self, key):
        now = time.monotonic()
        if key in self.held or now - self.last_t < 0.15:
            return
        self.held.add(key)
        self.last_t = now
        if self.i >= len(self.files):
            return
        src = self.files[self.i]
        dst = DIR / TARGETS[key] / src.name
        shutil.move(str(src), str(dst))
        self.history.append((dst, src))
        self.i += 1
        self.show()

    def skip(self):
        self.i += 1
        self.show()

    def undo(self):
        if not self.history:
            return
        dst, src = self.history.pop()
        if dst.exists():
            shutil.move(str(dst), str(src))
        self.i = self.files.index(src)
        self.show()


if __name__ == "__main__":
    for t in TARGETS.values():
        (DIR / t).mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    Sorter(root)
    root.mainloop()
