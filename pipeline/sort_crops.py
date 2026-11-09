"""
Fast keyboard sorter for leaf crops (healthy / diseased).

Keys
  D  = diseased        H = healthy
  X  = not a leaf / unusable (moved to crops\\rejected, not deleted)
  S  = skip (decide later)
  U  = undo last move
  Esc = quit (everything already sorted stays sorted)

It goes through, in order:
  crops\\auto_diseased (classifier's picks, most confident first) -> crops\\review_likely_diseased
  -> crops\\review_likely_healthy -> crops\\unsorted -> crops\\auto_healthy

Run:
  "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" "F:\\Project and completed courses\\Agri EYE 360\\sort_crops.py"
"""

import csv
import shutil
import sys
import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

CROPS = Path(__file__).resolve().parent / "disease_autolabel" / "crops"
SOURCES = ["auto_diseased", "review_likely_diseased", "review_likely_healthy", "unsorted", "auto_healthy"]
SCORES = CROPS.parent / "prelabel_scores.csv"
UNCERTAIN = CROPS.parent / "uncertain_crops.csv"
UNCERTAIN_MODE = "--uncertain" in sys.argv
# --review: re-check crops labelled diseased, cleanest-looking first (made by find_clean_diseased.py).
# D keeps the crop in diseased, H moves it to healthy, X to rejected. Kept ones are remembered
# in reviewed_ok.txt so they don't come back next time.
# "--review healthy" re-checks crops labelled healthy instead (list from find_suspect_healthy.py);
# there H keeps the crop and D moves it to diseased.
REVIEW_MODE = "--review" in sys.argv
_i = sys.argv.index("--review") + 1 if REVIEW_MODE else 0
REVIEW_CLASS = sys.argv[_i] if REVIEW_MODE and _i < len(sys.argv) and sys.argv[_i] in ("healthy", "diseased") else "diseased"
REVIEW = CROPS.parent / f"review_{REVIEW_CLASS}.csv"
REVIEW_DONE = CROPS.parent / ("reviewed_ok.txt" if REVIEW_CLASS == "diseased" else "reviewed_ok_healthy.txt")
KEEP_KEY = REVIEW_CLASS[0].upper()
REVIEW_LIMIT = 100000  # whole list
TARGETS = {"d": "diseased", "h": "healthy", "x": "rejected"}
EXAMPLE = Path(__file__).resolve().parent / "disease_samples" / "03_zoom3.jpg"
VIEW = 520


class Sorter:
    def __init__(self, root):
        self.root = root
        root.title(f"AgriEYE leaf sorter - REVIEW {REVIEW_CLASS} ({KEEP_KEY} = keep)" if REVIEW_MODE
                   else "AgriEYE leaf sorter")
        root.configure(bg="#1e1e1e")
        for t in TARGETS.values():
            (CROPS / t).mkdir(parents=True, exist_ok=True)
        self.files = []
        if UNCERTAIN_MODE:
            # only the crops the classifier is least sure about (made by score_uncertain.py)
            with UNCERTAIN.open(encoding="utf-8") as fh:
                self.files = [Path(r["path"]) for r in csv.DictReader(fh)]
        self.kept = set()
        if REVIEW_MODE:
            if REVIEW_DONE.exists():
                self.kept = set(REVIEW_DONE.read_text(encoding="utf-8").split())
            with REVIEW.open(encoding="utf-8") as fh:
                ranked = [Path(r["path"]) for r in csv.DictReader(fh)][:REVIEW_LIMIT]
            self.files = [f for f in ranked if f.name not in self.kept and f.exists()]
        for s in ([] if UNCERTAIN_MODE or REVIEW_MODE else SOURCES):
            files = sorted((CROPS / s).glob("*.png"))
            if s == "auto_diseased" and SCORES.exists():
                # most-likely-diseased first, using the classifier's scores
                with SCORES.open(encoding="utf-8") as fh:
                    p = {r["crop"]: float(r["p_diseased"]) for r in csv.DictReader(fh)}
                files.sort(key=lambda f: -p.get(f.name, 0))
            self.files += files
        self.i = 0
        self.history = []  # (dest_path, original_path)

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
        side.pack(side="left", padx=12, anchor="n")
        tk.Label(side, text="Diseased example", fg="#f99", bg="#1e1e1e", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        if EXAMPLE.exists():
            ex = Image.open(EXAMPLE)
            ex.thumbnail((300, 300))
            self.ex_img = ImageTk.PhotoImage(ex)
            tk.Label(side, image=self.ex_img, bg="#1e1e1e").pack(anchor="w", pady=4)
        tk.Label(side, justify="left", fg="#ddd", bg="#1e1e1e", font=("Segoe UI", 11), text=(
            "DISEASED: spots, specks, brown/tan\n"
            "dead patches, yellow-brown blotches\n\n"
            "HEALTHY: clean leaf, green OR purple\n\n" +
            (("REVIEW MODE: these are all labelled\n"
              "diseased now. D = keep diseased,\n"
              "H = actually healthy, X = unclear\n\n") if REVIEW_MODE and REVIEW_CLASS == "diseased" else
             ("REVIEW MODE: these are all labelled\n"
              "healthy now. H = keep healthy,\n"
              "D = actually diseased, X = unclear\n\n") if REVIEW_MODE else "") +
            "D  diseased      H  healthy\n"
            "X  not a leaf    S  skip\n"
            "U  undo          Esc  quit")).pack(anchor="w", pady=8)

        self.status = tk.Label(root, fg="#aaa", bg="#1e1e1e", font=("Segoe UI", 10))
        self.status.pack(fill="x", padx=12, pady=6)
        # holding a key down must not label many crops: ignore auto-repeat
        self.held = set()
        self.last_t = 0.0
        root.bind("<KeyRelease>", lambda e: self.held.discard(e.keysym.lower()))
        for k in ("d", "h", "x"):
            root.bind(k, lambda e, k=k: self.press(k))
            root.bind(k.upper(), lambda e, k=k: self.press(k))
        root.bind("s", lambda e: self.skip())
        root.bind("S", lambda e: self.skip())
        root.bind("u", lambda e: self.undo())
        root.bind("U", lambda e: self.undo())
        root.bind("<Escape>", lambda e: root.destroy())
        self.show()

    def tally(self):
        return "   ".join(f"{t}: {len(list((CROPS / t).glob('*.png')))}" for t in ("diseased", "healthy", "rejected"))

    def show(self):
        while self.i < len(self.files) and not self.files[self.i].exists():
            self.i += 1
        self.counts.config(text=self.tally())
        if self.i >= len(self.files):
            self.canvas.config(image="", text="All done - close the window", fg="white")
            self.info.config(text="Finished")
            return
        f = self.files[self.i]
        im = Image.open(f).convert("RGB")
        s = VIEW / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(im)
        self.canvas.config(image=self.img)
        self.info.config(text=f"{self.i + 1} / {len(self.files)}   from {f.parent.name}")
        self.status.config(text=str(f))

    def press(self, key):
        now = time.monotonic()
        if key in self.held or now - self.last_t < 0.15:
            return  # key held down / auto-repeat
        self.held.add(key)
        self.last_t = now
        self.move(key)

    def move(self, key):
        if self.i >= len(self.files):
            return
        src = self.files[self.i]
        dst = CROPS / TARGETS[key] / src.name
        if dst == src:  # review mode: D = keep as diseased
            self.kept.add(src.name)
            self.save_kept()
        else:
            shutil.move(str(src), str(dst))
        self.history.append((dst, src))
        self.i += 1
        self.show()

    def save_kept(self):
        REVIEW_DONE.write_text("\n".join(sorted(self.kept)), encoding="utf-8")

    def skip(self):
        self.i += 1
        self.show()

    def undo(self):
        if not self.history:
            return
        dst, src = self.history.pop()
        if dst == src:
            self.kept.discard(src.name)
            self.save_kept()
        elif dst.exists():
            shutil.move(str(dst), str(src))
        self.i = self.files.index(src)
        self.show()


if __name__ == "__main__":
    root = tk.Tk()
    Sorter(root)
    root.mainloop()
