"""
Check how reliable the disease model is on whole field images.

1) Run the model on some views (pipeline.py predict ... --out <folder>)
2) python verify_leaves.py [<folder>]          -> you label every leaf the model found
     H = healthy cotton leaf     D = diseased cotton leaf
     X = not a cotton leaf (boll, weed, stem, sky, two leaves merged, can't tell)
     U = undo     Esc = quit (answers are saved after every key, continue any time)
   The model's answer is HIDDEN while you label, so it can't influence you.
3) python verify_leaves.py [<folder>] --report  -> reliability per image + overall
     writes <folder>/reliability_report.csv and <folder>/reliability_mistakes.jpg

Default folder: sample_results/field_test
"""
import csv
import sys
import time
import tkinter as tk
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

ROOT = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FOLDER = Path(args[0]) if args else ROOT / "sample_results" / "field_test"
ANS = FOLDER / "verify_labels.csv"
KEYS = {"h": "healthy", "d": "diseased", "x": "not_leaf"}


def load_items():
    imgs = {Path(r["image"]).stem: r["image"] for r in csv.DictReader(open(FOLDER / "summary.csv", encoding="utf-8"))}
    items = []
    for f in sorted(FOLDER.glob("*_leaves.csv")):
        stem = next((s for s in imgs if f.name.startswith(s + "_")), None)
        if stem is None:
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            items.append(dict(r, image=imgs[stem], view=stem))
    return items


def load_answers():
    if not ANS.exists():
        return {}
    return {(r["view"], r["leaf"]): r["truth"] for r in csv.DictReader(open(ANS, encoding="utf-8"))}


def save_answers(ans):
    with open(ANS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["view", "leaf", "truth"])
        for (v, l), t in sorted(ans.items()):
            w.writerow([v, l, t])


class Checker:
    def __init__(self, root):
        self.root = root
        root.title("AgriEYE - check the model, leaf by leaf")
        root.configure(bg="#1e1e1e")
        self.items = load_items()
        self.ans = load_answers()
        self.order = [i for i, it in enumerate(self.items) if (it["view"], it["leaf"]) not in self.ans]
        self.k = 0
        self.hist = []
        self.cache = {}
        top = tk.Frame(root, bg="#1e1e1e"); top.pack(fill="x", padx=12, pady=8)
        self.info = tk.Label(top, fg="white", bg="#1e1e1e", font=("Segoe UI", 12)); self.info.pack(side="left")
        body = tk.Frame(root, bg="#1e1e1e"); body.pack(padx=12)
        self.full = tk.Label(body, bg="black"); self.full.pack(side="left")
        self.zoom = tk.Label(body, bg="black"); self.zoom.pack(side="left", padx=10)
        tk.Label(body, justify="left", fg="#ddd", bg="#1e1e1e", font=("Segoe UI", 11), text=(
            "Judge the leaf in the YELLOW box.\n\n"
            "H  healthy cotton leaf\n"
            "D  diseased cotton leaf\n     (spots, specks, holes,\n      brown dead patches)\n"
            "X  not a cotton leaf\n     (boll, weed, stem, sky,\n      merged leaves, can't tell)\n\n"
            "U  undo     Esc  quit\n\n"
            "The model's answer is hidden.")).pack(side="left", anchor="n", padx=10)
        self.held, self.last = set(), 0.0
        root.bind("<KeyRelease>", lambda e: self.held.discard(e.keysym.lower()))
        for k in KEYS:
            root.bind(k, lambda e, k=k: self.press(k)); root.bind(k.upper(), lambda e, k=k: self.press(k))
        root.bind("u", lambda e: self.undo()); root.bind("U", lambda e: self.undo())
        root.bind("<Escape>", lambda e: root.destroy())
        self.show()

    def img(self, path):
        if path not in self.cache:
            self.cache = {path: Image.open(path).convert("RGB")}
        return self.cache[path]

    def show(self):
        done = len(self.ans)
        if self.k >= len(self.order):
            self.info.config(text=f"All {len(self.items)} leaves checked - close the window and ask for the report")
            self.full.config(image=""); self.zoom.config(image="")
            return
        it = self.items[self.order[self.k]]
        im = self.img(it["image"])
        x1, y1, x2, y2 = (float(it[c]) for c in ("x1", "y1", "x2", "y2"))
        full = im.copy()
        d = ImageDraw.Draw(full)
        d.rectangle([x1, y1, x2, y2], outline=(255, 220, 0), width=6)
        full.thumbnail((560, 560))
        self.full_img = ImageTk.PhotoImage(full)
        self.full.config(image=self.full_img)
        m = max(x2 - x1, y2 - y1) * 0.35 + 20
        z = im.crop((max(0, x1 - m), max(0, y1 - m), min(im.width, x2 + m), min(im.height, y2 + m)))
        dz = ImageDraw.Draw(z)
        dz.rectangle([x1 - max(0, x1 - m), y1 - max(0, y1 - m), x2 - max(0, x1 - m), y2 - max(0, y1 - m)],
                     outline=(255, 220, 0), width=3)
        s = 460 / max(z.size)
        z = z.resize((max(1, int(z.width * s)), max(1, int(z.height * s))), Image.LANCZOS)
        self.zoom_img = ImageTk.PhotoImage(z)
        self.zoom.config(image=self.zoom_img)
        self.info.config(text=f"leaf {self.k + 1} / {len(self.order)} left   (checked {done} of {len(self.items)})   "
                              f"view {it['view']}")

    def press(self, key):
        now = time.monotonic()
        if key in self.held or now - self.last < 0.15 or self.k >= len(self.order):
            return
        self.held.add(key); self.last = now
        it = self.items[self.order[self.k]]
        self.ans[(it["view"], it["leaf"])] = KEYS[key]
        save_answers(self.ans)
        self.hist.append((it["view"], it["leaf"]))
        self.k += 1
        self.show()

    def undo(self):
        if not self.hist:
            return
        self.ans.pop(self.hist.pop(), None)
        save_answers(self.ans)
        self.k -= 1
        self.show()


def report():
    items = load_items()
    ans = load_answers()
    per = defaultdict(lambda: defaultdict(int))
    mistakes = []
    for it in items:
        t = ans.get((it["view"], it["leaf"]))
        if t is None:
            continue
        v = per[it["view"]]
        v["checked"] += 1
        pred = it["label"]
        chk = it["needs_check"] in ("1", "True", "true")
        if t == "not_leaf":
            v["not_leaf"] += 1
            v["not_leaf_called_diseased"] += pred == "diseased"
            continue
        v["leaves"] += 1
        v["correct"] += pred == t
        v["true_diseased"] += t == "diseased"
        v["pred_diseased"] += pred == "diseased"
        v["tp"] += pred == t == "diseased"
        v["check"] += chk
        v["confident"] += not chk
        v["confident_correct"] += (not chk) and pred == t
        if pred != t:
            mistakes.append((it, t))
    if not per:
        raise SystemExit("No answers yet - run verify_leaves.py first.")
    rows = []
    tot = defaultdict(int)
    for view, v in sorted(per.items()):
        for k_, n in v.items():
            tot[k_] += n
        rows.append((view, v))
    rows.append(("ALL", tot))
    out = []
    print(f"\n{'view':22s} {'checked':>7s} {'notleaf':>7s} {'acc':>6s} {'conf.acc':>8s} "
          f"{'model %dis':>10s} {'true %dis':>9s} {'dis.recall':>10s} {'dis.prec':>8s}")
    for view, v in rows:
        L = max(1, v["leaves"])
        r = dict(view=view, checked=v["checked"], not_leaf=v["not_leaf"],
                 not_leaf_pct=round(100 * v["not_leaf"] / max(1, v["checked"]), 1),
                 accuracy=round(100 * v["correct"] / L, 1),
                 confident_accuracy=round(100 * v["confident_correct"] / max(1, v["confident"]), 1),
                 model_pct_diseased=round(100 * v["pred_diseased"] / L, 1),
                 true_pct_diseased=round(100 * v["true_diseased"] / L, 1),
                 diseased_recall=round(100 * v["tp"] / max(1, v["true_diseased"]), 1),
                 diseased_precision=round(100 * v["tp"] / max(1, v["pred_diseased"]), 1),
                 not_leaf_called_diseased=v["not_leaf_called_diseased"])
        out.append(r)
        print(f"{view:22s} {r['checked']:7d} {r['not_leaf']:7d} {r['accuracy']:5.1f}% {r['confident_accuracy']:7.1f}% "
              f"{r['model_pct_diseased']:9.1f}% {r['true_pct_diseased']:8.1f}% {r['diseased_recall']:9.1f}% "
              f"{r['diseased_precision']:7.1f}%")
    with open(FOLDER / "reliability_report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    # mistakes sheet
    T = 150
    tiles = []
    cache = {}
    for it, t in mistakes[:96]:
        im = cache.setdefault(it["image"], Image.open(it["image"]).convert("RGB"))
        x1, y1, x2, y2 = (float(it[c]) for c in ("x1", "y1", "x2", "y2"))
        c = im.crop((x1, y1, x2, y2)); c.thumbnail((T, T))
        tile = Image.new("RGB", (T, T + 28), "white"); tile.paste(c, ((T - c.width) // 2, 0))
        dr = ImageDraw.Draw(tile)
        dr.text((3, T + 1), f"model: {it['label']} {float(it['p_diseased']):.2f}", fill=(200, 0, 0))
        dr.text((3, T + 14), f"you: {t}", fill=(0, 120, 0))
        tiles.append(tile)
    if tiles:
        cols = 8
        sheet = Image.new("RGB", (cols * (T + 4), ((len(tiles) - 1) // cols + 1) * (T + 32)), (235, 235, 235))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * (T + 4) + 2, (i // cols) * (T + 32) + 2))
        sheet.save(FOLDER / "reliability_mistakes.jpg", quality=88)
    print(f"\nReport:   {FOLDER / 'reliability_report.csv'}")
    print(f"Mistakes: {FOLDER / 'reliability_mistakes.jpg'}  ({len(mistakes)} leaves where model and you disagree)")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        root = tk.Tk()
        Checker(root)
        root.mainloop()
