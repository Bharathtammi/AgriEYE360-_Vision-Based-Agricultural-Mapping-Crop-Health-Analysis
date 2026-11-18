"""
Option C: a CLEAN split of ALL different boll pictures, by PHOTO - no photo (and no copy of a picture) in two sets.

  python build_photo_split.py

Why: the POV camera saved the same frame again under new timestamps (one picture up to 38 times), and the old
hand-checked split put other directions of the same photo into train, valid and test. Both made scores look better.

Pictures: every view of pov_all_labels/bolls_dataset + the 140 hand-checked views (roboflow_export/1_cotton_boll_detection),
          copies merged (same bytes, or a 24 px grey thumbnail differing < SAME). One file per different picture:
          the hand-checked view if there is one (hand labels), else the POV view (automatic labels, hand-reviewed today).
Groups  : pictures that belong together never split up: copies, the 4 directions of a photo, photos of the same day
          within 15 s, the hand view and its POV view.
Folds   : groups dealt into 5 folds, balanced per day and by number of views (seed 0) -> split.csv has the fold of
          every picture, ready for 5-fold cross-validation later.
Split   : test = fold 0, valid = fold 1, train = folds 2-4.
Output  : agrieye_output/boll_detector/data_photo_split/{train,valid,test}/{images,labels}, split.csv, summary.txt
No training. data_big and the old hand split are not changed.
"""
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "roboflow_export" / "1_cotton_boll_detection"
NEW = ROOT / "agrieye_output" / "pov_all_labels" / "bolls_dataset"
POV = ROOT / "agrieye_output" / "pov_all"
DST = ROOT / "agrieye_output" / "boll_detector" / "data_photo_split"
SAME, WINDOW_S, FOLDS, SEED = 1.0, 15, 5, 0


def thumb(p):
    return np.asarray(Image.open(p).convert("L").resize((24, 24)), np.float32)


def secs(stem):                                             # ..._20260804_171515[_N] -> (date, seconds of day)
    parts = stem.split("_")
    i = next(k for k, x in enumerate(parts) if len(x) == 8 and x.isdigit() and x.startswith("2026"))
    d, t = parts[i], parts[i + 1]
    return d, int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


class Groups:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a

    def join(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    # all views: POV first, then hand-checked
    nodes = []
    for r in csv.DictReader(open(NEW / "views.csv", encoding="utf-8")):
        if not r["copy_of"]:
            nodes.append(dict(img=POV / r["date"] / r["view"], lab=NEW / "labels" / f"{Path(r['view']).stem}.txt",
                              photo="pov:" + r["file"], src="automatic (SAM 3 + R1, hand-reviewed)", hand=None))
    for img in sorted(OLD.glob("*/images/*.jpg")):
        nodes.append(dict(img=img, lab=OLD / img.parent.parent.name / "labels" / f"{img.stem}.txt",
                          photo="hand:" + img.stem.rsplit("_", 1)[0], src="hand-checked",
                          hand=img.parent.parent.name))
    T = np.stack([thumb(n["img"]) for n in nodes])
    md5 = [hashlib.md5(n["img"].read_bytes()).hexdigest() for n in nodes]
    # pictures = copies merged
    pic = Groups(len(nodes))
    first = {}
    for i, h in enumerate(md5):
        pic.join(i, first.setdefault(h, i))
    reps = sorted({pic.find(i) for i in range(len(nodes))})
    for a in reps:
        for b in np.where(np.abs(T - T[a]).mean((1, 2)) < SAME)[0]:
            pic.join(int(b), a)
    for i, n in enumerate(nodes):                            # hand view <-> its POV view (slightly different slicing)
        if n["hand"]:
            d = np.abs(T[:len(nodes)] - T[i]).mean((1, 2)); d[i] = 99
            j = int(d.argmin())
            if d[j] < 3:
                pic.join(i, j)
    # groups = pictures that must stay together
    grp = Groups(len(nodes))
    for i in range(len(nodes)):
        grp.join(i, pic.find(i))
    by_photo = defaultdict(list)
    for i, n in enumerate(nodes):
        by_photo[n["photo"]].append(i)
    for ms in by_photo.values():
        for m in ms[1:]:
            grp.join(m, ms[0])
    pov_photos = [(secs(p.split(":", 1)[1].rsplit(".", 1)[0]), ms[0]) for p, ms in by_photo.items() if p.startswith("pov:")]
    pov_photos.sort()
    for k, ((d, t), i) in enumerate(pov_photos):
        for (d2, t2), j in pov_photos[k + 1:]:
            if d2 != d or t2 - t > WINDOW_S:
                break
            grp.join(i, j)
    # one file per picture: the hand-checked one if any (test/valid labels preferred over train), else the earliest POV
    pictures = defaultdict(list)
    for i in range(len(nodes)):
        pictures[pic.find(i)].append(i)
    rank = {"test": 0, "valid": 1, "train": 2, None: 3}
    chosen = []
    for p, ms in pictures.items():
        best = min(ms, key=lambda i: (rank[nodes[i]["hand"]], str(nodes[i]["img"].name)))
        chosen.append(dict(nodes[best], group=grp.find(best), copies=len(ms)))
    # deal groups into folds, per day, biggest first, to the fold with fewest views of that day so far
    groups = defaultdict(list)
    for c in chosen:
        groups[c["group"]].append(c)
    day = {g: Counter(secs(c["img"].stem)[0] for c in cs).most_common(1)[0][0] for g, cs in groups.items()}
    rng = random.Random(SEED)
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: (day[g], -len(groups[g])))
    load = defaultdict(lambda: [0] * FOLDS)
    total = [0] * FOLDS
    fold = {}
    for g in order:
        f = min(range(FOLDS), key=lambda k: (load[day[g]][k], total[k], k))
        fold[g] = f; load[day[g]][f] += len(groups[g]); total[f] += len(groups[g])
    # write
    if DST.exists():
        shutil.rmtree(DST)                                   # only this script writes here
    split_of = lambda f: "test" if f == 0 else "valid" if f == 1 else "train"
    rows, boxes, views = [], Counter(), Counter()
    for c in sorted(chosen, key=lambda c: c["img"].name):
        f = fold[c["group"]]; s = split_of(f)
        name = c["img"].name
        for sub in ("images", "labels"):
            (DST / s / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy2(c["img"], DST / s / "images" / name)
        shutil.copy2(c["lab"], DST / s / "labels" / f"{Path(name).stem}.txt")
        for line in c["lab"].read_text().splitlines():
            if line.strip():
                boxes[(s, line.split()[0])] += 1
        views[s] += 1; views[(s, c["src"])] += 1
        rows.append(dict(view=name, date=secs(c["img"].stem)[0], fold=f, split=s, group=c["group"],
                         labels=c["src"], old_hand_split=c["hand"] or "", copies=c["copies"]))
    with open(DST / "split.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (DST / "data.yaml").write_text(f"path: {DST.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                                   "nc: 2\nnames: ['open_boll', 'closed_boll']\n", encoding="utf-8")
    ng = Counter(split_of(fold[g]) for g in groups)
    days = sorted({r["date"] for r in rows})
    lines = [f"Clean photo split of all different boll pictures ({len(nodes)} views looked at -> {len(chosen)} different "
             f"pictures in {len(groups)} independent groups)", ""]
    lines.append(f"{'split':<7}{'pictures':>9}{'groups':>8}{'hand':>6}{'auto':>6}{'open':>7}{'closed':>8}   per day")
    for s in ("train", "valid", "test"):
        per = Counter(r["date"] for r in rows if r["split"] == s)
        lines.append(f"{s:<7}{views[s]:>9}{ng[s]:>8}{views[(s, 'hand-checked')]:>6}"
                     f"{views[s] - views[(s, 'hand-checked')]:>6}{boxes[(s, '0')]:>7}{boxes[(s, '1')]:>8}   "
                     + "  ".join(f"{d[4:6]}/{d[6:]}:{per[d]}" for d in days if per[d]))
    lines.append("")
    lines.append("Folds (for cross-validation later): " + ", ".join(f"fold {k}: {total[k]}" for k in range(FOLDS)))
    old = Counter((r["old_hand_split"], r["split"]) for r in rows if r["old_hand_split"])
    lines.append("Old hand split -> new split: " + ", ".join(f"{a}->{b} {v}" for (a, b), v in sorted(old.items())))
    lines.append(f"Files: {DST}")
    (DST / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
