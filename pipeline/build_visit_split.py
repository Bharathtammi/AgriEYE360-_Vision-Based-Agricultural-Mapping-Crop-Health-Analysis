"""
ONE-VISIT development set: improve the boll detector on ONE visit only, then test once on every other visit.

  python build_visit_split.py            # default visit 2026-08-04
  python build_visit_split.py 20260731

Source : agrieye_output/boll_detector/data_photo_split (140 different pictures = 35 photos, copies merged).
Visit  : its photos sorted by time; split by PHOTO (all 4 directions together):
         9 photos -> train 6 / valid 1 / test 2, chosen so test holds ~22 % and valid ~11 % of BOTH the open
         and the closed bolls (open bolls are rare; a test without them cannot measure them).
Others : every other visit = an untouched test set of its own (used only at the end).
Output : agrieye_output/boll_detector/visit_<date>/{train,valid,test}/{images,labels}, data.yaml,
         other_visits/<date>/{images,labels}, photos.csv
No training.
"""
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "agrieye_output" / "boll_detector" / "data_photo_split"
VISIT = sys.argv[1].replace("-", "") if len(sys.argv) > 1 else "20260804"
DST = ROOT / "agrieye_output" / "boll_detector" / f"visit_{VISIT}"


def put(r, d):
    for sub in ("images", "labels"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / r["split"] / "images" / r["view"], d / "images" / r["view"])
    shutil.copy2(SRC / r["split"] / "labels" / f"{Path(r['view']).stem}.txt", d / "labels" / f"{Path(r['view']).stem}.txt")


def boxes(d):
    o = c = 0
    for f in (d / "labels").glob("*.txt"):
        for l in f.read_text().splitlines():
            if l.strip():
                o += l[0] == "0"; c += l[0] == "1"
    return o, c


def main():
    rows = list(csv.DictReader(open(SRC / "split.csv", encoding="utf-8")))
    if DST.exists():
        shutil.rmtree(DST)                               # only this script writes here
    photos = defaultdict(list)
    for r in rows:
        if r["date"] == VISIT:
            photos[r["group"]].append(r)
    order = sorted(photos, key=lambda g: min(x["view"].split("_")[-2] for x in photos[g]))
    n = len(order)
    def cnt(g):                                          # open / closed bolls of one photo
        o = c = 0
        for r in photos[g]:
            for l in (SRC / r["split"] / "labels" / f"{Path(r['view']).stem}.txt").read_text().splitlines():
                if l.strip():
                    o += l[0] == "0"; c += l[0] == "1"
        return o, c
    oc = {g: cnt(g) for g in order}
    O, C = sum(v[0] for v in oc.values()) or 1, sum(v[1] for v in oc.values()) or 1
    def cost(t, v):                                      # aim: test ~22 %, valid ~11 % of BOTH open and closed bolls
        to, tc = sum(oc[g][0] for g in t), sum(oc[g][1] for g in t)
        return abs(to / O - .22) + abs(tc / C - .22) + abs(oc[v][0] / O - .11) + abs(oc[v][1] / C - .11)
    from itertools import combinations
    k = 2 if n >= 6 else 1
    t, v = min(((set(t), v) for t in combinations(order, k) for v in order if v not in t), key=lambda x: cost(*x))
    test, valid = t, {v}
    out = []
    for g in order:
        s = "test" if g in test else "valid" if g in valid else "train"
        for r in photos[g]:
            put(r, DST / s)
        out.append(dict(photo=photos[g][0]["view"].rsplit("_", 1)[0], split=s, pictures=len(photos[g])))
    for r in rows:
        if r["date"] != VISIT:
            put(r, DST / "other_visits" / r["date"])

    (DST / "data.yaml").write_text(f"path: {DST.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                                   "nc: 2\nnames: ['open_boll', 'closed_boll']\n", encoding="utf-8")
    with open(DST / "photos.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["photo", "split", "pictures"]); w.writeheader(); w.writerows(out)
    print(f"Development visit {VISIT[:4]}-{VISIT[4:6]}-{VISIT[6:]} ({n} photos)")
    for s in ("train", "valid", "test"):
        o, c = boxes(DST / s)
        print(f"  {s:<6}{sum(x['split'] == s for x in out):>3} photos {len(list((DST / s / 'images').glob('*.jpg'))):>4} pictures"
              f"   open {o:>4}  closed {c:>5}")
    print("Other visits (untouched, for the final test only):")
    hand = {r["view"]: r["labels"] for r in rows}
    for d in sorted((DST / "other_visits").iterdir()):
        o, c = boxes(d); imgs = list((d / "images").glob("*.jpg"))
        auto = sum(hand[f.name] != "hand-checked" for f in imgs)
        print(f"  {d.name:<10}{len(imgs):>4} pictures   open {o:>4}  closed {c:>5}"
              + (f"   ({auto} with automatic labels only)" if auto else ""))
    print(f"Files: {DST}")


if __name__ == "__main__":
    main()
