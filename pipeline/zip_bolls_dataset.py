"""
Pack agrieye_output/pov_all_labels/bolls_dataset into ONE zip ready to upload to Roboflow (YOLO format).

  python zip_bolls_dataset.py

Split 70 / 20 / 10 (train / valid / test) by PHOTO: the 4 views of a photo always go to the same split, so the test
views are never views of a photo seen in training. Split is made per date, so every visit is in every split.
Exact byte copies of another photo are left out (Roboflow would reject them as duplicates; they add no new labels).
Views with no boll are kept (they teach the model what is not a boll).

Output: roboflow_export/3_cotton_bolls_all_povs.zip     train/ valid/ test/ (images + labels) + data.yaml + README.txt
"""
import csv
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "agrieye_output" / "pov_all_labels" / "bolls_dataset"
OUT = ROOT / "roboflow_export" / "3_cotton_bolls_all_povs.zip"
NAMES = ["open_boll", "closed_boll"]


def main():
    rows = [r for r in csv.DictReader(open(SRC / "views.csv", encoding="utf-8")) if not r["copy_of"]]
    photos = defaultdict(list)
    for r in rows:
        photos[(r["date"], r["file"])].append(r)
    split = {}
    rnd = random.Random(360)
    by_date = defaultdict(list)
    for d, f in photos:
        by_date[d].append(f)
    for d, files in by_date.items():
        files = sorted(files); rnd.shuffle(files)
        n = len(files); nt = max(1, round(0.1 * n)); nv = max(1, round(0.2 * n))
        for i, f in enumerate(files):
            split[(d, f)] = "test" if i < nt else "valid" if i < nt + nv else "train"
    cnt = defaultdict(Counter)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:          # JPGs are already compressed
        for key, views in sorted(photos.items()):
            s = split[key]
            for r in views:
                stem = Path(r["view"]).stem
                z.write(SRC / "images" / r["view"], f"{s}/images/{r['view']}")
                z.write(SRC / "labels" / f"{stem}.txt", f"{s}/labels/{stem}.txt")
                cnt[s]["views"] += 1; cnt[s]["open"] += int(r["open"]); cnt[s]["closed"] += int(r["closed"])
            cnt[s]["photos"] += 1
        z.writestr("data.yaml", f"train: train/images\nval: valid/images\ntest: test/images\n"
                                f"nc: {len(NAMES)}\nnames: {NAMES}\n")
        lines = ["AgriEYE 360 - cotton bolls, all 4-direction views (N E S W) of the overall dataset",
                 "Boxes: SAM 3 (text prompts) ; class: R1 Roboflow boll classifier (open / closed; not-boll removed)",
                 "Split by photo (a photo's 4 views stay together), 70/20/10 per visit date; exact copies left out.", "",
                 f"{'split':<7}{'photos':>8}{'views':>8}{'open':>8}{'closed':>8}"]
        for s in ("train", "valid", "test"):
            c = cnt[s]; lines.append(f"{s:<7}{c['photos']:>8}{c['views']:>8}{c['open']:>8}{c['closed']:>8}")
        z.writestr("README.txt", "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nZip: {OUT}  ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
