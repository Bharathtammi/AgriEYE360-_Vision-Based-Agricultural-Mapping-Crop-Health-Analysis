"""
Step 2: turn your hand-sorted leaf crops into YOLO training datasets.

Before running, move every crop from
    disease_autolabel/crops/unsorted/
into one of
    disease_autolabel/crops/healthy/
    disease_autolabel/crops/diseased/
(Delete crops that are not leaves, e.g. soil or sky.)

Outputs:
  disease_autolabel/dataset_cls/   folder dataset for  yolo classify  (healthy / diseased)
  disease_autolabel/dataset_seg/   YOLO-seg dataset for  yolo segment  (healthy_leaf / diseased_leaf)
                                   includes only source images whose leaves are all sorted,
                                   so no unlabeled leaf turns into "background"

The train/val split is by source photo, not by crop: leaves from one photo
never end up in both train and val, which would inflate
validation accuracy.

Usage:
  python build_disease_dataset.py
  python build_disease_dataset.py --val 0.2 --seed 0
"""

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = ["healthy", "diseased"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=ROOT / "disease_autolabel")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-healthy", type=int, default=2500,
                    help="classifier only: use at most this many healthy crops (random) so the "
                         "classes stay roughly balanced; 0 = use all")
    args = ap.parse_args()

    crops_dir = args.work / "crops"
    label_of = {}
    for cls in CLASSES:
        for p in (crops_dir / cls).glob("*.png"):
            label_of[p.name] = cls
    unsorted = {p.name for p in (crops_dir / "unsorted").glob("*.png")}
    print(f"Sorted crops: " + ", ".join(f"{c}={sum(v == c for v in label_of.values())}" for c in CLASSES)
          + f"  | still unsorted: {len(unsorted)}")
    if not label_of:
        raise SystemExit("No sorted crops found. Move crops into crops/healthy and crops/diseased first.")

    # Read manifest, group crop records by source image.
    by_source = defaultdict(list)
    for line in (args.work / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("crop"):
                by_source[r["source"]].append(r)

    # Split by source photo: all leaves from one photo go to the same side.
    # Only photos that contributed sorted crops count, so val is never empty.
    groups = sorted(s for s, recs in by_source.items() if any(r["crop"] in label_of for r in recs))
    # Stratify by visit date (from image_YYYYMMDD_ in the file name) so every visit with 2+
    # photos has at least one photo in val - otherwise a whole visit can go untested.
    by_date = defaultdict(list)
    for s in groups:
        m = re.search(r"(20\d{6})_\d{6}", Path(s).name)
        by_date[m.group(1) if m else "other"].append(s)
    rng = random.Random(args.seed)
    val_groups = set()
    for date in sorted(by_date):
        g = by_date[date]
        rng.shuffle(g)
        k = max(1, round(len(g) * args.val)) if len(g) > 1 else 0
        val_groups.update(g[:k])
    print("Val photos per visit: " + ", ".join(
        f"{d}={sum(s in val_groups for s in g)}/{len(g)}" for d, g in sorted(by_date.items())))
    split_of = lambda src: "val" if src in val_groups else "train"

    # ---- classification dataset ----
    # Balance: keep a random subset of healthy crops (segmentation below still uses all).
    healthy = sorted(n for n, c in label_of.items() if c == "healthy")
    cls_drop = set()
    if args.max_healthy and len(healthy) > args.max_healthy:
        random.Random(args.seed).shuffle(healthy)
        cls_drop = set(healthy[args.max_healthy:])
        print(f"Classifier: using {args.max_healthy} of {len(healthy)} healthy crops (balance)")
    cls_root = args.work / "dataset_cls"
    shutil.rmtree(cls_root, ignore_errors=True)
    counts = defaultdict(int)
    for src, recs in by_source.items():
        split = split_of(src)
        for r in recs:
            cls = label_of.get(r["crop"])
            if cls is None or r["crop"] in cls_drop:
                continue
            dst = cls_root / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(crops_dir / cls / r["crop"], dst / r["crop"])
            counts[(split, cls)] += 1
    print("Classification dataset:", dict(counts))

    # ---- segmentation dataset (only fully-labeled images) ----
    seg_root = args.work / "dataset_seg"
    shutil.rmtree(seg_root, ignore_errors=True)
    n_imgs = defaultdict(int)
    skipped = 0
    for src, recs in by_source.items():
        # Crops that were deleted (not a leaf) are simply dropped; any crop
        # still in unsorted/ means the image is not fully labeled yet.
        if any(r["crop"] in unsorted for r in recs):
            skipped += 1
            continue
        lines = []
        for r in recs:
            cls = label_of.get(r["crop"])
            if cls is None:
                continue
            pts = " ".join(f"{x:.5f} {y:.5f}" for x, y in r["polygon"])
            lines.append(f"{CLASSES.index(cls)} {pts}")
        if not lines:
            continue
        split = split_of(src)
        stem = Path(r["crop"]).name.rsplit("__", 1)[0]
        (seg_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (seg_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        srcp = Path(src)
        shutil.copy2(srcp, seg_root / "images" / split / f"{stem}{srcp.suffix.lower()}")
        (seg_root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n_imgs[split] += 1

    (seg_root).mkdir(parents=True, exist_ok=True)
    (seg_root / "data.yaml").write_text(
        f"path: {seg_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        "names: ['healthy_leaf', 'diseased_leaf']\n"
    )
    print(f"Segmentation dataset: {dict(n_imgs)} images ({skipped} skipped: not fully sorted)")

    print("\nTrain with, e.g.:")
    print(f'  yolo classify train data="{cls_root}" model=yolo11n-cls.pt epochs=50 imgsz=224')
    print(f'  yolo segment  train data="{seg_root / "data.yaml"}" model=yolo11n-seg.pt epochs=100 imgsz=640')


if __name__ == "__main__":
    main()
