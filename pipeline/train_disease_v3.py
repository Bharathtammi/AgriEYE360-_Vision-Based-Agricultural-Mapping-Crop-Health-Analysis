"""
Disease classifier v3 = your field leaves + public cotton leaf datasets (healthy / diseased).

  python train_disease_v3.py build     # make disease_autolabel/dataset_cls_v3 (no training)
  python train_disease_v3.py train     # train YOLO11n-cls -> disease_autolabel/models/disease_cls_v3
  python train_disease_v3.py compare   # v2 vs v3 on: field test leaves, outside test leaves, your test images

Field part = exactly the v2 dataset (disease_autolabel/dataset_cls), so the field test is unchanged.
Outside part = external_data/crops, split 80/20 per class folder (fixed seed); file names start with "ext_".
v2 (models/disease_cls) is never overwritten.
"""
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pipeline as P

ROOT = Path(__file__).resolve().parent
FIELD = P.WORK / "dataset_cls"
EXT = ROOT / "external_data" / "crops"
DATA = P.WORK / "dataset_cls_v3"
RUN = P.MODELS / "disease_cls_v3"
V2 = P.CLS_MODEL
V3 = RUN / "weights" / "best.pt"
USER_TESTS = ROOT / "sample_results" / "user_tests"


def build():
    shutil.rmtree(DATA, ignore_errors=True)
    n = defaultdict(int)
    for split in ("train", "val"):
        for cls in ("healthy", "diseased"):
            dst = DATA / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for f in (FIELD / split / cls).glob("*.png"):
                shutil.copy2(f, dst / f.name)
                n[("field", split, cls)] += 1
    rng = random.Random(0)
    for cls in ("healthy", "diseased"):
        groups = defaultdict(list)
        for f in sorted((EXT / cls).glob("*.png")):
            groups[f.name.split("_")[1] + f.name.split("_")[2]].append(f)   # dataset + class folder
        for g, fs in sorted(groups.items()):
            rng.shuffle(fs)
            k = round(len(fs) * 0.2)
            for i, f in enumerate(fs):
                split = "val" if i < k else "train"
                shutil.copy2(f, DATA / split / cls / f.name)
                n[("outside", split, cls)] += 1
    for k in sorted(n):
        print(f"{k[0]:8s} {k[1]:5s} {k[2]:8s} {n[k]:5d}")
    print(f"\nDataset: {DATA}")


def train():
    from ultralytics import YOLO
    model = YOLO("yolo11n-cls.pt")
    model.train(data=str(DATA), epochs=40, imgsz=P.CLS_IMGSZ, batch=32, device="cpu", patience=10,
                project=str(P.MODELS), name="disease_cls_v3", exist_ok=True, save_period=1, plots=True,
                verbose=False)
    print(f"\nModel v3: {V3}")


def evaluate(model, files):
    from PIL import Image
    ims = [Image.open(f).convert("RGB") for f, _ in files]
    out = []
    for i in range(0, len(ims), 64):
        out += P.classify(model, ims[i:i + 64])
    tp = sum(t == "diseased" and l == "diseased" for (_, t), (l, _) in zip(files, out))
    fn = sum(t == "diseased" and l != "diseased" for (_, t), (l, _) in zip(files, out))
    fp = sum(t == "healthy" and l == "diseased" for (_, t), (l, _) in zip(files, out))
    tn = sum(t == "healthy" and l == "healthy" for (_, t), (l, _) in zip(files, out))
    acc = (tp + tn) / max(1, len(files))
    return dict(n=len(files), acc=acc, rec=tp / max(1, tp + fn), prec=tp / max(1, tp + fp),
                hrec=tn / max(1, tn + fp)), out


def compare():
    from ultralytics import YOLO
    sets = {"field test": [], "outside test": []}
    for cls in ("healthy", "diseased"):
        for f in (DATA / "val" / cls).glob("*.png"):
            sets["outside test" if f.name.startswith("ext_") else "field test"].append((f, cls))
    models = {"v2 (field only)": YOLO(str(V2))}
    if V3.exists():
        models["v3 (field + outside)"] = YOLO(str(V3))
    print(f"\n{'model':22s} {'test set':14s} {'leaves':>6s} {'accuracy':>9s} {'dis.recall':>10s} "
          f"{'dis.prec':>8s} {'healthy ok':>10s}")
    for mn, m in models.items():
        for sn, fs in sets.items():
            r, _ = evaluate(m, fs)
            print(f"{mn:22s} {sn:14s} {r['n']:6d} {100 * r['acc']:8.1f}% {100 * r['rec']:9.1f}% "
                  f"{100 * r['prec']:7.1f}% {100 * r['hrec']:9.1f}%")
    tests = sorted(p for p in USER_TESTS.glob("*.jpg"))
    if tests:
        from PIL import Image
        print("\nYour test images (whole image):")
        for t in tests:
            im = Image.open(t).convert("RGB")
            res = {mn: P.classify(m, [im])[0] for mn, m in models.items()}
            print(f"  {t.name:30s} " + "   ".join(f"{mn.split()[0]}: {l} ({p:.2f})" for mn, (l, p) in res.items()))


if __name__ == "__main__":
    {"build": build, "train": train, "compare": compare}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
