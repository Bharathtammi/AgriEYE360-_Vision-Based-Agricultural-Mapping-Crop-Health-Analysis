"""
PURE Qwen segmentation (Qwen2.5-VL-3B only - no SAM, no YOLO, no training, no labels).

Qwen has no mask output; it answers in text. So Qwen is asked to write the OUTLINE of every object as a
polygon (list of x,y points, in pixels of the picture it saw). The polygons are scaled back to the original
picture and filled in = the mask. If Qwen gives only a box for an object, the box is used as its mask.
Expect coarse outlines (a handful of points per object) - this is what Qwen itself can do.

  python qwen_segment.py <image or folder> [more ...]
         [--object "open cotton boll"] [--object "closed cotton boll"] [--object "cotton leaf"]
         [--max-pixels 1024] [--n 5]

Default objects: open cotton boll, closed cotton boll.
--max-pixels = size Qwen sees, in 28x28 patches (1024 patches ~ 900x900 px). Bigger = sees small bolls, slower.

Output in sample_results/qwen_segment/:
  <name>_overlay.jpg   filled Qwen polygons, colour per object
  <name>_mask.png      label image: 0 = background, 1 = first object, 2 = second ...
  <name>_objects.csv   object, kind (polygon/box), points, bounding box, mask area (original pixels)
  <name>_qwen.txt      Qwen's raw answer (to check what it said)

Needs (install once, in cmd - first run downloads ~7.5 GB to C:\\Users\\bhara\\.cache\\huggingface):
  "C:\\Users\\bhara\\Downloads\\.venv\\Scripts\\python.exe" -m pip install "transformers>=4.49,<5" accelerate
CPU only: expect a few minutes per picture. Close browsers first (~8-10 GB RAM).
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLS = [(255, 60, 60), (60, 160, 255), (60, 220, 90), (255, 200, 0), (200, 80, 255)]
PROMPT = ('Segment every {obj} in the image, including small and partly hidden ones. '
          'For each one, trace its outline as a polygon of 8 to 16 points. '
          'Output JSON only, as a list: [{{"label": "{obj}", "polygon": [[x1, y1], [x2, y2], ...]}}, ...]. '
          'If there is none, output [].')


def load_qwen(max_pixels):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    torch.set_num_threads(8)
    print(f"Loading {MODEL} (CPU, bfloat16) ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=max_pixels * 28 * 28)
    return model, proc


NUM = r"[-\d.]+"
PT = rf"\[\s*{NUM}\s*,\s*{NUM}\s*\]"


def parse_shapes(text):
    """Every object Qwen wrote: ('polygon', [[x,y],...]) or ('box', [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]).
    Works on cut-off JSON too (reads object by object)."""
    shapes = []
    for m in re.finditer(rf'"(polygon|points|bbox_2d)"\s*:\s*\[((?:\s*{PT}\s*,?)+|\s*{NUM}(?:\s*,\s*{NUM})+\s*)\]', text):
        key, body = m.group(1), m.group(2)
        if "[" in body:
            pts = [[float(a), float(b)] for a, b in re.findall(rf"\[\s*({NUM})\s*,\s*({NUM})\s*\]", body)]
        else:
            v = [float(a) for a in re.findall(NUM, body)]
            pts = [[v[i], v[i + 1]] for i in range(0, len(v) - 1, 2)]
        if key == "bbox_2d" or len(pts) == 2:
            if len(pts) >= 2:
                (x1, y1), (x2, y2) = pts[0], pts[1]
                shapes.append(("box", [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
        elif len(pts) >= 3:
            shapes.append(("polygon", pts))
    return shapes


def ask_qwen(model, proc, img, obj, max_new=3000):
    import torch
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(obj=obj)}]}]
    chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[chat], images=[img], return_tensors="pt")
    # Qwen2.5-VL answers in pixels of the RESIZED picture it saw: grid (t, h, w) x 14 px patches
    _, gh, gw = inp["image_grid_thw"][0].tolist()
    with torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05)
    text = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    W, H = img.size
    sx, sy = W / (gw * 14), H / (gh * 14)
    shapes = []
    for kind, pts in parse_shapes(text):
        p = [(max(0.0, min(W, x * sx)), max(0.0, min(H, y * sy))) for x, y in pts]
        xs, ys = [q[0] for q in p], [q[1] for q in p]
        if max(xs) - min(xs) < 4 or max(ys) - min(ys) < 4:
            continue
        if any(kind == k and np.allclose(p[:3], q[:3], atol=3) for k, q in shapes):  # Qwen repeats itself
            continue
        shapes.append((kind, p))
    return shapes, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--object", action="append", help="what to segment (repeat), default open + closed cotton boll")
    ap.add_argument("--max-pixels", type=int, default=1024, help="Qwen picture size in 28x28 patches")
    ap.add_argument("--n", type=int, default=0, help="only the first N pictures")
    ap.add_argument("--out", default=str(ROOT / "sample_results" / "qwen_segment"))
    args = ap.parse_args()
    objs = args.object or ["open cotton boll", "closed cotton boll"]

    files = []
    for i in args.inputs:
        p = Path(i)
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    if args.n:
        files = files[:args.n]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, proc = load_qwen(args.max_pixels)

    for f in files:
        t0 = time.time()
        img = Image.open(f).convert("RGB")
        W, H = img.size
        ov = img.convert("RGBA"); layer = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
        label = Image.new("L", img.size, 0); dl = ImageDraw.Draw(label)
        rows, raw = [], []
        for k, obj in enumerate(objs):
            shapes, text = ask_qwen(model, proc, img, obj)
            raw.append(f"=== {obj} ===\n{text}\n")
            c = COLS[k % len(COLS)]
            for kind, p in shapes:
                one = Image.new("L", img.size, 0); ImageDraw.Draw(one).polygon(p, fill=1)
                area = int(np.asarray(one).sum())
                dl.polygon(p, fill=k + 1)
                d.polygon(p, fill=c + (110,), outline=c + (255,), width=2)
                xs, ys = [q[0] for q in p], [q[1] for q in p]
                rows.append([obj, kind, json.dumps([[round(x), round(y)] for x, y in p]),
                             round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)), area])
        Image.alpha_composite(ov, layer).convert("RGB").save(out / f"{f.stem}_overlay.jpg", quality=90)
        label.save(out / f"{f.stem}_mask.png")
        (out / f"{f.stem}_qwen.txt").write_text("\n".join(raw), encoding="utf-8")
        with open(out / f"{f.stem}_objects.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["object", "kind", "points", "x1", "y1", "x2", "y2", "mask_px"]); w.writerows(rows)
        counts = ", ".join(f"{o}: {sum(r[0] == o for r in rows)}" for o in objs)
        print(f"{f.name}: {counts}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"Saved in {out}")


if __name__ == "__main__":
    main()
