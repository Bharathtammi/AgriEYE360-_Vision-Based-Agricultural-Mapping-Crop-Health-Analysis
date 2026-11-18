"""
Write the method-A labels of ALL 4-direction views as TWO SEPARATE label sets: cotton bolls and leaves.

  python export_pov_all_labels.py leaves     # leaves only (ready after label_all_povs.py)
  python export_pov_all_labels.py bolls      # bolls only (needs boll_r1.csv from label_all_bolls_r1.py)
  python export_pov_all_labels.py both

Output (YOLO format, images are hard links to agrieye_output/pov_all, so no extra disk space):
  agrieye_output/pov_all_labels/leaves_dataset/
      images/<view>.jpg  labels/<view>.txt   polygons: 0 healthy_leaf, 1 diseased_leaf, 2 small_leaf (too small to judge)
      data.yaml  views.csv  summary.txt  previews/<date>/  gallery.html
  agrieye_output/pov_all_labels/bolls_dataset/
      images/<view>.jpg  labels/<view>.txt   boxes: 0 open_boll, 1 closed_boll   (R1's "not_boll" boxes are dropped)
      data.yaml  views.csv  summary.txt  previews/<date>/  gallery.html

No train / valid / test split is made here. views.csv gives each view's photo group (an exact copy shares the group
of its original), so a later split can keep a photo's 4 views and its copies together.
"""
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
LAB = ROOT / "agrieye_output" / "pov_all_labels"
LEAF_NAMES = ["healthy_leaf", "diseased_leaf", "small_leaf"]
BOLL_NAMES = ["open_boll", "closed_boll"]
LEAF_COL = [(40, 200, 60), (230, 40, 40), (160, 160, 160)]
BOLL_COL = [(0, 230, 255), (255, 0, 200)]


def link(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def records():
    recs = [json.loads(l) for l in (LAB / "sam3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seen, out = set(), []
    for r in recs:                                   # one line per view (guards against a doubled run)
        k = (r["file"], r["direction"])
        if k not in seen:
            seen.add(k); out.append(r)
    return out


def leaf_class(d):
    return {"healthy": 0, "diseased": 1}.get(d["label"], 2)


def gallery(dst, title, rows, legend):
    dates = sorted({r["date"] for r in rows})
    cards = "\n".join(
        f'<figure data-date="{r["date"]}"><img loading="lazy" src="previews/{r["date"]}/{r["name"]}.jpg">'
        f'<figcaption>{r["name"]}<br>{r["text"]}</figcaption></figure>' for r in rows)
    opts = "".join(f'<option>{d}</option>' for d in dates)
    (dst / "gallery.html").write_text(f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:Segoe UI,Arial;background:#111;color:#eee;margin:16px}}
#g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}}
figure{{margin:0;background:#222;padding:6px;border-radius:6px}}img{{width:100%;border-radius:4px}}
figcaption{{font-size:13px;color:#bbb;margin-top:4px}}select{{font-size:15px;padding:4px}}</style></head><body>
<h2>{title}</h2><p>{legend}</p>
<p>Date: <select id="s"><option value="">all</option>{opts}</select> <span id="n"></span></p><div id="g">{cards}</div>
<script>const s=document.getElementById('s'),n=document.getElementById('n');
function f(){{let k=0;document.querySelectorAll('figure').forEach(e=>{{const v=!s.value||e.dataset.date==s.value;e.style.display=v?'':'none';k+=v}});n.textContent=k+' views'}}
s.onchange=f;f();</script></body></html>""", encoding="utf-8")


def prepare(dst, names):
    for sub in ("images", "labels"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    (dst / "data.yaml").write_text(
        f"path: {dst.as_posix()}\ntrain: images\nval: images\n"
        f"# no split yet: see views.csv (group column) before training\n"
        f"nc: {len(names)}\nnames: {names}\n", encoding="utf-8")


def views_csv(dst, rows):
    with open(dst / "views.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def export_leaves(recs):
    dst = LAB / "leaves_dataset"; prepare(dst, LEAF_NAMES)
    vrows, grows, per = [], [], defaultdict(Counter)
    for k, r in enumerate(recs, 1):
        view = Path(r["view"]); name = view.stem; W, H = r["width"], r["height"]
        link(view, dst / "images" / view.name)
        lines, cnt = [], Counter()
        for d in r["leaves"]:
            c = leaf_class(d); cnt[c] += 1
            pts = " ".join(f"{min(max(x / W, 0), 1):.5f} {min(max(y / H, 0), 1):.5f}" for x, y in d["poly"])
            lines.append(f"{c} {pts}")
        (dst / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        per[r["date"]].update(cnt); per[r["date"]]["views"] += 1
        p = dst / "previews" / r["date"] / f"{name}.jpg"
        if not p.exists():
            im = Image.open(view).convert("RGB"); dr = ImageDraw.Draw(im, "RGBA")
            for d in r["leaves"]:
                col = LEAF_COL[leaf_class(d)]
                dr.polygon([tuple(q) for q in d["poly"]], fill=col + (70,), outline=col + (255,), width=3)
            im.thumbnail((900, 900)); p.parent.mkdir(parents=True, exist_ok=True); im.save(p, quality=82)
        vrows.append(dict(view=view.name, file=r["file"], date=r["date"], direction=r["direction"],
                          group=r["copy_of"] or r["file"], copy_of=r["copy_of"],
                          healthy=cnt[0], diseased=cnt[1], small=cnt[2]))
        grows.append(dict(date=r["date"], name=name,
                          text=f"healthy {cnt[0]}, diseased {cnt[1]}, small {cnt[2]}"))
        if k % 200 == 0:
            print(f"  leaves {k}/{len(recs)}", flush=True)
    views_csv(dst, vrows)
    gallery(dst, "AgriEYE 360 - leaf labels (SAM 3 outline + L2 healthy / diseased)", grows,
            "green = healthy leaf, red = diseased leaf, grey = leaf too small to judge (&lt; 40 px)")
    lines = ["LEAF LABELS - all 4-direction views (SAM 3 outlines, L2 disease classifier v2)", "",
             f"{'date':<12}{'views':>7}{'healthy':>9}{'diseased':>10}{'small':>8}{'% diseased':>12}"]
    T = Counter()
    for dt, c in sorted(per.items()):
        T.update(c)
        lines.append(f"{dt:<12}{c['views']:>7}{c[0]:>9}{c[1]:>10}{c[2]:>8}{100 * c[1] / max(c[0] + c[1], 1):>11.1f}%")
    lines.append(f"{'TOTAL':<12}{T['views']:>7}{T[0]:>9}{T[1]:>10}{T[2]:>8}{100 * T[1] / max(T[0] + T[1], 1):>11.1f}%")
    lines.append("\n% diseased = diseased / (healthy + diseased); small leaves are not judged.")
    (dst / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines) + f"\nFiles: {dst}\n")


def export_bolls(recs, draft=False):
    f = LAB / "boll_r1.csv"
    if draft:                                   # SAM 3's own guess: open / closed / "flower" (= mostly red closed bolls)
        sam = ["open_boll", "closed_boll", "closed_boll"]
        r1 = {(r["file"], r["direction"], i): sam[d["cls"]] for r in recs if not r["copy_of"]
              for i, d in enumerate(r["bolls"])}
        dst = LAB / "bolls_dataset_DRAFT_sam3_only"
    else:
        if not f.exists():
            print(f"Bolls: {f} not found yet - run label_all_bolls_r1.py first (it needs your key),\n"
                  f"or see a draft now:  python export_pov_all_labels.py bolls-draft")
            return
        r1 = {(r["file"], r["direction"], int(r["k"])): r["pred"] for r in csv.DictReader(open(f, encoding="utf-8"))}
        dst = LAB / "bolls_dataset"
    prepare(dst, BOLL_NAMES)
    vrows, grows, per, missing = [], [], defaultdict(Counter), 0
    for k, r in enumerate(recs, 1):
        view = Path(r["view"]); name = view.stem; W, H = r["width"], r["height"]
        src_file = r["copy_of"] or r["file"]                      # copies use their original's R1 answers
        link(view, dst / "images" / view.name)
        lines, cnt, kept = [], Counter(), []
        for i, d in enumerate(r["bolls"]):
            pred = r1.get((src_file, r["direction"], i))
            if pred is None:
                missing += 1; continue
            if pred == "not_boll":
                cnt["not_boll"] += 1; continue
            if pred not in BOLL_NAMES:                       # R1 gave no answer ("?"): left out
                cnt["no_answer"] += 1; continue
            c = BOLL_NAMES.index(pred); cnt[c] += 1; kept.append((c, d["box"]))
            x1, y1, x2, y2 = d["box"]
            lines.append(f"{c} {(x1 + x2) / 2 / W:.5f} {(y1 + y2) / 2 / H:.5f} {(x2 - x1) / W:.5f} {(y2 - y1) / H:.5f}")
        (dst / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        per[r["date"]].update(cnt); per[r["date"]]["views"] += 1
        p = dst / "previews" / r["date"] / f"{name}.jpg"
        im = Image.open(view).convert("RGB"); dr = ImageDraw.Draw(im)
        for c, b in sorted(kept, key=lambda t: -(t[1][2] - t[1][0]) * (t[1][3] - t[1][1])):
            dr.rectangle([b[0] - 3, b[1] - 3, b[2] + 3, b[3] + 3], outline=(0, 0, 0), width=8)
            dr.rectangle(b, outline=BOLL_COL[c], width=5)
        im.thumbnail((900, 900)); p.parent.mkdir(parents=True, exist_ok=True); im.save(p, quality=82)
        vrows.append(dict(view=view.name, file=r["file"], date=r["date"], direction=r["direction"],
                          group=src_file, copy_of=r["copy_of"], open=cnt[0], closed=cnt[1], not_boll=cnt["not_boll"]))
        grows.append(dict(date=r["date"], name=name, text=f"open {cnt[0]}, closed {cnt[1]}"))
        if k % 200 == 0:
            print(f"  bolls {k}/{len(recs)}", flush=True)
    views_csv(dst, vrows)
    gallery(dst, "AgriEYE 360 - cotton boll labels " + ("DRAFT (SAM 3 only, before R1)" if draft else "(SAM 3 box + R1 open / closed)"), grows,
            "cyan = open boll, magenta = closed boll (boxes R1 called not a boll are removed)")
    lines = ["COTTON BOLL LABELS - all 4-direction views " + ("DRAFT: SAM 3 boxes and SAM 3 open/closed guess, NOT checked by R1" if draft else "(SAM 3 boxes, R1 Roboflow classifier)"), "",
             f"{'date':<12}{'views':>7}{'open':>8}{'closed':>8}{'removed (not boll)':>20}"]
    T = Counter()
    for dt, c in sorted(per.items()):
        T.update(c)
        lines.append(f"{dt:<12}{c['views']:>7}{c[0]:>8}{c[1]:>8}{c['not_boll']:>20}")
    lines.append(f"{'TOTAL':<12}{T['views']:>7}{T[0]:>8}{T[1]:>8}{T['not_boll']:>20}")
    if missing:
        lines.append(f"\n{missing} boxes not judged by R1 yet (left out) - run label_all_bolls_r1.py again, then this.")
    (dst / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines) + f"\nFiles: {dst}\n")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "both"
    recs = records()
    if part in ("leaves", "both"):
        export_leaves(recs)
    if part in ("bolls", "both"):
        export_bolls(recs)
    if part == "bolls-draft":
        export_bolls(recs, draft=True)
