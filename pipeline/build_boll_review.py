"""
Step 2a: OPEN / CLOSED REVIEW page - you check the boll labels the machines are least sure about.

  python build_boll_review.py            # builds the page (no training, no key)
  python build_boll_review.py apply      # after you saved your answers: shows what changed (labels are NOT changed yet)

Which bolls: only views used for TRAINING the local detector (data_big/sources.csv, automatic labels), only bolls
>= 24 px, ranked by doubt:
  R1 confidence < 0.75 (+2) | SAM 3 and R1 disagree on open/closed (+1) | R1 says closed but the crop is very white (+1)
  | R1 says open with confidence < 0.9 (+1)        then lowest R1 confidence first; spread over all dates. Top N.
Output: agrieye_output/boll_review/open_closed/review.html  (+ crops/, items.json)
Your answers: the page's 'Download answers' button saves boll_review_answers.csv (normally in your Downloads folder).
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
LAB = ROOT / "agrieye_output" / "pov_all_labels"
SRC = ROOT / "agrieye_output" / "boll_detector" / "data_big" / "sources.csv"
OUT = ROOT / "agrieye_output" / "boll_review" / "open_closed"
N = 500
SAM = ["open_boll", "closed_boll", "closed_boll"]       # SAM 3 classes: open, closed, flower (mostly red closed bolls)


def candidates():
    train = {r["view"] for r in csv.DictReader(open(SRC, encoding="utf-8")) if r["labels"].startswith("automatic")}
    r1 = {(r["file"], r["direction"], int(r["k"])): (r["pred"], float(r["conf"]))
          for r in csv.DictReader(open(LAB / "boll_r1.csv", encoding="utf-8"))}
    out = []
    for line in open(LAB / "sam3.jsonl", encoding="utf-8"):
        r = json.loads(line)
        view = Path(r["view"])
        if r["copy_of"] or view.name not in train:
            continue
        for k, b in enumerate(r["bolls"]):
            p = r1.get((r["file"], r["direction"], k))
            if not p or p[0] not in ("open_boll", "closed_boll"):
                continue
            x1, y1, x2, y2 = b["box"]
            if max(x2 - x1, y2 - y1) < 24:
                continue
            pts, why = 0, []
            if p[1] < 0.75:
                pts += 2; why.append(f"R1 unsure ({100 * p[1]:.0f}%)")
            if SAM[b["cls"]] != p[0]:
                pts += 1; why.append("SAM 3 and R1 disagree")
            if p[0] == "closed_boll" and b["white"] >= 0.4:
                pts += 1; why.append("very white for a closed boll")
            if p[0] == "open_boll" and p[1] < 0.9:
                pts += 1; why.append("open, not sure")
            if pts:
                out.append(dict(view=str(view), name=view.name, date=r["date"], file=r["file"], direction=r["direction"],
                                k=k, box=b["box"], label=p[0], conf=p[1], pts=pts, why="; ".join(why)))
    return out


def secs(name):
    t = Path(name).stem.split("_")[2]
    return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


_cache = {}


def thumb(x):
    """16x16 grey picture of the boll's surroundings - to spot the same boll in a near-identical next photo."""
    import numpy as np
    if x["view"] not in _cache:
        if len(_cache) > 64:
            _cache.clear()
        _cache[x["view"]] = Image.open(x["view"]).convert("L")
    x1, y1, x2, y2 = x["box"]; s = max(x2 - x1, y2 - y1) * 1.5; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return np.asarray(_cache[x["view"]].crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s))).resize((16, 16)), np.float32)


def same_boll(a, b):
    import eval_roboflow_detect as E
    if a["date"] != b["date"] or a["direction"] != b["direction"] or abs(secs(a["name"]) - secs(b["name"])) > 60:
        return False
    if E.iou(a["box"], b["box"]) < 0.5:
        return False
    return float(abs(thumb(a) - thumb(b)).mean()) < 8


def pick(c):
    """Most doubtful first, round-robin over dates so every visit is checked. The same boll seen in near-identical
    consecutive photos is shown once; the others are kept as its 'twins' and get the same answer."""
    by = defaultdict(list)
    for x in sorted(c, key=lambda x: (-x["pts"], x["conf"])):
        by[x["date"]].append(x)
    out = []
    while len(out) < N and any(by.values()):
        for d in sorted(by):
            while by[d] and len(out) < N:
                x = by[d].pop(0)
                twin = next((o for o in out if same_boll(o, x)), None)
                if twin:
                    twin["twins"].append(x); continue
                x["twins"] = []; out.append(x); break
    for o in out:                                        # twins further down the list (not reached yet)
        for d in by:
            rest = []
            for x in by[d]:
                (o["twins"].append(x) if same_boll(o, x) else rest.append(x))
            by[d] = rest
    return sorted(out, key=lambda x: (-x["pts"], x["conf"]))


def crop(x, path):
    im = Image.open(x["view"]).convert("RGB")
    x1, y1, x2, y2 = x["box"]
    s = max(120, 2.6 * max(x2 - x1, y2 - y1)); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    L, T = max(0, cx - s / 2), max(0, cy - s / 2)
    R, B = min(im.width, L + s), min(im.height, T + s)
    c = im.crop((int(L), int(T), int(R), int(B)))
    sc = 220 / max(c.size)
    c = c.resize((max(1, round(c.width * sc)), max(1, round(c.height * sc))), Image.LANCZOS)
    d = ImageDraw.Draw(c)
    b = [(x1 - L) * sc - 3, (y1 - T) * sc - 3, (x2 - L) * sc + 3, (y2 - T) * sc + 3]
    d.rectangle(b, outline=(0, 0, 0), width=4); d.rectangle(b, outline=(255, 230, 0), width=2)
    c.save(path, quality=88)


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Boll open / closed review</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f4f2;color:#222}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:10px 16px;z-index:5}
h1{font-size:18px;margin:0 0 4px}.help{font-size:13px;color:#555}
.bar{height:6px;background:#ddd;border-radius:3px;margin-top:6px}.bar div{height:6px;background:#2a7;border-radius:3px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(236px,1fr));gap:12px;padding:16px}
.card{background:#fff;border:3px solid #ddd;border-radius:8px;padding:6px;font-size:12px}
.card.hover{outline:2px dashed #888}.card.changed{border-color:#e08a00}.card.kept{border-color:#2a7}
.card img{width:220px;height:220px;object-fit:contain;background:#111;display:block;margin:auto}
.meta{margin:4px 0;color:#555;min-height:30px}
.btns{display:flex;gap:4px}.btns button{flex:1;padding:6px 0;border:1px solid #999;border-radius:5px;background:#fafafa;cursor:pointer;font-size:12px}
.btns button.on{color:#fff;font-weight:600}
button.o.on{background:#0aa;border-color:#0aa}button.c.on{background:#b0179a;border-color:#b0179a}
button.n.on{background:#555;border-color:#555}button.u.on{background:#c90;border-color:#c90}
.nav{padding:0 16px 30px;display:flex;gap:10px;align-items:center}
.nav button,.top button{padding:8px 14px;border-radius:6px;border:1px solid #888;background:#fff;cursor:pointer}
.nav .done{background:#2a7;color:#fff;border-color:#2a7;font-weight:600}
</style></head><body>
<header><h1>Boll review: is the yellow-boxed boll OPEN or CLOSED?</h1>
<div class="help">The current label is already selected. <b>Only click when it is wrong.</b>
Keys (mouse over a card): <b>1</b> open &nbsp;<b>2</b> closed &nbsp;<b>3</b> not a boll &nbsp;<b>4</b> unsure.
When a page looks right, press <b>Page checked &rarr;</b>. Your work is saved in this browser automatically;
press <b>Download answers</b> at the end (or any time).</div>
<div class="top" style="margin-top:6px"><span id="stat"></span> &nbsp;
<button onclick="download()">Download answers</button></div><div class="bar"><div id="prog"></div></div></header>
<div class="grid" id="grid"></div>
<div class="nav"><button onclick="go(-1)">&larr; Previous page</button><span id="pg"></span>
<button class="done" onclick="checked()">Page checked &rarr;</button></div>
<script>
const ITEMS=__ITEMS__, PER=40, KEY="boll_review_v1";
const NAMES={open_boll:"open",closed_boll:"closed",not_boll:"not a boll",unsure:"unsure"};
let st=JSON.parse(localStorage.getItem(KEY)||'{"ans":{},"done":[],"page":0}'), hover=null;
function save(){localStorage.setItem(KEY,JSON.stringify(st));stat();}
function ans(i){return st.ans[i]||ITEMS[i].label}
function stat(){const d=new Set(st.done);let n=0,ch=0;ITEMS.forEach((x,i)=>{const p=Math.floor(i/PER);if(d.has(p)||st.ans[i]){n++;if(ans(i)!==x.label)ch++;}});
document.getElementById('stat').textContent=`checked ${n} / ${ITEMS.length}   changed ${ch}`;
document.getElementById('prog').style.width=(100*n/ITEMS.length)+'%';}
function set(i,v){st.ans[i]=v;save();draw();}
function draw(){const g=document.getElementById('grid');g.innerHTML='';const p=st.page,d=new Set(st.done);
const pages=Math.ceil(ITEMS.length/PER);document.getElementById('pg').textContent=`page ${p+1} / ${pages}`;
for(let i=p*PER;i<Math.min(ITEMS.length,(p+1)*PER);i++){const x=ITEMS[i],a=ans(i);
const c=document.createElement('div');c.className='card'+(a!==x.label?' changed':(d.has(p)?' kept':''));
c.onmouseenter=()=>{hover=i;c.classList.add('hover')};c.onmouseleave=()=>{hover=null;c.classList.remove('hover')};
c.innerHTML=`<img src="crops/${i}.jpg" loading="lazy"><div class="meta">#${i+1} &middot; now <b>${NAMES[x.label]}</b> &middot; ${x.why}${x.twins.length?` &middot; <i>+${x.twins.length} same boll in next photos</i>`:''}</div>
<div class="btns">${[['open_boll','o','1 open'],['closed_boll','c','2 closed'],['not_boll','n','3 not'],['unsure','u','4 ?']]
.map(([v,k,t])=>`<button class="${k}${a===v?' on':''}" onclick="set(${i},'${v}')">${t}</button>`).join('')}</div>`;g.appendChild(c);}
window.scrollTo(0,0);}
function go(s){const pages=Math.ceil(ITEMS.length/PER);st.page=Math.max(0,Math.min(pages-1,st.page+s));save();draw();}
function checked(){if(!st.done.includes(st.page))st.done.push(st.page);save();
if(st.page<Math.ceil(ITEMS.length/PER)-1)go(1);else{draw();alert('All pages checked. Press Download answers.');}}
document.addEventListener('keydown',e=>{if(hover===null)return;const m={'1':'open_boll','2':'closed_boll','3':'not_boll','4':'unsure'}[e.key];if(m)set(hover,m);});
function download(){const d=new Set(st.done);const rows=[['id','view','file','direction','k','old','new','why']];
ITEMS.forEach((x,i)=>{if(d.has(Math.floor(i/PER))||st.ans[i]){rows.push([i,x.name,x.file,x.direction,x.k,x.label,ans(i),x.why]);
x.twins.forEach(t=>rows.push([i,t[0],t[1],t[2],t[3],t[4],ans(i),'same boll as #'+(i+1)]));}});
const csv=rows.map(r=>r.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(',')).join('\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='boll_review_answers.csv';a.click();}
draw();stat();
</script></body></html>"""


def build():
    c = candidates()
    items = pick(c)
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    for i, x in enumerate(items):
        crop(x, OUT / "crops" / f"{i}.jpg")
    keep = [dict({k: x[k] for k in ("name", "file", "direction", "k", "label", "conf", "why")},
                 twins=[[t["name"], t["file"], t["direction"], t["k"], t["label"]] for t in x["twins"]]) for x in items]
    (OUT / "items.json").write_text(json.dumps(keep, indent=0), encoding="utf-8")
    (OUT / "review.html").write_text(PAGE.replace("__ITEMS__", json.dumps(keep)), encoding="utf-8")
    dates = defaultdict(int)
    for x in items:
        dates[x["date"]] += 1
    print(f"Doubtful bolls in training views: {len(c)}; picked {len(items)} "
          f"(+ {sum(len(x['twins']) for x in items)} copies in near-identical photos, answered together) "
          f"(open {sum(x['label'] == 'open_boll' for x in items)}, closed {sum(x['label'] == 'closed_boll' for x in items)})")
    print("Per date: " + ", ".join(f"{d} {n}" for d, n in sorted(dates.items())))
    print(f"Open in your browser: {OUT / 'review.html'}")


def apply():
    f = Path.home() / "Downloads" / "boll_review_answers.csv"
    if len(sys.argv) > 2:
        f = Path(sys.argv[2])
    if not f.exists():
        sys.exit(f"Answers not found: {f}  (press 'Download answers' on the page, or give the file: apply <path>)")
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    ch = defaultdict(int)
    for r in rows:
        ch[(r["old"], r["new"])] += 1
    print(f"{len(rows)} answers in {f}")
    for (o, n), v in sorted(ch.items(), key=lambda t: -t[1]):
        print(f"  {o:<12} -> {n:<12}{v:>5}{'' if o != n else '   (kept)'}")


def write():
    """Apply the answers to the label files (backup first). Boxes are found by their coordinates, not line order."""
    import shutil
    from datetime import datetime
    f = OUT / "answers.csv"
    src = Path.home() / "Downloads" / "boll_review_answers.csv"
    if src.exists() and (not f.exists() or src.stat().st_mtime > f.stat().st_mtime):
        shutil.copy2(src, f)
    rows = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["old"] != r["new"] and r["new"] != "unsure"]
    boxes = {}
    for line in open(LAB / "sam3.jsonl", encoding="utf-8"):
        r = json.loads(line)
        boxes[Path(r["view"]).name] = (r["bolls"], r["width"], r["height"])
    dirs = [LAB / "bolls_dataset" / "labels", ROOT / "agrieye_output" / "boll_detector" / "data_big" / "train" / "labels"]
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    for d in dirs:
        shutil.copytree(d, d.parent / f"labels_before_review_{stamp}")
    done, miss = defaultdict(int), 0
    for d in dirs:
        by = defaultdict(list)
        for r in rows:
            by[r["view"]].append(r)
        for view, rs in by.items():
            lf = d / f"{Path(view).stem}.txt"
            if not lf.exists():
                continue
            lines = [l.split() for l in lf.read_text().splitlines() if l.strip()]
            bolls, W, H = boxes[view]
            for r in rs:
                x1, y1, x2, y2 = bolls[int(r["k"])]["box"]
                want = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
                hit = [i for i, l in enumerate(lines) if l and all(abs(float(a) - b) < 2e-5 for a, b in zip(l[1:], want))]
                if not hit:
                    miss += 1; continue
                i = hit[0]
                if r["new"] == "not_boll":
                    lines[i] = None
                else:
                    lines[i][0] = "0" if r["new"] == "open_boll" else "1"
                done[(d.parent.parent.name if d.parent.name == "train" else d.parent.name, r["old"], r["new"])] += 1
            keep = [" ".join(l) for l in lines if l]
            lf.write_text(chr(10).join(keep) + (chr(10) if keep else ""), encoding="utf-8")
    for k, v in sorted(done.items()):
        print(f"  {k[0]:<14} {k[1]:<12} -> {k[2]:<12}{v:>5}")
    print(f"  not found: {miss}   backups: labels_before_review_{stamp} next to each labels folder")
    print(f"  your answers kept in: {f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    apply() if cmd == "apply" else write() if cmd == "write" else build()
