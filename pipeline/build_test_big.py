"""
Step 3: a BIGGER HAND-CHECKED TEST SET for the boll detector (the 18-view test is too small: one photo moves the score).

  python build_test_big.py plan     # only show which photos would be picked and what training would lose (no changes)
  python build_test_big.py          # build: copy the picked views + the check page (training not touched yet)
  python build_test_big.py write    # after your check: write the test labels, take the photos (and their
                                    # neighbours) OUT of training - moved, not deleted - backups first

Pick : PHOTOS photos per main field day (31 Jul, 4 Aug, 6 Aug), one per time quarter of the day, all 4 directions,
       with plenty of bolls (photos with open bolls preferred, they are rare).
Hold out: the photo, every photo the same day within 15 s, and every near-identical shot (still camera) up to 120 s
       away -> moved from data_big/train to data_big/held_out_test_big, so no model trains on the same scene.
Check: agrieye_output/boll_review/test_big/review.html - every box of every view; click to change, drag to add.
Output: agrieye_output/boll_detector/test_big/{images,labels}, selection.csv, held_out.txt
No training.
"""
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import eval_roboflow_detect as E

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "agrieye_output" / "boll_detector" / "data_big"
TEST = ROOT / "agrieye_output" / "boll_detector" / "test_big"
HELD = DATA / "held_out_test_big"
ALL = ROOT / "agrieye_output" / "pov_all_labels" / "bolls_dataset" / "labels"
PAGE_DIR = ROOT / "agrieye_output" / "boll_review" / "test_big"
ANS_NAME = "boll_test_big_answers.json"
DAYS, PHOTOS = ["2026-07-31", "2026-08-04", "2026-08-06"], 4
WINDOW_S, SAME_S, SAME_DIFF = 15, 120, 6.0
DIRS = "NESW"


def secs(stem):                                   # image_20260804_171515 -> seconds of the day
    t = stem.split("_")[2]
    return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


def photos():
    out = {}
    rows = list(csv.DictReader(open(DATA / "sources.csv", encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        by[(r["date"], r["view"].rsplit("_", 1)[0])].append(r)
    for (date, stem), vs in by.items():
        lab = [(DATA / "train" / "labels" / f"{Path(v['view']).stem}.txt").read_text().split("\n") for v in vs]
        out[stem] = dict(date=date, stem=stem, t=secs(stem), views=sorted(v["view"] for v in vs),
                         auto=all(v["labels"].startswith("automatic") for v in vs),
                         boxes=sum(1 for t in lab for l in t if l.strip()),
                         open=sum(1 for t in lab for l in t if l.startswith("0 ")))
    return out


_th = {}


def thumb(view):
    if view not in _th:
        _th[view] = np.asarray(Image.open(DATA / "train" / "images" / view).convert("L").resize((24, 24)), np.float32)
    return _th[view]


def same_scene(a, b):
    """Neighbouring shot (<= 15 s) or the same scene from a still camera (near-identical pictures, <= 120 s)."""
    if a["date"] != b["date"]:
        return False
    dt = abs(a["t"] - b["t"])
    if dt <= WINDOW_S:
        return True
    if dt > SAME_S:
        return False
    d = [float(abs(thumb(f"{a['stem']}_{c}.jpg") - thumb(f"{b['stem']}_{c}.jpg")).mean()) for c in DIRS
         if f"{a['stem']}_{c}.jpg" in a["views"] and f"{b['stem']}_{c}.jpg" in b["views"]]
    return bool(d) and float(np.mean(d)) < SAME_DIFF


def plan():
    P = photos()
    picks, held = [], set()
    for day in DAYS:
        ps = sorted((p for p in P.values() if p["date"] == day), key=lambda p: p["t"])
        q = np.array_split(np.arange(len(ps)), PHOTOS)
        for part in q:
            best = None
            for i in part:
                p = ps[i]
                if not p["auto"] or len(p["views"]) != 4 or p["stem"] in held:
                    continue
                zone = {o["stem"] for o in ps if same_scene(p, o)}
                if zone & {x["stem"] for x in picks} or zone & held:
                    continue
                cost = sum(len(P[s]["views"]) for s in zone)
                score = min(p["boxes"], 120) + 25 * min(p["open"], 3) - 4 * cost
                if best is None or score > best[0]:
                    best = (score, p, zone)
            if best:
                picks.append(best[1]); held |= best[2]
    return P, picks, held


def show(P, picks, held):
    nv = sum(len(P[s]["views"]) for s in held)
    print(f"{'photo':<24}{'date':<12}{'time':<10}{'boxes':>6}{'open':>6}  held out with it")
    for p in picks:
        zone = sorted(s for s in held if same_scene(p, P[s]) and s != p["stem"])
        t = p["stem"].split("_")[2]
        print(f"{p['stem']:<24}{p['date']:<12}{t[:2]}:{t[2:4]}:{t[4:]:<4}{p['boxes']:>6}{p['open']:>6}  "
              f"{len(zone)} photos: {', '.join(s.split('_')[2] for s in zone)}")
    print(f"\nTest: {len(picks)} photos = {sum(len(p['views']) for p in picks)} views, "
          f"{sum(p['boxes'] for p in picks)} boxes to check (open {sum(p['open'] for p in picks)})")
    print(f"Training loses {len(held)} photos = {nv} of {sum(len(p['views']) for p in P.values())} views "
          f"(the test photos + their neighbours / near-identical shots)")


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Test set check - every boll</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#1d1f21;color:#eee}
#bar{position:sticky;top:0;background:#2b2e31;padding:8px 12px;z-index:5;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{background:#444;color:#eee;border:1px solid #666;border-radius:4px;padding:5px 10px;cursor:pointer;font-size:14px}
button.on{background:#08795a;border-color:#0c9}
#help{font-size:13px;color:#bbb;margin:6px 12px;line-height:1.5}
#list{margin:0 12px}#list span{display:inline-block;min-width:24px;padding:1px 2px;margin:1px;font-size:11px;text-align:center;background:#444;cursor:pointer;border-radius:3px}
#list span.done{background:#08795a}#list span.cur{outline:2px solid #fff}
#wrap{position:relative;display:inline-block;margin:8px 12px;user-select:none}
#wrap img{display:block}
svg{position:absolute;left:0;top:0}
rect{fill:transparent;pointer-events:all;stroke-width:2.5;cursor:pointer}
rect.c1{stroke:#2fd0ff}rect.c0{stroke:#ff4fd8;stroke-width:3}
rect.del{stroke:#ff3030;stroke-dasharray:6 4;stroke-width:1.5}
rect.small{stroke-width:1.2;stroke-dasharray:3 3}
rect.tmp{stroke:#ffe14a;stroke-dasharray:4 3;pointer-events:none}
.k1{color:#2fd0ff;font-weight:bold}.k0{color:#ff4fd8;font-weight:bold}.kd{color:#ff3030;font-weight:bold}
</style></head><body>
<div id="bar">
<button onclick="go(-1)">&#9664; Prev</button><b id="title"></b><button onclick="go(1)">Next &#9654;</button>
<span>New boxes: <button id="d1" onclick="setDraw(1)">CLOSED (c)</button> <button id="d0" onclick="setDraw(0)">OPEN (o)</button></span>
<button onclick="undo()">Undo (z)</button><button id="zb" onclick="zoom()">Zoom 100% (f)</button>
<button id="chk" onclick="check()">View checked (Enter)</button>
<span id="count"></span><button onclick="dl()">Download answers</button>
</div>
<div id="help">Check <b>every boll</b> in each view. <span class="k1">Blue = closed</span>, <span class="k0">pink = open</span>,
<span class="kd">red dashed = removed</span>.
<b>Click a box</b> to change it: closed &rarr; open &rarr; removed &rarr; closed.
<b>Drag</b> on the picture to add a missed boll (hold <b>Shift</b> to draw inside another box); the new box gets the class chosen above.
Thin dotted boxes are tiny far-away bolls (under 24 px) - they are not scored, no need to fix them, and no need to add tiny ones.
When a view is right, press <b>Enter</b> (View checked) - it goes to the next view. Saved in this browser as you go.
When all squares are green: <b>Download answers</b>.</div>
<div id="list"></div>
<div id="wrap"><img id="im"><svg id="sv"></svg></div>
<script>
const V=__VIEWS__, KEY="boll_test_big_v1", W=1440;
let S=JSON.parse(localStorage.getItem(KEY)||"{}"), i=+(localStorage.getItem(KEY+"_i")||0), draw=1, full=false, hist=[], drag=null, k=1;
function st(n){if(!S[n.name])S[n.name]={checked:false,boxes:n.boxes.map(b=>b.concat([0]))};return S[n.name];}
function save(){localStorage.setItem(KEY,JSON.stringify(S));localStorage.setItem(KEY+"_i",i);}
function edit(){hist.push(JSON.stringify(st(V[i]).boxes));}
function render(){
 const n=V[i],s=st(n),im=document.getElementById('im'),sv=document.getElementById('sv');
 const size=full?W:Math.min(window.innerWidth-40,W);k=size/W;
 if(im.getAttribute('src')!==n.src)im.src=n.src;im.style.width=size+'px';sv.setAttribute('width',size);sv.setAttribute('height',size);
 sv.innerHTML=s.boxes.map((b,j)=>{const sm=Math.max(b[3]-b[1],b[4]-b[2])<24;
  return `<rect data-j="${j}" class="${b[5]?'del':'c'+b[0]}${sm&&!b[5]?' small':''}" x="${b[1]*k}" y="${b[2]*k}" width="${(b[3]-b[1])*k}" height="${(b[4]-b[2])*k}"/>`}).join('');
 const live=s.boxes.filter(b=>!b[5]);
 document.getElementById('title').textContent=`View ${i+1} / ${V.length}: ${n.name}  (closed ${live.filter(b=>b[0]==1).length}, open ${live.filter(b=>b[0]==0).length})`;
 const c=document.getElementById('chk');c.className=s.checked?'on':'';c.textContent=s.checked?'View checked ✓':'View checked (Enter)';
 document.getElementById('d1').className=draw==1?'on':'';document.getElementById('d0').className=draw==0?'on':'';
 document.getElementById('zb').textContent=full?'Fit width (f)':'Zoom 100% (f)';
 const done=V.filter(x=>S[x.name]&&S[x.name].checked).length;
 document.getElementById('count').textContent=`${done} / ${V.length} views checked`;
 document.getElementById('list').innerHTML=V.map((x,j)=>`<span class="${S[x.name]&&S[x.name].checked?'done':''}${j==i?' cur':''}" onclick="jump(${j})">${j+1}</span>`).join('');
}
function go(d){i=Math.max(0,Math.min(V.length-1,i+d));hist=[];save();render();window.scrollTo(0,0);}
function jump(j){i=j;hist=[];save();render();}
function setDraw(c){draw=c;render();}
function zoom(){full=!full;render();}
function undo(){if(hist.length){st(V[i]).boxes=JSON.parse(hist.pop());save();render();}}
function check(){const s=st(V[i]);s.checked=!s.checked;save();if(s.checked&&i<V.length-1)go(1);else render();}
function pos(e){const r=document.getElementById('sv').getBoundingClientRect();return[(e.clientX-r.left)/k,(e.clientY-r.top)/k];}
const sv=document.getElementById('sv');
sv.addEventListener('mousedown',e=>{
 if(e.button!==0)return;e.preventDefault();
 const t=e.target;
 if(t.tagName==='rect'&&!e.shiftKey){
  const b=st(V[i]).boxes[+t.dataset.j];edit();
  if(b[5]){b[5]=0;b[0]=1;}else if(b[0]==1)b[0]=0;else b[5]=1;
  save();render();return;}
 drag={a:pos(e)};});
window.addEventListener('mousemove',e=>{
 if(!drag)return;const b=pos(e),a=drag.a;
 let r=document.querySelector('rect.tmp');if(!r){r=document.createElementNS('http://www.w3.org/2000/svg','rect');r.setAttribute('class','tmp');sv.appendChild(r);}
 r.setAttribute('x',Math.min(a[0],b[0])*k);r.setAttribute('y',Math.min(a[1],b[1])*k);
 r.setAttribute('width',Math.abs(b[0]-a[0])*k);r.setAttribute('height',Math.abs(b[1]-a[1])*k);});
window.addEventListener('mouseup',e=>{
 if(!drag)return;const a=drag.a,b=pos(e);drag=null;
 const x1=Math.max(0,Math.min(a[0],b[0])),y1=Math.max(0,Math.min(a[1],b[1])),x2=Math.min(W,Math.max(a[0],b[0])),y2=Math.min(W,Math.max(a[1],b[1]));
 if(x2-x1>=8&&y2-y1>=8){edit();st(V[i]).boxes.push([draw,Math.round(x1),Math.round(y1),Math.round(x2),Math.round(y2),0]);save();}
 render();});
document.addEventListener('keydown',e=>{
 if(e.key==='c')setDraw(1);else if(e.key==='o')setDraw(0);else if(e.key==='z')undo();else if(e.key==='f')zoom();
 else if(e.key==='Enter'){e.preventDefault();check();}else if(e.key==='ArrowRight')go(1);else if(e.key==='ArrowLeft')go(-1);});
window.addEventListener('resize',render);
function dl(){
 const out={};V.forEach(n=>{const s=st(n);out[n.name]={checked:s.checked,boxes:s.boxes.filter(b=>!b[5]).map(b=>b.slice(0,5))};});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out)],{type:'application/json'}));
 a.download='__ANS__';a.click();}
render();
</script></body></html>"""


def build():
    P, picks, held = plan()
    show(P, picks, held)
    (TEST / "images").mkdir(parents=True, exist_ok=True)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    views = []
    for p in picks:
        for v in p["views"]:
            shutil.copy2(DATA / "train" / "images" / v, TEST / "images" / v)
            boxes = [[c, round(b[0]), round(b[1]), round(b[2]), round(b[3])]
                     for c, b in E.ground_truth(DATA / "train" / "labels" / f"{Path(v).stem}.txt", 1440, 1440)]
            views.append(dict(name=v, src=f"../../boll_detector/test_big/images/{v}", boxes=boxes))
    with open(TEST / "selection.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["photo", "date", "boxes", "open"])
        for p in picks:
            w.writerow([p["stem"], p["date"], p["boxes"], p["open"]])
    (TEST / "held_out.txt").write_text(chr(10).join(sorted(held)) + chr(10), encoding="utf-8")
    page = PAGE.replace("__VIEWS__", json.dumps(views)).replace("__ANS__", ANS_NAME)
    (PAGE_DIR / "review.html").write_text(page, encoding="utf-8")
    print(f"\nTraining NOT changed yet (that happens in 'write', after your check).")
    print(f"Open in your browser: {PAGE_DIR / 'review.html'}")


def write():
    ans = PAGE_DIR / "answers.json"
    src = Path.home() / "Downloads" / ANS_NAME
    if src.exists() and (not ans.exists() or src.stat().st_mtime > ans.stat().st_mtime):
        shutil.copy2(src, ans)
    A = json.loads(ans.read_text(encoding="utf-8"))
    todo = [v for v, a in A.items() if not a["checked"]]
    if todo and "--partial" not in sys.argv:
        sys.exit(f"{len(todo)} views are not marked checked yet (e.g. {todo[0]}). Finish them, or add --partial.")
    held = [s for s in (TEST / "held_out.txt").read_text().split() if s]
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    # 1) test labels
    (TEST / "labels").mkdir(parents=True, exist_ok=True)
    n = defaultdict(int)
    for v, a in A.items():
        lines = [f"{c} {(x1 + x2) / 2 / 1440:.5f} {(y1 + y2) / 2 / 1440:.5f} {(x2 - x1) / 1440:.5f} {(y2 - y1) / 1440:.5f}"
                 for c, x1, y1, x2, y2 in a["boxes"]]
        (TEST / "labels" / f"{Path(v).stem}.txt").write_text(chr(10).join(lines) + (chr(10) if lines else ""), encoding="utf-8")
        for c, *_ in a["boxes"]:
            n[c] += 1
    # 2) the same checked labels into the full boll dataset (backup first)
    shutil.copytree(ALL, ALL.parent / f"labels_before_test_big_{stamp}")
    for v in A:
        shutil.copy2(TEST / "labels" / f"{Path(v).stem}.txt", ALL / f"{Path(v).stem}.txt")
    # 3) take the test photos and their neighbours OUT of training (moved, so it can be undone)
    shutil.copy2(DATA / "sources.csv", DATA / f"sources_before_test_big_{stamp}.csv")
    moved = 0
    for sub in ("images", "labels"):
        (HELD / sub).mkdir(parents=True, exist_ok=True)
    for f in list((DATA / "train" / "images").glob("*.jpg")):
        if f.stem.rsplit("_", 1)[0] in held:
            f.replace(HELD / "images" / f.name)
            lf = DATA / "train" / "labels" / f"{f.stem}.txt"
            if lf.exists():
                lf.replace(HELD / "labels" / lf.name)
            moved += 1
    rows = list(csv.DictReader(open(DATA / "sources.csv", encoding="utf-8")))
    keep = [r for r in rows if r["view"].rsplit("_", 1)[0] not in held]
    with open(DATA / "sources.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["view", "date", "labels"]); w.writeheader(); w.writerows(keep)
    cache = DATA / "train" / "labels.cache"
    if cache.exists():
        cache.unlink()
    print(f"Test set: {len(A)} views, open {n[0]}, closed {n[1]} -> {TEST}")
    print(f"Training: {moved} views moved out to {HELD} (left: {len(keep)} views)")
    print(f"Backups: {ALL.parent / f'labels_before_test_big_{stamp}'}, {DATA / f'sources_before_test_big_{stamp}.csv'}")
    print(f"Answers kept in {ans}. No training.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "build"
    if cmd == "plan":
        show(*plan())
    elif cmd == "write":
        write()
    else:
        build()
