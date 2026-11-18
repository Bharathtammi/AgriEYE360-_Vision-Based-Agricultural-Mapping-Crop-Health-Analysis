"""
LABEL REVIEW of all 140 different pictures (clean photo split) - one structured rule for hidden / unclear bolls,
plus the bolls SAM 3 never boxed (suggested by the tile80 model). No training.

  python build_label_review.py            # 1) model suggestions (cached)  2) build the page
  python build_label_review.py write      # apply your answers -> agrieye_output/boll_detector/data_clean

THE RULE (same for train, valid and test):
  CLEAR          about half or more of the boll visible AND you can tell open / closed   -> label kept
  HIDDEN/UNCLEAR less than about half visible, blurred, dark shape against the sky, or
                 you cannot say open / closed                                            -> label removed and
                 the area painted grey in train + valid; in TEST an IGNORE ZONE (finding it or missing it does not count)
  TINY           < 24 px                                                                 -> ignored in scoring (as before)

On the page (one picture at a time, order test -> valid -> train):
  solid box (pink open / cyan closed) = existing label   click: clear <-> hidden/unclear   right-click: open <-> closed
  dashed white box = model found something with no label  click: not a boll -> open -> closed -> hidden boll -> ...
  drag on an empty spot = draw a boll nobody found        click it: closed -> open -> hidden -> delete
  N / Right arrow = picture done, next    P / Left = back    H (hold) = hide boxes    Z = zoom
Answers stay in the browser; 'Download answers' saves label_review_answers.csv in Downloads.
Output: agrieye_output/boll_review/labels_all/review.html (+ img/, items.json)
"""
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

import eval_roboflow_detect as E

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "agrieye_output" / "boll_detector" / "data_photo_split"
RUN = SRC / "runs" / "tile80"
OUT = ROOT / "agrieye_output" / "boll_review" / "labels_all"
DST = ROOT / "agrieye_output" / "boll_detector" / "data_clean"
CONF, MIN_PX = 0.25, 16
NAMES = ["open_boll", "closed_boll"]
ORDER = {"test": 0, "valid": 1, "train": 2}


def rows():
    r = list(csv.DictReader(open(SRC / "split.csv", encoding="utf-8")))
    return sorted(r, key=lambda x: (ORDER[x["split"]], x["date"], x["view"]))


def suggestions(rs):
    import train_visit as T
    preds = {}
    for s in ("test", "valid", "train"):
        preds.update(T.predict(RUN, SRC / s / "images", 800, False, tile=True))   # cached pred_<split>.json
    out = {}
    for r in rs:
        stem = Path(r["view"]).stem
        g = E.ground_truth(SRC / r["split"] / "labels" / f"{stem}.txt", 1440, 1440)
        cand = [p for p in preds.get(stem, []) if p[1] >= CONF and max(p[2][2] - p[2][0], p[2][3] - p[2][1]) >= MIN_PX]
        cand = E.nms([(0, s, b, c) for c, s, b in [(p[0], p[1], p[2]) for p in cand]], 0.4)   # one box per spot
        keep = []
        for _, s, b, c in cand:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if any(E.iou(b, gb) >= 0.3 or (gb[0] <= cx <= gb[2] and gb[1] <= cy <= gb[3]) for _, gb in g):
                continue
            keep.append(dict(box=[round(v, 1) for v in b], cls=int(c), conf=round(s, 2)))
        out[r["view"]] = (g, keep)
    return out


def build():
    rs = rows()
    sug = suggestions(rs)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for r in rs:
        src = SRC / r["split"] / "images" / r["view"]
        dst = OUT / "img" / r["view"]
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        g, s = sug[r["view"]]
        items.append(dict(view=r["view"], split=r["split"], date=r["date"], labels=r["labels"],
                          lab=[dict(box=[round(v, 1) for v in b], cls=c) for c, b in g], sug=s))
    (OUT / "items.json").write_text(json.dumps(items), encoding="utf-8")
    (OUT / "review.html").write_text(PAGE.replace("__ITEMS__", json.dumps(items)), encoding="utf-8")
    n_lab = sum(len(x["lab"]) for x in items); n_sug = sum(len(x["sug"]) for x in items)
    by = Counter(x["split"] for x in items)
    print(f"{len(items)} pictures (test {by['test']}, valid {by['valid']}, train {by['train']}): "
          f"{n_lab} labelled bolls, {n_sug} model suggestions with no label (conf >= {CONF})")
    for sp in ("test", "valid", "train"):
        print(f"  {sp:<6} suggestions {sum(len(x['sug']) for x in items if x['split'] == sp)}")
    print(f"Open in your browser: {OUT / 'review.html'}")


# ------------------------------------------------------------------------------------------------ write
def yolo(c, b):
    x1, y1, x2, y2 = b
    return f"{c} {(x1 + x2) / 2 / 1440:.5f} {(y1 + y2) / 2 / 1440:.5f} {(x2 - x1) / 1440:.5f} {(y2 - y1) / 1440:.5f}"


def write():
    f = OUT / "answers.csv"
    dls = sorted((Path.home() / "Downloads").glob("label_review_answers*.csv"), key=lambda x: x.stat().st_mtime)
    dl = dls[-1] if dls else Path.home() / "Downloads" / "label_review_answers.csv"   # newest ('... (1).csv' etc.)
    print(f"Answers: {dl}")
    if dl.exists() and (not f.exists() or dl.stat().st_mtime > f.stat().st_mtime):
        shutil.copy2(dl, f)
    if not f.exists():
        sys.exit(f"No answers yet: press 'Download answers' on the page (expected {dl})")
    items = {x["view"]: x for x in json.loads((OUT / "items.json").read_text(encoding="utf-8"))}
    ans = defaultdict(lambda: dict(done=False, lab={}, sug={}, new=[]))
    for r in csv.DictReader(open(f, encoding="utf-8")):
        a = ans[r["view"]]
        if r["kind"] == "picture":
            a["done"] = r["new"] == "done"
        elif r["kind"] == "label":
            a["lab"][int(r["k"])] = r["new"]
        elif r["kind"] == "suggestion":
            a["sug"][int(r["k"])] = r["new"]
        elif r["kind"] == "new":
            a["new"].append((json.loads(r["box"]), r["new"]))
    if DST.exists():
        DST.rename(DST.parent / f"data_clean_before_{datetime.now():%Y%m%d_%H%M%S}")
    cnt = Counter(); log = []
    for view, x in items.items():
        sp = x["split"]; a = ans[view]; stem = Path(view).stem
        for sub in ("images", "labels") + (("ignore",) if sp == "test" else ()):
            (DST / sp / sub).mkdir(parents=True, exist_ok=True)
        keep, hidden = [], []
        if not a["done"]:                                   # not reviewed: labels unchanged
            keep = [(l["cls"], l["box"]) for l in x["lab"]]; cnt[(sp, "pictures not reviewed")] += 1
        else:
            cnt[(sp, "pictures reviewed")] += 1
            for k, l in enumerate(x["lab"]):
                v = a["lab"].get(k, "clear")
                if v == "unclear":
                    hidden.append(l["box"]); cnt[(sp, "label -> hidden/unclear")] += 1
                elif v in ("open", "closed"):
                    c = 0 if v == "open" else 1
                    cnt[(sp, "label class changed")] += c != l["cls"]
                    keep.append((c, l["box"]))
                else:
                    keep.append((l["cls"], l["box"]))
                log.append([view, sp, "label", k, json.dumps(l["box"]), NAMES[l["cls"]], v])
            for k, s in enumerate(x["sug"]):
                v = a["sug"].get(k, "not_boll")
                if v in ("open", "closed"):
                    keep.append((0 if v == "open" else 1, s["box"])); cnt[(sp, f"model suggestion -> {v} boll added")] += 1
                elif v == "unclear":
                    hidden.append(s["box"]); cnt[(sp, "model suggestion -> hidden boll")] += 1
                else:
                    cnt[(sp, "model suggestion -> not a boll")] += 1
                log.append([view, sp, "suggestion", k, json.dumps(s["box"]), f"model {NAMES[s['cls']]} {s['conf']}", v])
            for b, v in a["new"]:
                if v in ("open", "closed"):
                    keep.append((0 if v == "open" else 1, b)); cnt[(sp, f"drawn by hand -> {v} boll added")] += 1
                elif v == "unclear":
                    hidden.append(b); cnt[(sp, "drawn by hand -> hidden boll")] += 1
                log.append([view, sp, "new", "", json.dumps(b), "", v])
        im = Image.open(SRC / sp / "images" / view).convert("RGB")
        if sp == "test":                                   # test picture untouched; hidden bolls = ignore zones
            shutil.copy2(SRC / sp / "images" / view, DST / sp / "images" / view)
            (DST / sp / "ignore" / f"{stem}.txt").write_text("".join(yolo(1, b) + "\n" for b in hidden), encoding="utf-8")
        else:                                              # train / valid: hidden bolls painted grey, label removed
            d = ImageDraw.Draw(im)
            for b in hidden:
                d.rectangle((b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2), fill=(114, 114, 114))
            im.save(DST / sp / "images" / view, quality=95)
        (DST / sp / "labels" / f"{stem}.txt").write_text("".join(yolo(c, b) + "\n" for c, b in keep), encoding="utf-8")
        for c, _ in keep:
            cnt[(sp, f"final {NAMES[c]}")] += 1
        cnt[(sp, "final hidden (grey / ignore)")] += len(hidden)
    NL = chr(10)
    (DST / "data.yaml").write_text(f"path: {DST.as_posix()}{NL}train: train/images{NL}val: valid/images{NL}test: test/images{NL}"
                                   f"nc: 2{NL}names: ['open_boll', 'closed_boll']{NL}", encoding="utf-8")
    shutil.copy2(SRC / "split.csv", DST / "split.csv")
    with open(DST / "review_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["view", "split", "kind", "k", "box", "before", "answer"]); w.writerows(log)
    keys = sorted({k for _, k in cnt})
    L = ["Label review applied (build_label_review.py). Rule: hidden/unclear = less than about half visible, blurred,",
         "dark against the sky, or open/closed cannot be told -> grey + unlabelled (train/valid), ignore zone (test).", "",
         f"{'':<40}{'train':>8}{'valid':>8}{'test':>8}"]
    L += [f"{k:<40}" + "".join(f"{cnt[(s, k)]:>8}" for s in ("train", "valid", "test")) for k in keys]
    (DST / "summary.txt").write_text(NL.join(L), encoding="utf-8")
    print(NL.join(L) + f"{NL}{NL}Written: {DST}  (source {SRC} unchanged; log review_log.csv)")


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Boll label review</title>
<style>
body{margin:0;background:#1b1b1b;color:#eee;font:14px system-ui,sans-serif}
#bar{position:sticky;top:0;background:#262626;padding:8px 12px;z-index:2;border-bottom:1px solid #444}
#bar button{background:#3a3a3a;color:#eee;border:1px solid #555;border-radius:4px;padding:5px 10px;margin-right:4px;cursor:pointer}
#bar button:hover{background:#4a4a4a}
#info{margin:6px 0 2px;color:#ccc}
#help{color:#999;font-size:12px}
#wrap{overflow:auto;height:calc(100vh - 110px)}
canvas{display:block;cursor:crosshair}
.k{display:inline-block;width:12px;height:12px;vertical-align:middle;margin:0 3px 0 10px;border:2px solid}
</style></head><body>
<div id="bar">
 <button onclick="go(-1)">&#9664; Back (P)</button><button onclick="go(1)">Done, next (N) &#9654;</button>
 <button onclick="zoom()">Zoom (Z)</button><button onclick="jump()">Go to first not done</button>
 <button onclick="dl()">Download answers</button>
 <span id="prog"></span>
 <div id="info"></div>
 <div id="help"><b>Solid box</b> = label: click = clear &harr; <b>hidden/unclear</b> (grey, crossed); right-click = open &harr; closed.
 <b>Dashed white</b> = model found something unlabelled: click = not a boll &rarr; open &rarr; closed &rarr; hidden boll.
 <b>Drag</b> on an empty spot = draw a missed boll (click it: closed &rarr; open &rarr; hidden &rarr; delete). Hold <b>H</b> = hide boxes.
 <span class="k" style="border-color:#ff4fd8"></span>open<span class="k" style="border-color:#2fd0ff"></span>closed
 <span class="k" style="border-color:#888;background:#888"></span>hidden/unclear<span class="k" style="border-color:#fff;border-style:dashed"></span>not decided = not a boll
 <br>Rule: <b>hidden/unclear</b> = less than about half of the boll visible, blurred, a dark shape against the sky, or you cannot say open or closed.</div>
</div>
<div id="wrap"><canvas id="cv"></canvas></div>
<script>
const ITEMS=__ITEMS__, KEY="label_review_all_v1";
let S=JSON.parse(localStorage.getItem(KEY)||"{}"), i=0, Z=[0.6,0.8,1.0,1.4], zi=1, hide=false, img=new Image(), drag=null;
const cv=document.getElementById("cv"), cx=cv.getContext("2d");
function st(v){return S[v]||(S[v]={done:false,lab:{},sug:{},new:[]})}
function save(){localStorage.setItem(KEY,JSON.stringify(S))}
function col(c){return c==0?"#ff4fd8":"#2fd0ff"}
function load(){img=new Image();img.onload=draw;img.src="img/"+encodeURIComponent(ITEMS[i].view);}
function boxes(){ // every clickable box of this picture, smallest first
 const x=ITEMS[i], s=st(x.view), out=[];
 x.lab.forEach((l,k)=>out.push({kind:"label",k,box:l.box,cls:l.cls}));
 x.sug.forEach((l,k)=>out.push({kind:"suggestion",k,box:l.box,cls:l.cls,conf:l.conf}));
 s.new.forEach((n,k)=>out.push({kind:"new",k,box:n.box}));
 return out.sort((a,b)=>(a.box[2]-a.box[0])*(a.box[3]-a.box[1])-(b.box[2]-b.box[0])*(b.box[3]-b.box[1]));
}
function state(b){const s=st(ITEMS[i].view);
 if(b.kind=="label"){const v=s.lab[b.k]; return v||(b.cls==0?"open":"closed");}
 if(b.kind=="suggestion") return s.sug[b.k]||"not_boll";
 return s.new[b.k].v;}
function draw(){
 const x=ITEMS[i], s=st(x.view), z=Z[zi];
 cv.width=1440*z; cv.height=1440*z; cx.drawImage(img,0,0,cv.width,cv.height);
 const done=ITEMS.filter(t=>st(t.view).done).length;
 document.getElementById("prog").textContent=`picture ${i+1} / ${ITEMS.length}   reviewed ${done}`;
 let cnt={label:0,unclear:0,sug:0,added:0};
 boxes().forEach(b=>{const v=state(b); if(b.kind=="label"){cnt.label++; if(v=="unclear")cnt.unclear++;}
  if(b.kind=="suggestion"){cnt.sug++; if(v!="not_boll")cnt.added++;} if(b.kind=="new"&&v!="unclear")cnt.added++;});
 document.getElementById("info").innerHTML=`<b>${x.split.toUpperCase()}</b> &nbsp; ${x.view} &nbsp; (${x.labels}) &nbsp; `+
  `labels ${cnt.label}, marked hidden ${cnt.unclear} &nbsp; model suggestions ${cnt.sug}, bolls added ${cnt.added}`+
  (s.done?' &nbsp; <span style="color:#7f7">&#10003; reviewed</span>':'');
 if(hide) return;
 boxes().forEach(b=>{const v=state(b), [x1,y1,x2,y2]=b.box.map(t=>t*z);
  cx.setLineDash(b.kind=="suggestion"&&v=="not_boll"?[6,4]:[]); cx.lineWidth=2;
  if(v=="unclear"){cx.fillStyle="rgba(140,140,140,.55)";cx.fillRect(x1,y1,x2-x1,y2-y1);cx.strokeStyle="#ff5050";
   cx.beginPath();cx.moveTo(x1,y1);cx.lineTo(x2,y2);cx.moveTo(x2,y1);cx.lineTo(x1,y2);cx.stroke();cx.strokeStyle="#bbb";}
  else if(v=="not_boll") cx.strokeStyle="#fff"; else cx.strokeStyle=col(v=="open"?0:1);
  cx.strokeRect(x1,y1,x2-x1,y2-y1);
  if(b.kind!="label"&&v!="not_boll"&&v!="unclear"){cx.fillStyle=cx.strokeStyle;cx.font="bold 14px sans-serif";cx.fillText("+",x1+2,y1+13);}
 });
 cx.setLineDash([]);
 if(drag&&drag.moved){cx.strokeStyle="#ff0";cx.strokeRect(drag.x0*z,drag.y0*z,(drag.x1-drag.x0)*z,(drag.y1-drag.y0)*z);}
}
function at(e){const r=cv.getBoundingClientRect(), z=Z[zi]; return [(e.clientX-r.left)/z,(e.clientY-r.top)/z];}
function hit(p){return boxes().find(b=>p[0]>=b.box[0]-3&&p[0]<=b.box[2]+3&&p[1]>=b.box[1]-3&&p[1]<=b.box[3]+3);}
const NEXT={suggestion:{not_boll:"open",open:"closed",closed:"unclear",unclear:"not_boll"},new:{closed:"open",open:"unclear"}};
cv.addEventListener("mousedown",e=>{if(e.button!=0)return; const p=at(e); drag={x0:p[0],y0:p[1],x1:p[0],y1:p[1],moved:false};});
cv.addEventListener("mousemove",e=>{if(!drag)return; const p=at(e); drag.x1=p[0]; drag.y1=p[1];
 if(Math.abs(drag.x1-drag.x0)>6||Math.abs(drag.y1-drag.y0)>6){drag.moved=true; draw();}});
cv.addEventListener("mouseup",e=>{if(e.button!=0||!drag)return; const s=st(ITEMS[i].view), d=drag; drag=null;
 if(d.moved){const b=[Math.min(d.x0,d.x1),Math.min(d.y0,d.y1),Math.max(d.x0,d.x1),Math.max(d.y0,d.y1)].map(t=>Math.round(t*10)/10);
  if(b[2]-b[0]>=8&&b[3]-b[1]>=8) s.new.push({box:b,v:"closed"}); save(); draw(); return;}
 const b=hit([d.x0,d.y0]); if(!b) return;
 if(b.kind=="label"){const v=state(b); if(v=="unclear") delete s.lab[b.k]; else s.lab[b.k]="unclear";}
 else if(b.kind=="suggestion") s.sug[b.k]=NEXT.suggestion[state(b)];
 else {const v=s.new[b.k].v; if(v=="unclear") s.new.splice(b.k,1); else s.new[b.k].v=NEXT.new[v];}
 save(); draw();});
cv.addEventListener("contextmenu",e=>{e.preventDefault(); const b=hit(at(e)); if(!b||b.kind!="label") return;
 const s=st(ITEMS[i].view), v=state(b); if(v=="unclear") return;
 const nv=v=="open"?"closed":"open"; if((nv=="open"?0:1)==b.cls) delete s.lab[b.k]; else s.lab[b.k]=nv; save(); draw();});
function go(d){if(d>0){st(ITEMS[i].view).done=true; save();} i=Math.max(0,Math.min(ITEMS.length-1,i+d)); localStorage.setItem(KEY+"_pos",i); load(); document.getElementById("wrap").scrollTo(0,0);}
function zoom(){zi=(zi+1)%Z.length; draw();}
function jump(){const k=ITEMS.findIndex(t=>!st(t.view).done); if(k>=0){i=k; load();}}
document.addEventListener("keydown",e=>{if(e.key=="n"||e.key=="N"||e.key=="ArrowRight")go(1); else if(e.key=="p"||e.key=="P"||e.key=="ArrowLeft")go(-1);
 else if(e.key=="z"||e.key=="Z")zoom(); else if((e.key=="h"||e.key=="H")&&!hide){hide=true;draw();}});
document.addEventListener("keyup",e=>{if(e.key=="h"||e.key=="H"){hide=false;draw();}});
function q(v){v=String(v);return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;}
function dl(){const R=[["view","split","kind","k","box","old","new"]];
 ITEMS.forEach(x=>{const s=st(x.view); R.push([x.view,x.split,"picture","","","",s.done?"done":"not done"]);
  Object.entries(s.lab).forEach(([k,v])=>R.push([x.view,x.split,"label",k,JSON.stringify(x.lab[k].box),x.lab[k].cls==0?"open":"closed",v]));
  Object.entries(s.sug).forEach(([k,v])=>R.push([x.view,x.split,"suggestion",k,JSON.stringify(x.sug[k].box),"model",v]));
  s.new.forEach(n=>R.push([x.view,x.split,"new","",JSON.stringify(n.box),"",n.v]));});
 const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([R.map(r=>r.map(q).join(",")).join("\n")],{type:"text/csv"}));
 a.download="label_review_answers.csv"; a.click();}
i=+(localStorage.getItem(KEY+"_pos")||0); load();
</script></body></html>"""


if __name__ == "__main__":
    write() if len(sys.argv) > 1 and sys.argv[1] == "write" else build()
