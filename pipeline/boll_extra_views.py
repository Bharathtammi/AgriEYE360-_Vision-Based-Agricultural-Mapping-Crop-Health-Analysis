"""
SAM 3 boll / flower boxes for the 4-POV views that the field analysis does not use
(the 22 Jul photo and the 4 repeat shots, from make_pov_dataset.py) - TRAINING DATA ONLY.

  python boll_extra_views.py

Same prompts, tiles and cleaning rules as 'agrieye.py autolabel'.
Output: agrieye_output/objects/sam3_labels_extra.jsonl  (kept apart from sam3_labels.jsonl so per-plant
results never count repeat shots twice), previews in agrieye_output/objects/previews_extra/.
Then: python make_boll_crops.py  (adds their crops to boll_crops/unsorted).
"""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import agrieye as A

ROOT = Path(__file__).resolve().parent
PD = ROOT / "agrieye_output" / "pov_dataset"


def main():
    cfg = A.Cfg(ROOT / "agrieye_config.json")
    used = {json.loads(l)["photo_id"] for l in cfg.labels.read_text(encoding="utf-8").splitlines() if l.strip()}
    extra = [r for r in csv.DictReader(open(PD / "catalog.csv", encoding="utf-8")) if r["photo_id"] not in used]
    views = [(dict(station=r["plant"], date=r["date"], photo_id=r["photo_id"]), d,
              PD / "views" / r["date"] / f"{r['photo_id']}_{d}.jpg") for r in extra for d in ("N", "E", "S", "W")]
    print(f"{len(extra)} extra photos ({', '.join(r['photo_id'] + ' ' + (r['note'] or r['plant']) for r in extra)})")
    cfg.labels = cfg.out / "objects" / "sam3_labels_extra.jsonl"
    cfg.obj_previews = cfg.out / "objects" / "previews_extra"
    A.all_povs = lambda c: views
    A.cmd_autolabel(cfg, SimpleNamespace(station=None, limit=None, device="cpu"))


if __name__ == "__main__":
    main()
