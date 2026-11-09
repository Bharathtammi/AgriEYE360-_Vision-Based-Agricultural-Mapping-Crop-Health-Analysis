"""
List every Roboflow project and trained model in your workspace (read only; nothing is changed, no credits used).

  set ROBOFLOW_API_KEY=<your private key>        (cmd; the key is never printed or saved)
  python list_roboflow_models.py

For each project: name, id, type (classification / object detection ...), image count,
and for each version: whether it has a trained model, its model id, and the scores Roboflow reports.
Marks which ones the AgriEYE pipeline uses (R1 boll classifier, R2 boll detector).
"""
import os
import sys

import requests

API = "https://api.roboflow.com"
USED = {"cotton-boll-open-or-close-3frah": "R1 boll classifier (used in analyze), model version 3",
        "cotton-bolls-field-view-trbz1": "R2 boll detector (140 labelled views, comparison only)"}


def get(url, key):
    r = requests.get(url, params={"api_key": key}, timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"Roboflow said {r.status_code} for {url.replace(API, '')}: {r.text[:200]}")
    return r.json()


def score(m):
    keys = [("map", "mAP@50"), ("map50", "mAP@50"), ("map5095", "mAP@50:95"), ("precision", "precision"),
            ("recall", "recall"), ("accuracy", "accuracy"), ("top1", "top-1")]
    out = []
    for k, name in keys:
        v = m.get(k)
        if isinstance(v, (int, float, str)) and str(v) not in ("", "None"):
            try:
                f = float(v); out.append(f"{name} {f * 100 if f <= 1 else f:.1f}%")
            except ValueError:
                pass
    return ", ".join(dict.fromkeys(out)) or "no scores reported"


def main():
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip().strip('"').strip("'").strip()
    if not key:
        sys.exit("Set your key first (cmd):  set ROBOFLOW_API_KEY=xxxxxxxx")
    if key.startswith("rf_"):
        sys.exit("That is the PUBLISHABLE key (rf_...). Use the PRIVATE key from Settings > API Keys.")
    ws = get(API + "/", key).get("workspace")
    print(f"Key OK - workspace: {ws}\n")
    projects = (get(f"{API}/{ws}", key).get("workspace") or {}).get("projects", [])
    for p in projects:
        pid = p.get("id", "").split("/")[-1]
        info = get(f"{API}/{ws}/{pid}", key)
        pr = info.get("project", {})
        name = pr.get("name") or p.get("name") or pid
        typ = pr.get("type") or p.get("type") or "?"
        imgs = pr.get("images") or p.get("images") or "?"
        tag = USED.get(pid, "")
        print(f"PROJECT  {name}   (id: {pid}, type: {typ}, images: {imgs}){'   <-- ' + tag if tag else ''}")
        for v in info.get("versions", []):
            vid = str(v.get("id", "")).split("/")[-1]
            m = v.get("model") or {}
            if m:
                arch = m.get("modelType") or m.get("type") or m.get("architecture") or ""
                print(f"   version {vid}: TRAINED MODEL  id {pid}/{vid}  {arch}  -> {score(m)}")
            else:
                print(f"   version {vid}: model id {pid}/{vid}  (Roboflow's API does not say whether this version was trained)")
        print()


if __name__ == "__main__":
    main()
