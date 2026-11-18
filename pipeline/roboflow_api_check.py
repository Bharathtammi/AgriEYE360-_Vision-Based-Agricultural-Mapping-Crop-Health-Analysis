"""
Why does the Roboflow model find bolls in the web preview but almost none through the API?
Sends ONE view to the model in a few different ways and prints how many boxes each way returns.

  $env:ROBOFLOW_API_KEY="..."   (PowerShell; never saved, never printed)
  python roboflow_api_check.py
"""
import base64
import io
import os
import sys
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "agrieye_output" / "pov_all_labels" / "bolls_dataset_DRAFT_sam3_only" / "images" / "image_20260820_141320_S.jpg"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "bharaths-workspace-3fhlw/bolls_dataset_draft_sam3_only-1-yolo26n-t1"
WORKSPACE, WORKFLOW = "bharaths-workspace-3fhlw", "bollsdatasetdraftsam3only-1-yolo26n-t1-vbollsdatasetdraftsam3only-1-yolo26n-t1-logic"
key = os.environ.get("ROBOFLOW_API_KEY", "").strip().strip('"').strip("'")
if not key:
    sys.exit('Set your key first (PowerShell):  $env:ROBOFLOW_API_KEY="xxxxxxxx"')


def find_preds(obj):
    """Roboflow Workflow outputs are named by the workflow; find the first detections block inside, whatever its name."""
    if isinstance(obj, dict):
        if isinstance(obj.get("predictions"), list):
            return obj
        vals = obj.values()
    elif isinstance(obj, list):
        vals = obj
    else:
        return None
    for v in vals:
        r = find_preds(v)
        if r is not None:
            return r
    return None


def b64(im):
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def show(name, resp):
    if isinstance(resp, Exception):
        print(f"{name:<52} ERROR {str(resp).split('api_key')[0][:150]}"); return
    p = resp.get("predictions", []) if isinstance(resp, dict) else []
    c = sorted((x["confidence"] for x in p), reverse=True)
    print(f"{name:<52} boxes {len(p):>4}   >=50%: {sum(v >= 0.5 for v in c):>4}   top: {[round(v, 2) for v in c[:5]]}"
          f"   image seen as {resp.get('image')}")


def rest(url, im):
    try:
        r = requests.post(url, params={"api_key": key, "confidence": 1, "overlap": 50},
                          data=b64(im), headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=60)
        return r.json() if r.ok else Exception(f"HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        return e


im = Image.open(IMG).convert("RGB")
small = im.resize((640, 640))                    # what the model was trained on (Stretch to 640x640)
print(f"Model {MODEL}\nImage {IMG.name} {im.size}\n")
for host in ("https://serverless.roboflow.com", "https://detect.roboflow.com"):
    show(f"REST {host.split('//')[1]} (full 1440)", rest(f"{host}/{MODEL}", im))
    show(f"REST {host.split('//')[1]} (pre-resized 640)", rest(f"{host}/{MODEL}", small))
try:
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    cl = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=key)
    cl.configure(InferenceConfiguration(confidence_threshold=0.01, iou_threshold=0.5, max_detections=1000,
                                        api_key_transport="header"))
    for name, x in (("inference_sdk serverless (full 1440)", im), ("inference_sdk serverless (pre-resized 640)", small)):
        try:
            show(name, cl.infer(x, model_id=MODEL))
        except Exception as e:
            show(name, e)
    for name, x in (("WORKFLOW (full 1440)", im), ("WORKFLOW (pre-resized 640)", small)):
        try:
            out = cl.run_workflow(workspace_name=WORKSPACE, workflow_id=WORKFLOW, images={"image": x})
            if name.endswith("1440)"):
                print("  workflow output names:", list(out[0].keys()) if out and isinstance(out[0], dict) else type(out))
            show(name, find_preds(out) or Exception("no detections block in the workflow output"))
        except Exception as e:
            show(name, e)
except ImportError:
    print("inference_sdk not installed - skipped")
print("\nThe web preview showed many boxes at 80-98%. The way that matches it is the one to use.")
