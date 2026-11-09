"""
Copy the current AgriEYE 360 work into the GitHub repo clone and commit it.

  python sync_to_github.py            # copy + show what changed (no commit)
  python sync_to_github.py --commit   # copy + commit (local only)
  python sync_to_github.py --push     # copy + commit + push to GitHub

Only the files listed below are copied (code, docs, small results, the two small trained models).
Never copied: photos, crops, datasets, sam3.pt, Street View tours, and any file with a hard-coded API key.
Before committing, every copied text file is scanned for API keys; one hit stops everything.

Repo clone: AgriEYE360-_Vision-Based-Agricultural-Mapping-Crop-Health-Analysis\ (github.com/Bharathtammi/...)
"""
import argparse
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC / "AgriEYE360-_Vision-Based-Agricultural-Mapping-Crop-Health-Analysis"

# scripts that hold a hard-coded key or are old experiments: never copied
SKIP_PY = {"test_cotton_workflow.py", "import base64.py", "test_cotton_base64.py",
           "ssh_agent.py", "ss-agent_4pov.py", "ssh_12pov.py"}

# (source, destination in repo); a source ending in "/*.ext" copies every match
FILES = [
    ("*.py", "pipeline/"),
    ("agrieye_config.json", "pipeline/agrieye_config.json"),
    ("run_pipeline.bat", "pipeline/run_pipeline.bat"),
    ("README.md", "pipeline/README.md"),
    ("AgriEYE360_project_guide.pdf", "pipeline/AgriEYE360_project_guide.pdf"),
    ("AgriEYE st.Viwer/agrieye-field-viewer.html", "pipeline/viewer/agrieye-field-viewer.html"),
    ("disease_autolabel/models/disease_cls/weights/best.pt", "models/disease_cls_v2_best.pt"),
    ("disease_autolabel/models/leaf_seg/weights/best.pt", "models/leaf_seg_best.pt"),
    ("agrieye_output/results/growth.csv", "results/growth.csv"),
    ("agrieye_output/results/detections.csv", "results/detections.csv"),
    ("agrieye_output/results/plots/*.png", "results/plots/"),
    ("sample_results/boll_figures/00_open_vs_closed.jpg", "results/boll_figures/00_open_vs_closed.jpg"),
    ("sample_results/sam3_seg_check/report.txt", "results/sam3_seg_check/report.txt"),
    ("sample_results/sam3_seg_check/per_view.csv", "results/sam3_seg_check/per_view.csv"),
    ("sample_results/sam3_seg_check/example_P09_0731_E.jpg", "results/sam3_seg_check/example_P09_0731_E.jpg"),
    ("sample_results/roboflow_boll_test/report.txt", "results/roboflow_boll_test/report.txt"),
    ("sample_results/roboflow_boll_test/mistakes.jpg", "results/roboflow_boll_test/mistakes.jpg"),
    ("disease_autolabel/metrics/00_results_grid.png", "results/disease_metrics/00_results_grid.png"),
    ("disease_autolabel/metrics/dashboard.png", "results/disease_metrics/dashboard.png"),
]
KEY = re.compile(r"""(api_key|apikey|token|secret|password)\s*[=:]\s*["'][A-Za-z0-9_\-]{12,}["']""", re.I)
TEXT = {".py", ".json", ".bat", ".md", ".txt", ".csv", ".html", ".yaml"}
MAX_MB = 50


def git(*a, check=True):
    r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"git {' '.join(a)} failed:\n{r.stderr}")
    return r.stdout.strip()


def copy_all():
    copied = []
    for src, dst in FILES:
        if "*" in src:
            matches = sorted(SRC.glob(src))
        else:
            matches = [SRC / src]
        for f in matches:
            if not f.is_file() or (f.suffix == ".py" and f.name in SKIP_PY):
                continue
            if f.stat().st_size > MAX_MB * 2**20:
                print(f"  SKIP (over {MAX_MB} MB): {f}"); continue
            out = REPO / (dst + f.name if dst.endswith("/") else dst)
            if f.suffix in TEXT:
                hit = KEY.search(f.read_text(encoding="utf-8", errors="ignore"))
                if hit:
                    sys.exit(f"STOP: {f} looks like it contains a key ({hit.group(1)} = ...). Nothing committed.")
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists() or out.read_bytes() != f.read_bytes():
                shutil.copy2(f, out)
            copied.append(out)
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("-m", "--message", default=f"Update AgriEYE 360 pipeline and results ({date.today():%d %b %Y})")
    a = ap.parse_args()
    if not (REPO / ".git").exists():
        sys.exit(f"Repo clone not found: {REPO}")
    copied = copy_all()
    git("add", "pipeline", "models", "results", ".gitignore", "README.md")
    status = git("status", "--short")
    print(f"Copied/checked {len(copied)} files into {REPO}")
    print(status or "No changes since the last commit.")
    if not (a.commit or a.push) or not status:
        if status:
            print("\nNothing committed. Run again with --commit (local) or --push (local + GitHub).")
        return
    git("commit", "-m", a.message)
    print(git("log", "--oneline", "-1"))
    if a.push:
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        r = subprocess.run(["git", "-C", str(REPO), "push", "origin", branch])
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
