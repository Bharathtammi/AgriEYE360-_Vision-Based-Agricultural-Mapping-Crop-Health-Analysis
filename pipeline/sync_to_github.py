"""
Copy the current AgriEYE 360 work into the GitHub repo clone and commit it.

  python sync_to_github.py            # copy + show what changed (no commit)
  python sync_to_github.py --commit   # copy + commit (local only)
  python sync_to_github.py --push     # copy + commit + push to GitHub
  python sync_to_github.py --push --tag v0.5 --note "clean_w model"   # ... and mark this commit as a named VERSION
                                      #     (tag + line in CHANGELOG.md; go back any time: git checkout v0.5)

Copied: the files listed below (code, docs, main results, the two small trained models) and ALL workflow
outputs (agrieye_output -> data/agrieye_output, sample_results -> data/sample_results).
Never copied: raw photos, Roboflow/training datasets, sam3.pt, files over 95 MB, and any file with a hard-coded API key.
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
# whole folders copied as they are (all workflow outputs); files over MAX_MB are skipped
TREES = [
    ("agrieye_output", "data/agrieye_output"),
    ("sample_results", "data/sample_results"),
]
# inside the TREES: never copied (photo copies, training tiles, backups, unfinished runs) - keeps the repo small
EXCLUDE_DIRS = {"pov_all", "variants", "img", "cv"}                 # any folder with this name
EXCLUDE_PREFIX = ("data_big_before_rebuild_", "data_clean_before_", "labels_before_")
EXCLUDE_NAMES = {"last.pt", "sam3_with_duplicates_backup.jsonl"}
DATASET_PARENTS = {"pov_all_labels", "boll_detector"}                         # their images/ folders = photo copies


def excluded(rel):
    parts = rel.parts
    if any(x in EXCLUDE_DIRS or x.startswith(EXCLUDE_PREFIX) for x in parts[:-1]) or rel.name in EXCLUDE_NAMES:
        return True
    if parts[0] == "pov_all_labels" and "previews" in parts[:-1]:              # 1 drawn preview per view (mostly copies)
        return True
    return "images" in parts[:-1] and parts[0] in DATASET_PARENTS


def unfinished_run(f):
    """A training run (runs/<name>/) is uploaded only when it FINISHED: train_visit.py writes experiment.json at the end."""
    for d in f.parents:
        if d.parent.name == "runs":
            return not (d / "experiment.json").exists()
    return False


KEY = re.compile(r"""(api_key|apikey|token|secret|password)\s*[=:]\s*["'][A-Za-z0-9_\-]{12,}["']""", re.I)
TEXT = {".py", ".json", ".bat", ".md", ".txt", ".csv", ".html", ".yaml"}
MAX_MB = 95          # GitHub refuses single files over 100 MB


def git(*a, check=True):
    r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"git {' '.join(a)} failed:\n{r.stderr}")
    return r.stdout.strip()


def copy_one(f, out, copied):
    if not f.is_file() or (f.suffix == ".py" and f.name in SKIP_PY):
        return
    st = f.stat()
    if st.st_size > MAX_MB * 2**20:
        print(f"  SKIP (over {MAX_MB} MB): {f}"); return
    if f.suffix in TEXT:
        hit = KEY.search(f.read_text(encoding="utf-8", errors="ignore"))
        if hit:
            sys.exit(f"STOP: {f} looks like it contains a key ({hit.group(1)} = ...). Nothing committed.")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.stat().st_size != st.st_size or out.stat().st_mtime < st.st_mtime - 1:
        shutil.copy2(f, out)
    copied.append(out)


def copy_all():
    copied = []
    for src, dst in FILES:
        matches = sorted(SRC.glob(src)) if "*" in src else [SRC / src]
        for f in matches:
            copy_one(f, REPO / (dst + f.name if dst.endswith("/") else dst), copied)
    for src, dst in TREES:
        for f in sorted((SRC / src).rglob("*")):
            if excluded(f.relative_to(SRC / src)) or unfinished_run(f):
                continue
            copy_one(f, REPO / dst / f.relative_to(SRC / src), copied)
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--tag", help="name this version, e.g. v0.5 (needs --commit or --push)")
    ap.add_argument("--note", default="", help="one line for CHANGELOG.md with --tag")
    ap.add_argument("-m", "--message", default=f"Update AgriEYE 360 pipeline and results ({date.today():%d %b %Y})")
    a = ap.parse_args()
    if not (REPO / ".git").exists():
        sys.exit(f"Repo clone not found: {REPO}")
    if a.tag and git("tag", "-l", a.tag):
        sys.exit(f"Version {a.tag} already exists - pick a new name.")
    copied = copy_all()
    if a.tag and (a.commit or a.push):
        cl = REPO / "CHANGELOG.md"
        old = cl.read_text(encoding="utf-8") if cl.exists() else "# Changelog - named versions\n\n"
        head, _, rest = old.partition("\n\n")              # newest version goes on top, under the title
        cl.write_text(f"{head}\n\n- **{a.tag}** ({date.today():%d %b %Y}): {a.note or a.message}\n{rest}", encoding="utf-8")
    git("add", "pipeline", "models", "results", "data", ".gitignore", "README.md", *(["CHANGELOG.md"] if (REPO / "CHANGELOG.md").exists() else []))
    status = git("status", "--short")
    print(f"Copied/checked {len(copied)} files into {REPO}")
    lines = status.splitlines()
    print("\n".join(lines[:40]) + (f"\n... and {len(lines) - 40} more" if len(lines) > 40 else "")
          if lines else "No changes since the last commit.")
    if not (a.commit or a.push):
        if status:
            print("\nNothing committed. Run again with --commit (local) or --push (local + GitHub).")
        return
    if status:
        git("commit", "-m", a.message)
        print(git("log", "--oneline", "-1"))
    if a.tag and (a.commit or a.push):
        git("tag", "-a", a.tag, "-m", a.note or a.message)
        print(f"Version {a.tag} = {git('rev-parse', '--short', 'HEAD')}")
    if a.push:
        git("fetch", "origin", check=False)
        ahead = git("rev-list", "--count", "@{u}..HEAD", check=False) if git(
            "rev-parse", "--abbrev-ref", "@{u}", check=False) else "1"
        if a.tag:
            subprocess.run(["git", "-C", str(REPO), "push", "origin", a.tag])
        if ahead in ("", "0"):
            print("GitHub is already up to date."); return
        print(f"Pushing {ahead} commit(s) to GitHub ...")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        r = subprocess.run(["git", "-C", str(REPO), "push", "origin", branch])
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
