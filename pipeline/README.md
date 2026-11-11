# AgriEYE 360: cotton field pipeline

AgriEYE 360 turns 360° photos of a cotton plot into results for each plant:
- open bolls and closed bolls
- leaf health (healthy or diseased leaves, % diseased)
- growth from one visit to the next
- a Street View style tour of the field

*Last updated: 9 Nov 2026*

---

## 1. Status and results

| Stage | Status | Result |
|---|---|---|
| Import 360 photos | ✅ Done | 35 unique panoramas from 6 data folders (765 files; the rest were copies). 5 visits: 22 Jul (1 photo, no plant ID), 31 Jul, 4 Aug, 6 Aug, 20 Aug 2026 |
| Plant IDs (stations) | ✅ Done, checked 6 Nov | 14 plants (P01–P13). Repeat shots removed from the analysis. At most 1 photo per plant per visit (30 photos used) |
| Slice into 4 views (N/E/S/W) | ✅ Done | 1440 px, 90° views, north-aligned. Analysis: 120 views. Training data: 140 views (all 35 photos) |
| Leaf crops (SAM 3) + manual sorting | ✅ Done | diseased 1,259 · healthy 5,424 · rejected 7,231 |
| Disease classifier v2 (YOLO11n-cls) | ✅ Trained 6 Nov | 89.2% accuracy on 835 field test leaves, ROC AUC 0.944. Only 22% on outside close-up photos (does not generalise) |
| Leaf finder (YOLO11n-seg) | ✅ Trained 6 Nov | Mask precision 0.69, recall 0.62, mAP50 0.67, mAP50-95 0.40 |
| SAM 3 leaf segmentation (text prompt "leaf") | ✅ Done | 5,085 leaves in 120 views, median score 0.57 |
| SAM 3 leaf outlines, hand check | ✅ Done 9 Nov | 426 outlines on 10 views: **all real leaves** (precision 100%), 423 with a good fit (99.3%). Recall not measured (see 1e) |
| SAM 3 + YOLO v2 disease, whole field | ✅ Done | 4,231 leaves judged. % diseased: 31 Jul 15.3 · 4 Aug 21.4 · 6 Aug 13.2 · 20 Aug 25.6 |
| Boll labels: SAM 3 + hand sorting | ✅ Done 8 Nov | 3,068 boll boxes; 2,970 hand-checked: open 441 · closed 2,465 · not a boll 64 |
| **Boll classifier (Roboflow ViT-Base)** | ✅ Trained 8 Nov | **97.7% on 355 test images** (open recall 91.7%, closed 100%, sky/cloud rejected 51/51) |
| Boll detector (Roboflow, one step), comparison only | ✅ Trained 9 Nov | **mAP@50 43.6%** on 18 test views (open AP 26.8%, closed AP 60.4%). Not used in the analysis (see 1f) |
| 4-POV training datasets for Roboflow | ✅ Done 8 Nov | 5 datasets in `roboflow_export\` (see 1g) |
| Field analysis, report, Street View | ✅ Done 8 Nov | Re-run with new plant IDs, leaf finder, disease v2 and boll labels (hand > Roboflow model > SAM 3) |
| Disease classifier v3 (field + public data) | ⬜ Data ready, not trained | `train_disease_v3.py train` (only when you decide to train) |

### 1a. Field results (growth.csv)

Open bolls per plant (sum of the 4 views):

| Plant | 31 Jul | 4 Aug | 6 Aug | 20 Aug |
|---|---|---|---|---|
| P02 | 0 | 3 | 13 | 27 |
| P03 | 0 | 11 | – | 53 |
| P06 | 0 | 6 | – | 53 |
| P09 | 2 | 2 | 5 | 41 |
| P10 | – | 3 | 1 | 66 |
| P13 | – | – | 15 | 74 |

- **Bolls are opening, not disappearing:** closed bolls fall at the same time (P06 110 → 18, P02 71 → 24, P09 105 → 26).
- **Leaf disease rises by 20 Aug:** P09 34%, P13 31%, P03 29%, P06 26% diseased leaves (31 Jul: about 7–25%).
- **Boll labels used:** 2,557 from your hand sorting, 85 from the Roboflow model; 9 uncertain model answers (`check_bolls`).
- **Caveat:** the 4 Aug photos were taken in evening light, which makes red/orange leaves look diseased; compare 31 Jul, 6 Aug and 20 Aug first.

### 1b. Disease classifier v2: performance

Tested on 835 leaves from photos the model never saw during training. Every visit is included in the test set.

| | Precision | Recall |
|---|---|---|
| Diseased | 85.7% | 82.7% |
| Healthy | 91.0% | 92.7% |

- Confusion matrix: 239 diseased found, 50 missed, 40 healthy wrongly flagged, 506 healthy correct.
- Accuracy by visit: 31 Jul 92% · 4 Aug (evening light) 85% · 6 Aug 94% · 20 Aug 88%.
- **"Needs checking" band:** leaves with a diseased probability between 0.2 and 0.8 get a yellow outline and are counted in `check_leaves`. The model is confident on 74% of leaves, and 94% of those are correct.
- **Can SAM 3 judge disease by itself? No (two tests).**
  - *Test 1, whole-leaf prompts* (`sam3_disease_test.py`, 444 leaves on photo tiles): "diseased leaf" picks almost every leaf (precision 43%, AUC 0.45); best rule 64% accuracy, AUC 0.75.
  - *Test 2, disease-spot prompts* (`sam3_spot_test.py`, all 835 test leaves, each enlarged to ~320 px): SAM 3 segments "brown spot", "leaf spot", "hole in leaf", "yellow patch"; score = share of the leaf covered by spots. Best: all spot prompts summed, AUC 0.691; "brown spot" alone 73% accuracy but finds only 37% of diseased leaves (always saying "healthy" gives 65%). "Leaf spot", "hole" and "yellow patch" are near chance (AUC 0.52-0.57).
  - Why: SAM 3 finds distinct spots, but many diseased leaves are discoloured all over (evenly brown, dried, curled) with no separate spot. Its most confident picks are mostly right (9 of the top 12).
  - Versus v2: 89.2% accuracy, AUC 0.944. So SAM 3 is used to **find** leaves and the trained classifier **judges** them. Files: `sample_results\sam3_spot_test\` (`report.txt`, `examples.jpg`).
- **Outside photos:** v2 scores 22% on public close-up leaf photos. Public CC BY 4.0 datasets (Mendeley t9hgvk2h9p and SAR-CLD-2024 b3jy2p6k8w) are prepared for v3 (2,681 leaf crops).

### 1c. Boll classifier (Roboflow) and the cloud problem

- **Pipeline:** SAM 3 finds every boll (prompts "open cotton boll", "closed cotton boll", "flower"), the boll classifier decides open / closed / not a boll.
- **Why not SAM 3's own label:** of the boxes SAM 3 called "open", only about half were open; "flower" was almost never a flower (red closed bolls of this red-leaf variety).
- **Hand sorting:** `sort_bolls_grid.py` shows 30 crops per page with SAM 3's guess filled in; you only click the wrong ones. Your sorting changed SAM 3's answer on 339 boxes.
- **Cloud vs cotton:** an early Roboflow model mixed up white clouds and open cotton. Fix: 383 sky/cloud patches (smooth, away from any boll) added as `not_boll`, plus no brightness augmentation.
- **Model:** Roboflow ViT-Base (ImageNet-21k), 224×224, flip + ±15° rotation, 3×, early stopping (0.16 credits). Roboflow id `cotton-boll-open-or-close-3frah/3`.
- **Test (355 images never used in training, `eval_roboflow_boll.py`):**

| Class | Precision | Recall | Test images |
|---|---|---|---|
| open_boll | 95.7% | 91.7% | 48 |
| closed_boll | 97.7% | 100% | 252 |
| not_boll | 100% | 92.7% | 55 |

- The 8 mistakes are bolls just starting to crack, tiny crops and one cloud-filled crop; 3 had low confidence (0.43–0.56). Answers below 0.6 are counted as `check_bolls`.
- **Limitation:** only 4 test images were real non-sky "not a boll" objects (all 4 wrong); open bolls are rare (48 in test).
- On all 3,068 field crops the model agrees with your hand labels 95.4% (includes training crops, so this is a consistency check, not a test score).

### 1d. Which models are used (final choice)

Every task is done in **two steps**: SAM 3 finds the objects, then a trained classifier judges each one.

| Task | Step 1: find | Step 2: judge | Score | Trained where |
|---|---|---|---|---|
| Cotton bolls | SAM 3 (ready-made, not trained) | Boll classifier, open / closed / not a boll | 97.7% | Roboflow website |
| Leaves and disease | SAM 3 / leaf finder (YOLO11n-seg) | Disease classifier v2, healthy / diseased | 89.2% | Local machine |

- Roboflow models: boll classifier (used) and boll detector (comparison only, see 1f).
- Local models: leaf finder `disease_autolabel\models\leaf_seg\weights\best.pt` and disease v2 `disease_autolabel\models\disease_cls\weights\best.pt`. `disease_cls_v1_backup\` is the older v1 of the same model. Disease v3 is not trained.

### 1e. SAM 3 leaf outlines: hand check

`sam3_seg_check.py` showed 10 views (spread over all 4 visits) with every SAM 3 leaf outline (prompt "leaf", conf 0.3).

- **Precision: 426 of 426 outlines (100%) were judged to be real leaves.** 423 had a good fit (99.3%); 3 were partial (all in P09, 31 Jul, east view: small dead leaves on the soil). No boll, stem, sky or soil was outlined. A few outlines are weed or grass leaves, not cotton.
- **Two rounds:** a precision round (`answers.json`) and a recall round (`answers_recall.json`, filled outlines, missed leaves to be marked). The report uses the recall round.
- **Recall: not measured.** Leaves with no outline were not marked. A visual check shows SAM 3 **misses some clearly visible leaves**, so recall is below 100%.
- **What this means:** SAM 3 gives a reliable **sample** of leaves on each plant. That is fine for **% diseased** (a ratio). **Leaf counts are not used as a measure of plant size.**
- Files: `F:\Project and completed courses\Agri EYE 360\sample_results\sam3_seg_check\` (`report.txt`, `per_view.csv`, `answers.json`).

### 1f. One-step boll detector (Roboflow): comparison

- **Data:** `1_cotton_boll_detection` (140 views, about 3,000 boll boxes: open 453, closed 2,544; split 94 / 28 / 18 views), project "Cotton boll labelling".
- **Roboflow id:** `cotton-bolls-field-view-trbz1/1`. Training graphs: mAP@50 about 35% by epoch 40, about 41% (validation) after about 320 epochs; losses flat after about 30 epochs.
- **Test (18 views never used in training, 308 hand-checked boxes, `eval_roboflow_detect.py`):** Roboflow's free plan shows no metrics, so they were computed from the hosted model's answers (IoU 0.5).

| Class | Boxes | AP@50 | AP@50:95 | Precision (conf 0.5) | Recall (conf 0.5) |
|---|---|---|---|---|---|
| open_boll | 50 | 26.8% | 11.0% | 56.2% | 18.0% |
| closed_boll | 258 | 60.4% | 32.4% | 75.5% | 43.0% |
| **All (mAP)** | 308 | **43.6%** | **21.7%** | 73.6% | 39.0% |

- **Confusion matrix (conf 0.5, IoU 0.5):** of 308 bolls, 122 got a box and 120 of those had the right class (only 2 open bolls called closed, none the other way); 186 were missed (39 of 50 open, 147 of 258 closed); 41 boxes were drawn where no boll was labelled. So R2 mostly **misses** bolls rather than mixing up open and closed. Pictures: `sample_results\confusion_matrices\` (`confusion_matrix_bolls.py`, no API key needed).
- **Tiling did not help:** cutting each view into 2×2 tiles gave mAP@50 28.8% (more false closed bolls), so whole views are better for this model.
- **What it misses:** mostly small, far-away bolls; it finds the large bolls well. Open bolls are found least (9 of 50 at conf 0.5), against 91.7% open recall for the two-step classifier.
- Files: `F:\Project and completed courses\Agri EYE 360\sample_results
oboflow_boll_detect_test\` (`report.txt`, `preview_*.jpg`, `predictions_*.csv`).
- **Why it is lower:** most bolls are only 20–40 px in a 1440 px view (smaller still at 640 px input); closed bolls are green on green leaves; only 140 photos (28 for validation, so the curve is noisy); the labels come from SAM 3 boxes, so real bolls that SAM 3 missed count as false detections.
- **Conclusion:** with this little data, **two steps (SAM 3 + classifier, 97.7%) work much better than one detector**. The analysis uses the two-step method.

### 1g. Roboflow datasets (all built from the 4-POV views)

Folder: `F:\Project and completed courses\Agri EYE 360\roboflow_export\`

The zip copies were deleted on 9 Nov 2026 to save space (0.9 GB); `make_roboflow_datasets.py` and `make_pov_dataset.py export` rebuild them, and Roboflow also accepts a whole folder upload.

| Folder | Roboflow project type | Contents |
|---|---|---|
| `1_cotton_boll_detection\` (upload copy: `UPLOAD_cotton_bolls_labelled\`) | Object Detection | 140 views, 3,004 boll boxes (2,906 hand-checked); split 94 / 28 / 18 views |
| `2_cotton_boll_classification\` | Classification (single-label) | open 441, closed 2,465, not_boll 64 + 383 sky; split train 2,328 / valid 670 / test 355 |
| `3_leaf_segmentation\` | Instance Segmentation | 140 views, about 11,200 leaf outlines |
| `4_leaf_disease_field\` | Classification | Field leaves cut from the views: diseased 816, healthy 2,263 |
| `5_leaf_disease_field_outside\` | Classification | Dataset 4 + 2,681 public leaves |

- **Labels moved from panorama to views:** the hand-labelled leaf outlines were projected with the exact inverse of the slicer's maths (checked visually). 10,108 leaves landed on a view; 3,785 were above or below the 45° view limit.
- **Split without leakage:** grouped by plant + visit (+ view direction for the boll sets; the 4 views do not overlap). Repeat shots always stay with their twin. The boll sets are balanced so open bolls are also 70 / 20 / 10.
- **Roboflow tips learned:** choose "Use Existing Values" for the split (never the random slider); Roboflow needs a test split of at least 2 images; the hosted API needs the **private** API key (the `rf_…` publishable key gives 401).

### 1h. Plant ID fixes (6 Nov)

- **Repeat shots blanked**, keeping the sharper first shot: P04 6 Aug 12:35 · P07 4 Aug 17:26 · P06 4 Aug 17:19:47 · P10 6 Aug 12:44 (same image as 12:43).
- **P06 split in two.** P12 is the stop on the walk back: 31 Jul 08:05 and 4 Aug 17:13.
- **North-east end assigned by walking order:** P09 first stop, P10 second, P13 third.
- Photos cannot confirm plant identity across visits (feature matching scores about the same for same and different plants). The split relies on: two separate stops on one walk are two plants, and the walking order is the same every visit (confirmed). Treat P06/P12 and P09/P10/P13 as best guesses.

### 1i. Lessons learned

- **Label noise was the main problem** for disease: about 1/3 of the first "diseased" crops were clean leaves in evening light. A strict second review raised diseased recall from 54% to 83%.
- **Hold-key accident:** 662 crops got the same label from a held key; re-checked, and all sorters now ignore a held key.
- **Same photo, two timestamps:** `360 files data\2.jpg` and `6.jpg` are the same pictures as P02/P11 on 31 Jul with EXIF times 30 s apart. Duplicates are matched by image content and the copy with the plant ID is kept.
- **Training images should look like the images the model will see:** all training data now comes from the 4-POV views, not the stretched panoramas.
- **Clouds look like open cotton** at low resolution: add sky examples, avoid brightness augmentation, and tile detection images in Roboflow.
- **A hand check must also look for what is missing:** in the SAM 3 outline check every outline starts green, so "all green" only tells you the outlines are right, not that every leaf was found.
- **Small objects favour two steps:** a one-step boll detector reached 43.6% mAP@50 on test views (open bolls 26.8% AP), while SAM 3 + a crop classifier reached 97.7%.
- **Panoramas:** the TOZO AnyVue 360 compass heading is correct (checked against the sun: east at 08:00, west at 17:00); GPS is only accurate to about 2–3 m.

---

## 2. Commands (Windows cmd)

Always use the venv Python. Open **cmd** and set two shortcuts first:

```bat
set PY="C:\Users\bhara\Downloads\.venv\Scripts\python.exe"
set P=F:\Project and completed courses\Agri EYE 360
```

### 2a. Field pipeline (`agrieye.py`)

**One click:** double-click `F:\Project and completed courses\Agri EYE 360\run_pipeline.bat` (import → stations → slice → autolabel → analyze → report → streetview, skips finished steps, log in `agrieye_output\pipeline_log.txt`).

```bat
%PY% "%P%\agrieye.py" import
%PY% "%P%\agrieye.py" stations
%PY% "%P%\check_stations.py"
%PY% "%P%\agrieye.py" slice
%PY% "%P%\agrieye.py" autolabel
%PY% "%P%\agrieye.py" analyze --use-sam3-labels
%PY% "%P%\agrieye.py" report
%PY% "%P%\agrieye.py" streetview
```

| Command | What it does |
|---|---|
| `import` | Reads GPS, time and heading from the photos and writes `catalog.csv` |
| `stations` | Groups photos into plant IDs (`stations.csv`); your edits are kept unless you add `--force` |
| `check_stations.py` | Picture sheet and map for checking the plant IDs |
| `slice` | Cuts each 360 photo into N/E/S/W views |
| `autolabel` | SAM 3 boll/flower boxes on all views |
| `analyze` | Bolls (hand label > Roboflow model > SAM 3), leaves, % diseased, canopy, growth tables and plots |
| `report` | Annotated views and one summary sheet per plant |
| `streetview` | Tour JSON for the field viewer, one per visit |

### 2b. Bolls: sorting, Roboflow datasets, Roboflow model

```bat
:: 1. SAM 3 boxes for the extra (training-only) views, then cut crops for sorting
%PY% "%P%\boll_extra_views.py"
%PY% "%P%\make_boll_crops.py"

:: 2. Hand sorting: grid of 30, click to fix, Enter = accept page, U = undo
%PY% "%P%\sort_bolls_grid.py"

:: 3. Roboflow datasets 1 + 2 (boll detection, boll classification with sky negatives)
%PY% "%P%\make_roboflow_datasets.py"

:: 4. Roboflow model: set your PRIVATE key in this cmd window first (never save it in a file)
set ROBOFLOW_API_KEY=your_private_key
%PY% "%P%\eval_roboflow_boll.py"
%PY% "%P%\boll_classify_roboflow.py"
```

- `eval_roboflow_boll.py` tests the model on the 355 test images (`--limit 5` for a quick check) → `sample_results\roboflow_boll_test\`.
- `boll_classify_roboflow.py` classifies every SAM 3 boll box (about 0.2 credits, resumable) → `agrieye_output\objects\boll_cls_roboflow.csv`, used by `analyze`.

### 2c. 4-POV training datasets (all models)

```bat
%PY% "%P%\make_pov_dataset.py" all
```
Finds all unique 360 photos in every Data set folder, slices them, moves the hand leaf labels onto the views and writes Roboflow datasets 3–5 (and calls `make_roboflow_datasets.py` for 1–2).

### 2d. Leaves and disease (`pipeline.py` and helpers)

```bat
:: Leaf crops from the panoramas with SAM 3, sorting, reviews
%PY% "%P%\sam3_autolabel.py"
%PY% "%P%\sort_crops.py"
%PY% "%P%\find_clean_diseased.py"
%PY% "%P%\sort_crops.py" --review diseased
%PY% "%P%\find_suspect_healthy.py" --n 500
%PY% "%P%\sort_crops.py" --review healthy

:: Disease classifier and leaf finder (training: only when you decide to train)
%PY% "%P%\build_disease_dataset.py" --max-healthy 2500
%PY% "%P%\pipeline.py" traincls
%PY% "%P%\pipeline.py" leafdata
%PY% "%P%\pipeline.py" trainleaf
%PY% "%P%\model_metrics.py"

:: SAM 3 leaf segmentation of the field, alone or + YOLO disease
%PY% "%P%\sam3_segment_field.py"
%PY% "%P%\sam3_segment_field.py" --classify

:: SAM 3 on any image (text prompt), disease-prompt experiment, model tests
%PY% "%P%\vlm_segment.py" "<image or folder>" --prompt "leaf"
%PY% "%P%\sam3_disease_test.py" report
%PY% "%P%\test_all_models.py"
%PY% "%P%\verify_leaves.py"

:: Disease v3 with public data (data built; train only when you decide)
%PY% "%P%\prepare_external.py"
%PY% "%P%\train_disease_v3.py" build
%PY% "%P%\train_disease_v3.py" train
%PY% "%P%\train_disease_v3.py" compare
```

### 2e. This guide

```bat
%PY% "%P%\make_readme_pdf.py"
```

### Still open

1. Optional: SAM 3 recall. Run `sam3_seg_check.py` again (after renaming `answers.json`) and right-click every clear leaf with no outline.
2. Optional: tiled test of the Roboflow boll detector on the 18 test views (open and closed scored separately).
3. Optional: train the leaf disease model on Roboflow (folder `roboflow_export\5_leaf_disease_field_outside\`) or v3 locally, and compare with v2.

SAM 3 hand check, after sorting:

```bat
%PY% "%P%\sam3_seg_check.py"
%PY% "%P%\sam3_seg_check.py" --report
```

Clear open / closed boll pictures (cyan solid = open, magenta dashed = closed):

```bat
%PY% "%P%\make_boll_figures.py"
%PY% "%P%\make_boll_figures.py" --plant P13
```

---

## 3. Architecture

### Data flow

```text
 360 photos (TOZO AnyVue, EXIF GPS + heading), 6 Data set folders
        │
        ├──► agrieye.py import ─► stations ─► slice (N/E/S/W, 1440 px)
        │                                       │
        │                                       ├─► autolabel: SAM 3 boll boxes ─► make_boll_crops ─► sort_bolls_grid (hand)
        │                                       │                                     │
        │                                       │        make_roboflow_datasets ◄─────┘
        │                                       │              │
        │                                       │              └─► Roboflow ViT boll model ─► boll_classify_roboflow
        │                                       │
        │                                       └─► analyze ◄── bolls: hand > Roboflow model > SAM 3
        │                                             │      ◄── leaf_seg + disease_cls (leaves, % diseased)
        │                                             ├─► report      (annotated views, station sheets, growth plots)
        │                                             └─► streetview  (tour JSON) ─► agrieye-field-viewer.html
        │
        ├──► sam3_autolabel.py ─► leaf crops ─► sort_crops (hand) ─► dataset_cls ─► traincls ─► disease_cls
        │                                   └─► leafdata ─► trainleaf ─► leaf_seg
        │
        └──► make_pov_dataset.py ─► pov_dataset\views (all 35 photos) + leaf labels projected ─► roboflow_export 3-5
```

### Scripts (all in `F:\Project and completed courses\Agri EYE 360\`)

| File | Purpose |
|---|---|
| `agrieye.py` + `agrieye_config.json` | Field pipeline: import, stations, slice, autolabel, refine, traindet, analyze, report, streetview, all |
| `run_pipeline.bat` | One-click field pipeline with log |
| `pipeline.py` | Leaf and disease pipeline: status, presort, leafdata, trainleaf, traincls, prelabel, predict |
| `sam3_autolabel.py` | SAM 3 leaf segmentation of the panoramas into crops and manifest |
| `sort_crops.py`, `find_clean_diseased.py`, `find_suspect_healthy.py`, `score_uncertain.py` | Leaf sorting and review tools |
| `build_disease_dataset.py`, `model_metrics.py` | Disease dataset and performance charts |
| `check_stations.py` | Plant-ID picture sheet, map and warnings |
| `make_boll_crops.py`, `sort_bolls.py`, `sort_bolls_grid.py`, `check_flowers.py` | Boll crops, sorters (one-by-one and grid), flower check sheets |
| `boll_extra_views.py` | SAM 3 boll boxes for the training-only views (22 Jul photo) |
| `make_pov_dataset.py` | All 35 photos → 4-POV views, leaf labels projected, Roboflow datasets 3–5 |
| `make_roboflow_datasets.py`, `export_roboflow_bolls.py` | Roboflow datasets 1–2, 70/20/10 split, sky negatives |
| `eval_roboflow_boll.py` | Tests the Roboflow boll model on the test split |
| `eval_roboflow_detect.py` | Tests the Roboflow boll detector (R2) on the 18 test views: mAP, precision, recall per class |
| `list_roboflow_models.py` | Lists every project and version in the Roboflow workspace |
| `confusion_matrix_bolls.py` | Confusion matrices of R1 and R2 from saved test answers (no key, no credits) |
| `boll_classify_roboflow.py` | Classifies every boll box with the Roboflow model for `analyze` |
| `sam3_segment_field.py` | SAM 3 leaf segmentation of the field (`--classify` adds YOLO disease) |
| `vlm_segment.py` | SAM 3 text-prompt segmentation of any image |
| `sam3_disease_test.py` | Experiment: can SAM 3 judge disease by itself (no) |
| `sam3_spot_test.py` | Experiment 2: SAM 3 segments disease spots on enlarged leaves (AUC 0.69, no) |
| `sam3_seg_check.py` | Hand check of SAM 3 leaf outlines (good / partial / wrong, missed leaves) → precision and recall |
| `make_boll_figures.py` | Open / closed boll pictures for the thesis (cyan solid / magenta dashed) |
| `prepare_external.py`, `train_disease_v3.py` | Public data and disease v3 (field + outside) |
| `test_all_models.py`, `verify_leaves.py`, `review_leaf_sorting.py` | Model tests and label checks |
| `make_readme_pdf.py` | Builds `AgriEYE360_project_guide.pdf` from this README |
| `sam3.pt` | SAM 3 weights (3.4 GB; do not put in GitHub) |
| `AgriEYE st.Viwer\agrieye-field-viewer.html` | Street View viewer that loads the tour JSON |

### Input data

| Location | Contents |
|---|---|
| `F:\Project and completed courses\Agri EYE 360\AgriEye 360 field data\Data set\overall dataset\` | The 34 panoramas used for the analysis |
| `...\Data set\360 files data\`, `fourth iteration\`, `Second 100 data sets\`, `Third new data set\` | Other copies; add the 22 Jul photo for training |
| `F:\Project and completed courses\Agri EYE 360\external_data\` | Public CC BY 4.0 leaf datasets (downloads, extracted, crops) |

### `F:\Project and completed courses\Agri EYE 360\agrieye_output\` (field results)

| Location | Contents |
|---|---|
| `...\agrieye_output\catalog.csv`, `stations.csv`, `stations_backup_auto.csv` | Photos, **plant ID for each photo** (edit `station`), automatic IDs before the fix |
| `...\agrieye_output\station_check\` | Plant-ID sheet and GPS map |
| `...\agrieye_output\povs\P01\` … `P13\`, `_skipped\` | 4 views per photo used in the analysis; views of blanked repeat shots |
| `...\agrieye_output\objects\sam3_labels.jsonl`, `sam3_labels_extra.jsonl` | SAM 3 boll boxes (analysis views; training-only views) |
| `...\agrieye_output\objects\boll_cls_roboflow.csv`, `boll_cls_vs_hand.txt` | Roboflow model answer for every boll box; agreement with hand labels |
| `...\agrieye_output\boll_crops\open\`, `closed\`, `not_boll\`, `unsorted\`, `manifest.csv` | Hand-sorted boll crops |
| `...\agrieye_output\pov_dataset\` | Training views of all 35 photos, `catalog.csv`, `split.csv`, `leaves.jsonl`, `check_projection.jpg` |
| `...\agrieye_output\sam3_segmentation\` | SAM 3 leaf outlines of the field |
| `...\agrieye_output\sam3_yolo_disease\` | SAM 3 leaves judged by disease v2 (`summary.csv`) |
| `...\agrieye_output\results\detections.csv`, `growth.csv` | **Results per view and per plant + visit** |
| `...\agrieye_output\results\plots\` | Growth charts (open/closed bolls, leaves, % diseased, canopy, check columns) |
| `...\agrieye_output\results\povs\`, `station_sheets\`, `leaf_health\` | Annotated views, one sheet per plant, leaf-health pictures |
| `...\agrieye_output\streetview\tour_2026-07-31.json` … `tour_2026-08-20.json` | Street View tours, one per visit |

### `F:\Project and completed courses\Agri EYE 360\disease_autolabel\` (leaf and disease work)

| Location | Contents |
|---|---|
| `...\disease_autolabel\manifest.jsonl` | One line per leaf crop: source photo, box, outline, score |
| `...\disease_autolabel\crops\diseased\`, `healthy\`, `rejected\` | Hand-sorted leaf crops (1,259 / 5,424 / 7,231) |
| `...\disease_autolabel\dataset_cls\`, `dataset_cls_v3\`, `dataset_leaf\`, `dataset_seg\` | Classifier v2 data, v3 data (field + outside), leaf finder tiles, healthy/diseased segmentation |
| `...\disease_autolabel\models\disease_cls\weights\best.pt` | **Disease classifier v2 (current)**; v1 backup in `disease_cls_v1_backup\` |
| `...\disease_autolabel\models\leaf_seg\weights\best.pt` | **Leaf finder** |
| `...\disease_autolabel\metrics\` | Classifier charts (`00_results_grid.png`, `dashboard.png`, …) |

### Other result folders

| Location | Contents |
|---|---|
| `F:\Project and completed courses\Agri EYE 360\roboflow_export\` | The 5 Roboflow datasets (folders + zips), `2_sky_negatives_preview.jpg` |
| `F:\Project and completed courses\Agri EYE 360\sample_results\roboflow_boll_test\` | Roboflow boll model test: `report.txt`, `mistakes.jpg`, `predictions.csv` |
| `F:\Project and completed courses\Agri EYE 360\sample_results\sam3_seg_check\` | SAM 3 leaf outline check: `report.txt`, `per_view.csv`, `answers.json` |
| `F:\Project and completed courses\Agri EYE 360\sample_results\boll_figures\` | Open / closed boll pictures (`00_open_vs_closed.jpg`, one file per view) |
| `F:\Project and completed courses\Agri EYE 360\sample_results\` | Sample outputs, model tests, SAM 3 experiments |
| `F:\Project and completed courses\Agri EYE 360\Results\` | Workspace links to the segmentation folders and models (junctions: delete with `rmdir`, never `Remove-Item -Recurse`) |

---

## 4. Environment

- **PC:** Windows 11, Intel i5-1155G7, 16 GB RAM, CPU only (no GPU).
- **Python:** `C:\Users\bhara\Downloads\.venv\Scripts\python.exe` (ultralytics with SAM 3, OpenCV, matplotlib, inference-sdk for Roboflow).
- **Roboflow:** workspace `bharaths-workspace-3fhlw`, free plan (public projects, about 10 credits a month). Keep the private API key out of files, screenshots and GitHub.
- **Before long runs:** set sleep to "never" so the PC stays awake.
