AgriEYE 360 - sample leaf disease results (6 Nov 2026)

01_field_view_P02_20Aug_N_leaf_health.jpg  Plant P02, 20 Aug, north view. Leaf-finder + disease model.
                                           Green = healthy, red = diseased, yellow = needs checking (model unsure).
                                           53 leaves, 13 diseased (24.5%), 17 need checking.
                                           Note: some red outlines are red closed bolls, not leaves
                                           (fix ready in agrieye.py analyze: drop "leaves" on boll boxes).
02_..._per_leaf.csv                        One row per leaf: box, label, p_diseased, needs_check.
03_..._summary.csv                         Totals for that view.
04/05_test_photo_*.jpg                     Earlier test of the disease model on a field photo (full + zoom).
08_leaf_finder_training_curves.png         Leaf-finder training: 60 epochs, mask mAP50 0.67.
09_disease_classifier_dashboard.png        Disease model v2: 835 test leaves, accuracy 89.2%, AUC 0.944.
10_disease_classifier_results_grid.png     Disease model v2 per-epoch metrics.
11_disease_classifier_metrics.csv          Final numbers.
12/13_sample_*_leaves.jpg                  Random hand-sorted diseased / healthy leaf crops (training data).
