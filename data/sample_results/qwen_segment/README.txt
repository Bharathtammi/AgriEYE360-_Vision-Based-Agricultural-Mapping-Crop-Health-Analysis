QWEN-ONLY SEGMENTATION TEST (Qwen2.5-VL-3B-Instruct, zero-shot, no SAM, no training) - 24/25 Nov 2026
Script: qwen_segment.py (Qwen is asked to write each object's outline as polygon points; the polygons are filled = mask)

RESULT: Qwen gave NO segmentation of cotton bolls.
  P01 31 Jul E  (hand: 23 closed)  -> 0 open, 0 closed   (225 s)
  P01 31 Jul N  (hand: 31 closed)  -> 0 open, 0 closed   (231 s)
  -> *_overlay.jpg are the unchanged pictures, *_mask.png are empty, *_qwen.txt shows Qwen answered [].

Extra check on P02 20 Aug N (hand: 10 open, 9 closed) - qwen_answers_P02_20Aug_N.txt:
  polygon outline          -> []  (nothing)
  native box format        -> ONE box over most of the plant, not one per boll  (see *_qwen_box_vs_hand.jpg)
  "how many bolls?"        -> "10 open, 15 closed"  (open count exactly right)

WHY: Qwen is a vision-language model - it answers in text and has no mask output. It was trained to write boxes,
not outlines. Bolls are small (picture shrunk to 896x896 -> a boll is ~20-40 px = 1-3 of Qwen's 28 px patches),
so the 3B model understands and counts the scene but cannot place each boll.

Files: run_log.txt = console output of the 2-picture run.
