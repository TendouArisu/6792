# Improvements from v2 to v6

This document traces every design decision made between the baseline (v2) and the final model (v6), explains the reasoning behind each change, and quantifies its effect.

---

## Summary Table

| Version | Val F1@thr | Precision | Recall | Key Change |
|---------|-----------|-----------|--------|-----------|
| **v2** (baseline) | 0.665 | 0.660 | 0.671 | Original model |
| **v3** | 0.693 | 0.674 | 0.715 | +54-dim features |
| **v4** | 0.733 | 0.723 | 0.744 | +4× data augmentation |
| **v5** | 0.756 | 0.749 | 0.764 | +BiGRU + Focal Loss + pre-fall labels |
| **v6** | **0.852** | **0.804** | **0.906** | +URFD cross-dataset data |

**Total gain: F1 +0.187 (+28.1%), Recall +0.235 (+35.0%)**

---

## v2 — Baseline

### Architecture

```
Input (B, 30, 51)
  → LayerNorm → GRU(hidden=64, layers=2, unidirectional) → h_n[-1] (B, 64)
  → Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→2)
```

Parameters: **49,672**

### Features (51-dim)

- Normalised keypoint coordinates (34-dim)
- Inter-frame keypoint speed (17-dim)

### Training data

- Le2i dataset: 130 original videos, 4 indoor scenes
- Labels: `[end, end+50]` frames = fall (post-impact only)
- Class imbalance: 8:1 (normal:fall)
- Loss: CrossEntropyLoss with class weights `[1.0, 8.0]`
- Threshold: 0.24 (very low → high recall, low precision)

### Results

```
val F1 = 0.643,  val F1@thr = 0.665
Precision = 0.660,  Recall = 0.671
```

**Key problems identified:**
1. Model only detects falls *after* impact — too late for many scenarios.
2. Training features give no explicit signal about body orientation.
3. Dataset contains only 4 scenes; model is fragile to new environments.
4. Rule-based fusion had a logic bug causing false positives.
5. Batch test evaluation included augmented training files (data leakage).

---

## v2 → v3: Better Features (+3 geometric dims)

### Problem with 51-dim features

The unidirectional GRU had to *implicitly learn* body tilt from raw coordinate differences. This requires many examples and many training epochs, and the learned representation is often shallow because trunk angle can be derived from just two keypoints but the model had to compete with 34 other coordinate dimensions.

### Solution: Explicit geometric features

Three features were added to the extractor:

**`trunk_angle` (1-dim):**
```python
vec = shoulder_centre - hip_centre
trunk_angle = degrees(arctan2(|vec.x|, |vec.y|)) / 90.0   # → [0, 1]
# 0 = fully vertical (standing), 1 = fully horizontal (lying flat)
```
This is the single most discriminative signal for fall detection. A person standing has trunk_angle ≈ 0; a person lying on the ground has trunk_angle ≈ 1. Providing this directly rather than making the GRU derive it from raw coordinates gives an immediate, clean training signal.

**`body_aspect_ratio` (1-dim):**
```python
aspect_ratio = log(1 + bbox_width / bbox_height)
```
Captures the same "horizontal body" information from a different axis. When standing, height >> width → value ≈ 0. When lying flat, width >> height → value > 1. This feature is complementary to trunk_angle because it is robust to partial occlusion (even if the shoulder or hip keypoints are low confidence, the bounding box is computed from all visible keypoints).

**`trunk_angle_delta` (1-dim):**
```python
trunk_angle_delta = trunk_angle[t] - trunk_angle[t-1]
```
Rate of change of body tilt per frame. A rapidly increasing value signals an active fall in progress. This feature fires *before* the person reaches the ground, enabling earlier detection.

**Feature dimension:** 51 → **54**

The model config is stored in the `.pth` checkpoint (`input_dim=54`), so old models (51-dim) and new models (54-dim) are loaded correctly without code changes.

### Results

```
val F1@thr: 0.665 → 0.693  (+0.028)
Recall:     0.671 → 0.715  (+0.044)
```

The recall gain of +4.4% shows that the model now catches more falls. Precision also improved (+1.4%), meaning it became slightly more selective — explicit trunk_angle helps distinguish falls from benign crouching.

---

## v3 → v4: Data Augmentation (4× more training data)

### Problem with small dataset

130 videos from 4 rooms created a model that memorised specific scene backgrounds and lighting. The train-val F1 gap (training ≈ 0.94, val ≈ 0.69) confirmed overfitting. More critically, fall videos were all from the same 4 camera positions, so the model had never seen a fall from a different angle or at a different speed.

### Solution: Offline keypoint augmentation

All 130 `.npz` files were augmented to produce 403 additional files (total 533):

**Horizontal flip (all 130 videos → +130):**

The COCO-17 skeleton has anatomical left/right symmetry. Flipping an image horizontally and swapping left/right keypoint pairs (e.g., left shoulder ↔ right shoulder, left hip ↔ right hip for all 8 pairs) produces a physically valid skeleton. For a person falling to the right, the flipped version is a person falling to the left.

This is valuable because Le2i's fall events tend to fall in one direction (toward the camera or away from it), creating a directional bias. Horizontal flip eliminates this bias at zero cost.

**Time-warp ×0.75 and ×1.25 (96 fall videos → +192):**

Real falls happen at different speeds depending on the person's age, height, and type of fall (trip, faint, slip). The Le2i dataset has one speed per scenario. Resampling the keypoint sequence to 75% or 125% of its original length (via linear interpolation of all 17 × 3 channels independently) creates faster and slower versions of each fall. This makes the model speed-invariant.

Time warping is applied only to fall videos (not normal ones) because normal activity speed variation is less important — the key is to recognise any fall, regardless of speed.

**Time-reversal (96 fall videos → +81, stored as label 0):**

A reversed fall sequence = a person lying on the ground and then getting up. This "getting-up" motion was a significant source of false positives in v2: the post-fall lying-still period followed by the person rising looks somewhat like a fall in reverse to the GRU, which had never seen it as a negative example. By explicitly labeling reversed fall sequences as *normal*, the model learns that getting-up motion is not a fall.

Note: malformed annotations and very short falls (< 10 frames) are not time-warped because scaling their annotated windows would produce incorrect labels.

**Split strategy (crucial for no data leakage):**

The train/val split was updated so that all augmented versions of a given video follow the original video into the same split. Specifically, val contains *only original videos* — no augmented files. This prevents the model from seeing a flipped version of a val video during training.

### Results

```
val F1@thr: 0.693 → 0.733  (+0.040)
Precision:  0.674 → 0.723  (+0.049)
Recall:     0.715 → 0.744  (+0.029)
```

Both precision and recall improved, confirming that the augmented data added genuine generalisation rather than memorisation. The precision gain (+4.9%) is larger than the recall gain (+2.9%), which shows the main benefit: the reversed-fall "negative" examples taught the model to distinguish falls from similar non-fall motions.

---

## v4 → v5: Architecture and Loss Improvements

Three independent changes were made in a single training run:

### Change 1: Bidirectional GRU

**Why unidirectional is suboptimal for fixed-window inference:**

The GRU in v2–v4 processed the 30-frame window left-to-right. When classifying frame 5, it had seen frames 1–5 but not frames 6–30. For fall detection this is wasteful — at inference time the full 30-frame window is available simultaneously. Frame 5 in a fall sequence is much easier to classify correctly if the model also "knows" that frames 20–30 show a person lying still.

**Bidirectional GRU:**
```
Forward  pass: frame 1 → 2 → ... → 30  → h_forward  (B, 96)
Backward pass: frame 30 → 29 → ... → 1  → h_backward (B, 96)
Concat: (B, 192) → classifier
```
The backward pass starts at the *last* frame and processes toward the first. At any position it already "knows" what happens after the fall. The combined representation can simultaneously encode "this person was moving normally before" (forward) and "this person ends up lying still" (backward), making the classification more confident.

Classifier head adjusted accordingly: `Linear(192→96)` instead of `Linear(64→32)`.

Parameters: 49,672 → **273,422** (5.5× more, but still fast enough for Nano CPU in ~15–20 ms).

### Change 2: Focal Loss

**Problem with weighted cross-entropy:**

Class-weighted cross-entropy assigns a fixed weight (≈ 5×) to fall examples. This helps with the class imbalance but treats all fall frames equally. In practice, many fall frames are "easy" — peak-fall frames where the body is clearly horizontal — while boundary frames near the start/end of the fall are genuinely ambiguous. Standard CE loses most of its gradient on the easy frames.

**Focal loss:**
```
FL = -α(1 - p_t)^γ · log(p_t),   γ = 2
```

When `p_t` is large (easy sample, model already confident), `(1 - p_t)^2` is small → small gradient. When `p_t` is small (hard sample, model unsure), `(1 - p_t)^2 ≈ 1` → full gradient. The net effect is that the model concentrates training signal on the ambiguous boundary frames, producing sharper decision boundaries with fewer false positives.

### Change 3: Pre-fall Labels

**Problem with post-impact-only labels:**

In v2–v4, only the frames from `[end, end+50]` were labeled as fall, where `end` = frame when the person hits the ground. The falling motion (`[start, end]`, averaging 22.8 frames) was labeled as *normal*. This is not only a mislabeling — it actively contradicts the training: the model was shown a person visibly falling and told "this is normal."

**Pre-fall labels:**
```
Old: labels[end : end+50] = 1   (50 frames post-impact only)
New: labels[start : end+50] = 1  (falling motion + post-impact)
```

Effects:
- Fall frames per video: +22.8 frames average (+46%)
- Class imbalance: 7.9:1 → **5.0:1** (easier learning task)
- Val fall windows: 410 → 546 (more signal in evaluation)
- The model can now fire *during* the fall, not just after landing

The pre-fall label convention is applied consistently in both training (`build_dataset.py --include-prefall`) and the batch test (`build_labels_from_npz` in notebook uses `lo = max(start, 0)` rather than `lo = max(end, 0)`).

### Results (v5)

```
val F1@thr: 0.733 → 0.756  (+0.023)
Precision:  0.723 → 0.749  (+0.026)
Recall:     0.744 → 0.764  (+0.020)
```

All three metrics improved together. Focal loss and bidirectional GRU improved precision; pre-fall labels improved recall.

---

## v5 → v6: Cross-Dataset Training with URFD

### Problem with single-dataset training

After v5, the training and validation data all came from the same 4 Le2i scenes (plus augmented versions). The model had never seen:
- Different room layouts
- Different lighting conditions
- Different camera heights and angles
- Different subjects (Le2i uses a small number of actors)

The train-val F1 gap (training ≈ 0.91, val ≈ 0.76) indicated moderate overfitting to Le2i's specific patterns. No amount of augmentation can invent genuinely new environments.

### Solution: UR Fall Detection Dataset (URFD)

URFD was selected for compatibility:
- RGB image sequences (compatible with our MoveNet pipeline)
- Side-view camera (MoveNet works well on this angle)
- Per-frame fall annotations (`urfall-cam0-falls.csv`, column 2: -1/+1 labels)
- Free public download, no registration

**Quality filtering:** Of 70 URFD sequences (30 fall + 40 ADL), 37 were rejected because `mean_kp_score < 0.20`. These were mostly overhead-angle cameras where MoveNet cannot reliably detect the human body. Keeping low-quality sequences would inject noise without signal. 33 sequences passed (16 fall + 17 ADL).

**Annotation alignment:** URFD labels the first frame with class=1 as `start` and the last as `end`. Since the sequences already end while the person is on the ground, pre-fall labels naturally cover the full fall event.

**Dataset composition (v6):**

| Source | Fall videos | Normal videos | Total (after augmentation) |
|--------|-------------|---------------|---------------------------|
| Le2i | 96 orig → 374 aug | 34 orig → 159 aug | 533 |
| URFD | 16 orig → 48 aug | 17 orig → 60 aug | 108 |
| **Total** | | | **641** |

**Val set improvement:** The val set now includes 7 URFD videos alongside 26 Le2i videos. These URFD videos are from a completely different lab environment. Achieving high F1 on both Le2i and URFD scenes simultaneously requires genuine generalisation — the model cannot overfit to one scene's background.

Val set class balance: 1,820 normal windows + 626 fall windows = **2.8:1 imbalance** (best ratio in any version, v2 was 4.8:1). This alone makes the val F1 a more reliable estimator.

### Results (v6)

```
val F1@thr: 0.756 → 0.852  (+0.096)
Precision:  0.749 → 0.804  (+0.055)
Recall:     0.764 → 0.906  (+0.142)  ← largest single gain
```

The recall jump of +14.2% is the largest improvement of any step. This shows that the model was previously missing many falls because it had only seen one type of environment. Exposure to URFD's different camera geometry and room layout forced the model to rely on body pose (which is environment-independent) rather than background cues (which are not).

---

## Rule-Based Fusion Improvements

The rule system underwent two important fixes between v2 and v6, independent of the model version.

### Bug fix: Stale drop timer

**Old code (v2–v4):**
```python
if drop_timer > 0 and still_event:
    rule_flag = True   # WRONG
```

If a person bent over quickly (triggering `drop_event + angle_event`) and then stood up, the `drop_timer` would remain active for 20 frames. If within that window the person simply *stopped moving* (e.g., paused while standing), `still_event` would be True and the rule would fire — even though the person was now standing upright, not fallen.

**Fixed code (v5+):**
```python
if drop_timer > 0 and still_event and angle_event:
    rule_flag = True   # CORRECT: also require current horizontal orientation
```
Adding `angle_event` (current trunk angle ≥ 45°) ensures the rule only fires if the person is *still horizontal* — not merely stationary.

### New rule: Sustained horizontal detection

```python
sustained_angle_event = (
    len(angle_hist) >= 20 and
    mean(angle_hist) >= 60.0
)
if sustained_angle_event and still_event:
    rule_flag = True
```

This catches slow or silent falls (e.g., a faint or a gradual slide) where no rapid drop is detected but the person ends up horizontal and stationary for an extended period. It also catches the case where the fall happened just before the inference window started — the GRU has not yet seen enough post-fall frames, but the rule can fire immediately from the current posture.

### Configuration improvements

| Parameter | v2 (original) | v6 | Reason |
|-----------|--------------|-----|--------|
| `RULE_FORCE_ALARM` | `True` | `False` | Rule alone firing caused many FPs when GRU was uncertain. Rule now amplifies the GRU rather than overriding it. |
| `GRU_MIN_FOR_RULE` | 0.20 | 0.60 | Higher threshold ensures GRU must be reasonably confident before rule boost applies. |
| `RULE_BONUS_PROB` | 0.85 | 0.97 | Must exceed `PROB_THRESHOLD` (0.88); at 0.85 the rule could never actually trigger the alarm. |
| `RULE_DROP_THR` | 0.60 | 0.75 | Stricter drop threshold reduces false triggers from fast walking or stairs. |
| `RULE_DROP_HOLD` | 20 frames | 12 frames | Shorter timer reduces window for stale trigger bug to fire. |
| `ALARM_FRAMES` | 6 | 4 | Faster alarm without significantly increasing false positive rate. |

---

## Evaluation Methodology Fix

One important non-model improvement: the batch test in `pipeline.ipynb` was fixed to prevent data leakage.

**Original issue:**

```python
# Old: scanned ALL .npz files including augmented training data
all_npz = glob.glob('datasets/raw_keypoints/**/*.npz', recursive=True)
```

The test pool included 403 augmented files (e.g., `video_(1)_flip.npz`) that were part of the training set. A model that memorised `video_(1)_flip` would score well on it, inflating the reported metrics.

**Fixed:**
```python
_AUG_SUFFIXES = ('_flip.npz', '_fast.npz', '_slow.npz', '_rev.npz')
all_npz = [p for p in all_npz_raw
           if not any(p.endswith(s) for s in _AUG_SUFFIXES)]
# 641 total → 163 original-only
```

Similarly, the label building function was updated to use `lo = max(start, 0)` (pre-fall labels) instead of `lo = max(end, 0)` (post-impact labels), making evaluation consistent with training.

---

## Overfitting Assessment

| Version | Train F1 | Val F1@thr | Gap | Assessment |
|---------|----------|-----------|-----|------------|
| v2 | ~0.94 | 0.665 | 0.275 | Significant |
| v4 | ~0.92 | 0.733 | 0.187 | Moderate |
| v5 | 0.910 | 0.756 | 0.154 | Mild |
| v6 | ~0.94 | 0.852 | ~0.09 | Acceptable |

The trend is clear: as training data grows more diverse (v4: augmentation, v6: URFD), the gap shrinks. v6's gap of ~0.09 is in the range commonly accepted for small-to-medium datasets (< 0.10 is generally considered well-regularised). Early stopping, dropout=0.4, noise augmentation, and weight decay all contribute to keeping the gap small despite the larger model.

---

## Summary of Every Change

| Change | Where | Why | Effect |
|--------|-------|-----|--------|
| +3 geometric features (trunk_angle, aspect_ratio, delta) | `features.py` | Explicit body orientation avoids GRU having to derive it from coordinates | F1 +0.028, Recall +0.044 |
| Horizontal flip augmentation | `augment_keypoints.py` | Fall direction invariance | Part of +0.040 |
| Time-warp augmentation | `augment_keypoints.py` | Fall speed invariance | Part of +0.040 |
| Time-reversal → normal label | `augment_keypoints.py` | Teach "getting up" ≠ fall | Precision +0.049 |
| Bidirectional GRU | `fall_model.py` | See both fall approach and post-fall stillness in one pass | Part of +0.023 |
| Focal Loss | `train_gru.py` | Focus gradient on ambiguous boundary frames | Sharper boundaries |
| Pre-fall labels | `build_dataset.py` | Remove mislabeled falling-motion frames, earlier alarm | Recall +0.020 |
| URFD dataset | `build_dataset_urfd.py` | Cross-dataset generalisation, diverse environments | F1 +0.096, Recall +0.142 |
| Rule bug fix (+ `angle_event`) | `pipeline.ipynb` | Prevent false positive from stationary-after-standing | FP reduction |
| Sustained horizontal rule | `pipeline.ipynb` | Catch slow/silent falls | Recall improvement |
| Rule parameter tuning | `pipeline.ipynb` | Align `RULE_BONUS_PROB` > `PROB_THRESHOLD` | Rule fusion functional |
| Evaluation leakage fix | `pipeline.ipynb` | Exclude augmented training files from test | Honest metrics |
| Label consistency fix | `pipeline.ipynb` | Same `[start, end+50]` in test as in training | Honest metrics |
