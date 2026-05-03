# Fall Detection System — v6 (Jetson Nano)

End-to-end real-time fall detection pipeline for Jetson Nano, using MoveNet for pose estimation and a Bidirectional GRU for temporal classification.

```
Camera / Video
     │
     ▼
┌─────────────────────────────┐
│  MoveNet  (TRT FP16 / PyTorch) │   → 17 keypoints (x, y, score)
└─────────────────────────────┘
     │
     ▼
┌──────────────────────────────────┐
│  KeypointFeatureExtractor  (54-d) │   → normalised coords + speed
│    + trunk angle + aspect ratio  │     + geometric features
└──────────────────────────────────┘
     │
     ▼
┌────────────────────────────┐
│  Sliding window (30 frames) │   → (30, 54) sequence buffer
└────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Bidirectional GRU (v6 model) │   → fall_prob ∈ [0, 1]
│   hidden=96 · 2 layers · 273K │
└──────────────────────────────┘
     │
     ▼
┌──────────────────────┐
│  Rule-based Fusion   │   → fused_prob
│  (drop + angle + still) │
└──────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Debounce (4 consecutive frames) │   → alarm
└──────────────────────────────┘
```

---

## Performance (v6)

| Metric | Value |
|--------|-------|
| Val F1 @ optimal threshold | **0.852** |
| Precision | 0.804 |
| Recall | 0.906 |
| Batch test F1 (100 original videos) | **0.818** |
| Optimal threshold | 0.87–0.90 |
| GRU parameters | 273,422 |
| Feature dimension | 54 |

*Evaluation uses only original (non-augmented) videos, with pre-fall labels consistent with training. No augmented training files appear in the test pool.*

---

## Repository Structure

```
movenet_jeston_nano/
├── pipeline.ipynb            # Live inference notebook (webcam / video / .npz)
├── features.py               # KeypointFeatureExtractor (54-dim)
├── fall_model.py             # FallDetectionGRU (BiGRU) + dataset classes
├── train_gru.py              # Training script (FocalLoss, noise augmentation)
├── build_dataset.py          # Stage A/B: extract keypoints → CSV
├── augment_keypoints.py      # Offline data augmentation (flip/warp/reverse)
├── build_dataset_urfd.py     # Keypoint extraction for URFD dataset
├── build_dataset_mcfd.py     # Keypoint extraction for MCFD dataset
├── download_urfd.py          # Auto-download URFD from official site
│
├── models/
│   ├── fall_gru_v6.pth       # ★ Best model — use this for deployment
│   ├── fall_gru_v5.pth
│   ├── fall_gru_v4.pth
│   ├── fall_gru_v3.pth
│   └── fall_gru_v2.pth
│
├── movenet/
│   ├── output/
│   │   ├── pose_fp16.engine  # TensorRT FP16 engine (Jetson Nano)
│   │   └── movenet.pth       # PyTorch weights (PC fallback)
│   └── lib/ config.py ...    # fire717's MoveNet implementation
│
└── datasets/
    ├── le2i_urfd_aug_prefall.csv   # Training CSV (Le2i + URFD, 641 videos)
    ├── raw_keypoints/
    │   ├── Coffee_room_01/   # Le2i original .npz
    │   ├── Home_01/ Home_02/ Coffee_room_02/
    │   └── URFD/             # URFD cam0 .npz
    └── urfd/                 # Raw URFD images + annotation CSVs
```

---

## Pipeline in Detail

### 1. MoveNet — Pose Estimation

MoveNet (fire717's PyTorch implementation, MobileNetV2 backbone) detects 17 COCO keypoints per frame.

**Critical preprocessing:**
```python
# BGR → RGB → resize to 192×192 (non-uniform, NO letterbox)
# Keep float32 in [0, 255] — backbone does x/127.5-1 internally
```
Using letterbox padding or dividing by 255 causes heatmap scores to drop from ~0.8 to ~0.3, breaking detection. On Jetson Nano the TRT FP16 engine (`pose_fp16.engine`) is used instead of PyTorch.

Output: `(17, 3)` array — `[x_px, y_px, confidence]` per keypoint, COCO-17 order.

### 2. Feature Extraction — 54 dimensions

`KeypointFeatureExtractor` converts each `(17, 3)` frame into a `(54,)` vector:

| Slice | Dims | Description |
|-------|------|-------------|
| `[0:34]` | 34 | Normalised (x, y) coords — centred on body centroid, scaled by bounding-box span. Same pose at any distance gives the same features. |
| `[34:51]` | 17 | Inter-frame L2 speed per keypoint (in normalised space). Captures how fast each body part is moving. |
| `[51]` | 1 | **trunk_angle** — angle of shoulder-centre → hip-centre vector from vertical, normalised to [0, 1]. 0 = upright, 1 = fully horizontal. |
| `[52]` | 1 | **body_aspect_ratio** — `log(1 + bbox_width/bbox_height)`. Near 0 when standing, large when lying flat. |
| `[53]` | 1 | **trunk_angle_delta** — frame-to-frame change in trunk angle. Large positive value = rapidly falling. |

The three geometric features (dims 51–53) are the key addition in v5/v6. Without them the GRU had to derive the trunk angle from raw coordinates, requiring more training data and more epochs.

An optional EMA smoothing (`smooth_alpha=0.2`) reduces keypoint jitter from MoveNet without introducing perceptible lag.

### 3. Sliding Window Buffer

```python
feat_buffer = collections.deque(maxlen=30)   # 30 frames ≈ 1.2 s @ 25 fps
```

Every frame appends one 54-d vector. When the buffer is full (30 frames) the BiGRU runs. Inference runs every `INFERENCE_STRIDE` frames (default 1 = every frame).

### 4. Bidirectional GRU Classifier

Architecture:
```
Input (B, 30, 54)
  → LayerNorm(54) → Dropout(0.1)
  → BiGRU(input=54, hidden=96, layers=2, dropout=0.4)
      forward  hidden h_n[-2]  (B, 96)
      backward hidden h_n[-1]  (B, 96)
  → concat → (B, 192)
  → Linear(192→96) → ReLU → Dropout(0.4)
  → Linear(96→2) → softmax
  → P(fall) ∈ [0, 1]
```

**Why bidirectional?** At inference time the full 30-frame window is available. The forward pass captures "person was walking, then fell." The backward pass sees "person is lying still" and traces back to the earlier frames. Together they classify the window more reliably than a unidirectional GRU.

### 5. Rule-Based Fusion

Three geometric rules run in parallel with the GRU, operating directly on raw keypoints:

**Rule 1 — Rapid Drop:**
```
drop_norm = Δhip_y (over 5 frames) / torso_length
if drop_norm ≥ 0.75 AND trunk_angle ≥ 45° → drop_event
```
Detects fast downward motion of the hip centre relative to body size.

**Rule 2 — Sustained Horizontal:**
```
if mean(trunk_angle[-20 frames]) ≥ 60° AND mean_speed ≤ 0.06 → sustained_event
```
Catches slow or silent falls where no rapid drop is detected — person gradually slides or is already on the ground.

**Rule 3 — Post-drop Stillness:**
```
if drop_timer > 0 AND still AND current_angle ≥ 45° → post_drop_event
```
`drop_timer` counts down 12 frames after a drop event. If the person is still horizontal and motionless within that window, a fall is confirmed. The `current_angle ≥ 45°` check was added as a bug fix — the original code fired on any stillness after a drop, even if the person had stood back up.

**Fusion:**
```python
fused_prob = fall_prob   # default = GRU output
if any_rule_fired AND fall_prob ≥ 0.60:
    fused_prob = max(fall_prob, 0.97)
```
Rules alone cannot trigger an alarm (`RULE_FORCE_ALARM = False`). The GRU must agree (≥ 0.60) before the rule boost applies. This prevents spurious alarms from people bending over or sitting down quickly.

### 6. Debounce & Alarm

```python
ALARM_FRAMES = 4    # consecutive frames above threshold to fire
ALARM_HOLD   = 50   # frames alarm stays active after firing (~2 s)
PROB_THRESHOLD = 0.88  # recommended for v6
```

A single frame above threshold is ignored. Four consecutive frames are required to rule out momentary misclassification. Once fired, the alarm holds for 2 seconds to allow downstream systems (SMS, speaker, log) to react.

---

## Training Pipeline

### Data Sources

| Source | Original videos | After augmentation | Notes |
|--------|-----------------|-------------------|-------|
| Le2i Fall Dataset | 130 | 533 | 4 indoor scenes, fixed cameras |
| URFD cam0 | 33 (QC-filtered from 70) | 108 | Lab environment, side view |
| **Total** | **163** | **641** | |

37 URFD sequences were rejected by quality filter (`mean_kp_score < 0.20`) — mainly overhead camera angles that MoveNet cannot reliably process.

### Data Augmentation (`augment_keypoints.py`)

Applied to every original `.npz`:

| Operation | Applied to | Effect |
|-----------|-----------|--------|
| Horizontal flip | All videos | Mirrors keypoints left↔right (COCO-17 pair-swap). Makes the model direction-invariant — fall-left and fall-right look identical. |
| Time-warp ×0.75 | Fall videos | Faster falling motion via linear interpolation. |
| Time-warp ×1.25 | Fall videos | Slower falling motion. |
| Time-reversal | Fall videos | Reversed fall = person getting up. **Stored as normal (label 0).** Teaches the GRU that "getting-up" motion ≠ fall, which was a significant source of false positives. |

**Split strategy:** The train/val split is done at the *base-video* level. All augmented versions of a training video go into train; all augmented versions of a validation video go into val. Val contains **only original, non-augmented videos** — guaranteeing honest evaluation.

### Label Convention — Pre-fall Labels

Classic approach: label only the post-impact resting state `[end, end+50]`.

**v6 approach:** label the entire fall event `[start, end+50]` where `start` = frame when the falling motion begins.

Why this matters:
1. The falling motion (avg. 22.8 frames in Le2i) was previously labeled as *normal*, creating mislabeled training data.
2. With pre-fall labels the model can fire during the fall, not just after impact — earlier alarms.
3. Class imbalance improves from 7.9:1 → 5.0:1, making the learning task easier.

### Loss Function — Focal Loss

```python
FL = -α(1 - p_t)^γ · log(p_t),   γ = 2.0
```

Standard cross-entropy with class weights treats all samples equally (after reweighting). Focal loss additionally down-weights *easy* well-classified normal samples and focuses the gradient on *hard* boundary frames near the fall transition. The net effect is sharper decision boundaries with fewer false positives.

### Training Command (v6)

```bash
python train_gru.py \
    --csv datasets/le2i_urfd_aug_prefall.csv \
    --out models/fall_gru_v6.pth \
    --epochs 80 \
    --batch-size 128 \
    --hidden-dim 96 \
    --bidirectional \
    --num-layers 2 \
    --dropout 0.4 \
    --input-dropout 0.1 \
    --stride 3 \
    --min-pos-frames 3 \
    --use-sampler \
    --threshold-search \
    --threshold-step 0.01 \
    --focal-loss \
    --focal-gamma 2.0 \
    --noise-std 0.02 \
    --early-stop-patience 12 \
    --patience 5 \
    --smooth-alpha 0.2 \
    --device auto
```

---

## Deployment on Jetson Nano

### Files Required

```
models/fall_gru_v6.pth
movenet/output/pose_fp16.engine
```

### Configuration (`pipeline.ipynb` Cell 1)

```python
GRU_PATH       = 'models/fall_gru_v6.pth'
ENGINE_PATH    = 'movenet/output/pose_fp16.engine'
PROB_THRESHOLD = None    # auto-loads 0.75 from checkpoint; tune to 0.88 for best F1
DISPLAY        = False   # Jetson Nano has no desktop GUI by default
SAVE_OUT       = 'output.mp4'   # save annotated video
ALARM_FRAMES   = 4
ALARM_HOLD     = 50
```

### Expected Performance on Jetson Nano

| Component | Backend | Estimated latency |
|-----------|---------|-------------------|
| MoveNet | TRT FP16 (Maxwell GPU) | ~15 ms |
| Feature extraction | CPU | < 1 ms |
| BiGRU inference | CPU | ~15–20 ms |
| Rules + debounce | CPU | < 1 ms |
| **Total** | | **~30–35 ms ≈ 30 FPS** |

Memory footprint: ~900 MB–1.2 GB on the 2 GB shared RAM. Use `opencv-python-headless` to save ~50 MB.

---

## Evaluation Methodology

All reported metrics use:
- **Test pool:** original (non-augmented) `.npz` files only — augmented files are training data and are excluded.
- **Labels:** pre-fall convention `[start, end+50]`, consistent with training.
- **Val/test split:** video-level, stratified by fall presence.

The batch-test cells in `pipeline.ipynb` enforce these constraints automatically.

---

## Dependencies

```
torch >= 2.0
opencv-python >= 4.8
numpy >= 1.24
pandas >= 2.0
scikit-learn >= 1.3
requests           # for download_urfd.py
```

Jetson Nano additionally requires TensorRT 8.2.1 and PyCUDA.

---

## Citation

Le2i dataset: Charfi et al., *Fuzzy Logic and Human Fall Detection*, 2013.
URFD dataset: Kępski & Kwolek, *Fall Detection Using Ceiling-Mounted 3D Depth Camera*, 2014.
MoveNet implementation: fire717, https://github.com/fire717/movenet.pytorch
