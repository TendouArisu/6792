"""
augment_keypoints.py
====================
Augment existing .npz keypoint sequences to expand the training dataset.

Augmentations
-------------
1. horizontal_flip  : mirror all keypoints left↔right around the horizontal
                      centroid. Swaps COCO-17 left/right pairs.
                      Applied to ALL videos (fall + normal).

2. time_warp_fast   : resample sequence to 0.75× original length (faster fall).
                      Fall videos ONLY (needs a valid start/end annotation).

3. time_warp_slow   : resample sequence to 1.25× original length (slower fall).
                      Fall videos ONLY.

4. time_reverse     : reverse the whole sequence.  The reversed video shows the
                      person *getting up*, which is a common false-positive for
                      GRU models.  Stored with start=0, end=0 so it trains as a
                      normal (non-fall) example — helping the model learn that
                      reversed motion is not a fall.
                      Fall videos ONLY.

Quality filters (applied before augmenting)
-------------------------------------------
- Skip malformed annotation files.
- Skip fall period shorter than MIN_FALL_LENGTH frames.
- Skip if mean keypoint score during fall frames < MIN_FALL_KP_SCORE.
- Skip if overall mean keypoint score < MIN_OVERALL_KP_SCORE
  (person mostly out of frame or heavily occluded throughout).

Expected gain
-------------
  130 original videos → ~520-548 entries after augmentation
  (≈ 4× dataset size), with the augmented normal-class bias balanced
  by the flipped normal videos.

Usage
-----
  # Dry run — see what would be created, nothing written:
  python augment_keypoints.py --dry-run

  # Full run (writes new .npz next to originals):
  python augment_keypoints.py

  # Then regenerate the CSV:
  python build_dataset.py --skip-extract --csv-name le2i_keypoints_aug.csv
"""

import argparse
import glob
import os

import numpy as np

# ─── COCO-17 left/right swap for horizontal flip ────────────────────────────
FLIP_PAIRS = [
    (1, 2),   # left_eye  ↔ right_eye
    (3, 4),   # left_ear  ↔ right_ear
    (5, 6),   # left_shoulder ↔ right_shoulder
    (7, 8),   # left_elbow ↔ right_elbow
    (9, 10),  # left_wrist ↔ right_wrist
    (11, 12), # left_hip ↔ right_hip
    (13, 14), # left_knee ↔ right_knee
    (15, 16), # left_ankle ↔ right_ankle
]

# ─── Quality thresholds ──────────────────────────────────────────────────────
# Le2i has naturally low keypoint scores (overhead/angled cameras) but the GRU
# trains fine on them.  We only reject truly degenerate cases:
#   - malformed annotation (no reliable start/end → bad window labels)
#   - fall period too short (< 10 frames → noisy label after time-warp)
#   - zero-score sequences (MoveNet never fired → pure noise)
MIN_FALL_LENGTH      = 10    # fall period must span at least 10 frames
MIN_FALL_KP_SCORE    = 0.02  # mean keypoint score during fall (only rejects score=0)
MIN_OVERALL_KP_SCORE = 0.02  # whole-video mean (only rejects score=0)


# ─── Augmentation primitives ─────────────────────────────────────────────────

def flip_sequence(kp_seq: np.ndarray) -> np.ndarray:
    """
    Horizontal flip of a (T, 17, 3) keypoint sequence.

    The x coordinate of each keypoint is mirrored around the per-frame
    centroid of all *visible* keypoints (score > 0.10).  Using the
    centroid rather than a fixed image width keeps the augmentation
    valid even though we don't have the original image dimensions.

    After mirroring, left/right pairs are swapped so that anatomical
    left still means anatomical left in the mirror image.
    """
    out = kp_seq.copy()
    T = out.shape[0]
    for t in range(T):
        valid = kp_seq[t, :, 2] > 0.10
        if valid.any():
            cx = float(kp_seq[t, valid, 0].mean())
            out[t, :, 0] = 2.0 * cx - kp_seq[t, :, 0]
    # swap anatomical pairs
    for l, r in FLIP_PAIRS:
        out[:, [l, r], :] = out[:, [r, l], :].copy()
    return out


def time_warp(kp_seq: np.ndarray, factor: float) -> np.ndarray:
    """
    Resample sequence to `factor` × original length via linear interpolation.

    factor < 1  → fewer frames (faster motion)
    factor > 1  → more frames (slower motion)

    All 3 channels (x, y, score) are interpolated independently.
    Enforces a minimum output length of 30 (== seq_len) so windows
    can always be formed.
    """
    T = kp_seq.shape[0]
    new_T = max(int(round(T * factor)), 30)
    old_idx = np.arange(T, dtype=np.float64)
    new_idx = np.linspace(0.0, T - 1.0, new_T)
    out = np.zeros((new_T, 17, 3), dtype=np.float32)
    for k in range(17):
        for c in range(3):
            out[:, k, c] = np.interp(new_idx, old_idx, kp_seq[:, k, c].astype(np.float64))
    return out


def scale_annotation(start: int, end: int, T_old: int, T_new: int):
    """Proportionally rescale frame-level fall annotations to the new length."""
    if start <= 0 or end <= 0:
        return 0, 0
    scale = T_new / T_old
    return int(round(start * scale)), int(round(end * scale))


def reverse_sequence(kp_seq: np.ndarray) -> np.ndarray:
    """Return the time-reversed sequence (T, 17, 3) → (T, 17, 3)."""
    return kp_seq[::-1].copy()


# ─── Quality assessment ───────────────────────────────────────────────────────

def _flip_ok(kp_seq: np.ndarray) -> tuple:
    """Flip is valid as long as there are any non-zero keypoints."""
    if float(kp_seq[:, :, 2].max()) < MIN_OVERALL_KP_SCORE:
        return False, "all-zero keypoints"
    return True, "ok"


def _timewarp_ok(kp_seq: np.ndarray, start: int, end: int,
                 malformed: bool) -> tuple:
    """
    Time warp and reversal require a valid fall annotation with enough frames,
    because scaling changes the fall window position — unreliable with bad annotations.
    """
    if malformed:
        return False, "malformed annotation"
    if start <= 0 or end <= 0:
        return False, "no fall annotation"
    fall_len = end - max(start, 0)
    if fall_len < MIN_FALL_LENGTH:
        return False, f"fall too short ({fall_len} frames < {MIN_FALL_LENGTH})"
    fall_frames = kp_seq[max(0, start): min(end + 1, kp_seq.shape[0])]
    if len(fall_frames) > 0:
        fall_mean = float(fall_frames[:, :, 2].mean())
        if fall_mean < MIN_FALL_KP_SCORE:
            return False, f"fall kp score too low ({fall_mean:.3f})"
    return True, "ok"


# ─── Per-file augmentation ────────────────────────────────────────────────────

def augment_file(npz_path: str, dry_run: bool = False,
                 verbose: bool = True) -> dict:
    """
    Generate augmented .npz files for one source .npz.

    Returns a dict with counts: {'created': N, 'skipped': bool}.
    New files are written to the SAME directory as the source.
    """
    data     = np.load(npz_path, allow_pickle=True)
    kp_seq   = data['keypoints'].astype(np.float32)
    start    = int(data['start'])    if 'start'    in data.files else 0
    end      = int(data['end'])      if 'end'      in data.files else 0
    malformed = bool(data['malformed']) if 'malformed' in data.files else False
    video_id = str(data['video_id']) if 'video_id' in data.files else \
               os.path.splitext(os.path.basename(npz_path))[0]

    out_dir = os.path.dirname(npz_path)
    stem    = os.path.splitext(os.path.basename(npz_path))[0]
    T_old   = kp_seq.shape[0]
    created = 0

    def _save(suffix, kp, s, e, mal, vid_suffix):
        nonlocal created
        out_path = os.path.join(out_dir, f"{stem}{suffix}.npz")
        if os.path.exists(out_path) and not dry_run:
            return
        if not dry_run:
            np.savez_compressed(
                out_path,
                keypoints=kp.astype(np.float32),
                start=s, end=e,
                malformed=False,
                video_id=video_id + vid_suffix,
            )
        created += 1

    # 1. Horizontal flip — apply to all videos with any valid keypoints
    flip_ok, flip_reason = _flip_ok(kp_seq)
    if flip_ok:
        kp_flip = flip_sequence(kp_seq)
        _save('_flip', kp_flip, start, end, malformed, '_flip')
    elif verbose:
        print(f"  SKIP  {video_id:<50s}  (flip: {flip_reason})")
        return {'created': 0, 'skipped': True}

    # 2-4. Time-based augmentations — require valid fall annotation
    tw_ok, tw_reason = _timewarp_ok(kp_seq, start, end, malformed)
    if tw_ok:
        # 2. Time-warp fast (0.75×)
        kp_fast = time_warp(kp_seq, 0.75)
        s_f, e_f = scale_annotation(start, end, T_old, kp_fast.shape[0])
        _save('_fast', kp_fast, s_f, e_f, False, '_fast')

        # 3. Time-warp slow (1.25×)
        kp_slow = time_warp(kp_seq, 1.25)
        s_s, e_s = scale_annotation(start, end, T_old, kp_slow.shape[0])
        _save('_slow', kp_slow, s_s, e_s, False, '_slow')

        # 4. Time reversal → stored as normal (start=0,end=0)
        #    Teaches the GRU that "getting up" motion is not a fall.
        kp_rev = reverse_sequence(kp_seq)
        _save('_rev', kp_rev, 0, 0, False, '_rev')

    if verbose:
        is_fall = start > 0 and end > 0
        tag = 'fall  ' if is_fall else 'normal'
        tw_tag = f'+timewarp' if tw_ok else f'  (no timewarp: {tw_reason})'
        dry_tag = '[DRY] ' if dry_run else ''
        print(f"  {dry_tag}OK    {video_id:<50s}  ({tag}) +{created} {tw_tag}")

    return {'created': created, 'skipped': False}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split('\n')[1])
    p.add_argument('--npz-dir', default='datasets/raw_keypoints',
                   help='Root directory of original .npz files (searched recursively).')
    p.add_argument('--dry-run', action='store_true',
                   help='Print what would be created without writing any files.')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress per-file output.')
    args = p.parse_args()

    all_npz = sorted(glob.glob(
        os.path.join(args.npz_dir, '**', '*.npz'), recursive=True))

    # Augmented suffixes we create — never augment augmentations
    aug_suffixes = ('_flip.npz', '_fast.npz', '_slow.npz', '_rev.npz')
    orig_npz = [p for p in all_npz
                if not any(p.endswith(s) for s in aug_suffixes)]

    print(f"Found {len(orig_npz)} original .npz files  "
          f"({len(all_npz) - len(orig_npz)} already-augmented files ignored)")

    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    total_created = 0
    total_skipped = 0
    fall_count    = 0
    normal_count  = 0

    for npz_path in orig_npz:
        result = augment_file(npz_path,
                              dry_run=args.dry_run,
                              verbose=not args.quiet)
        if result['skipped']:
            total_skipped += 1
        else:
            total_created += result['created']
            # Count type
            data = np.load(npz_path, allow_pickle=True)
            s = int(data['start']) if 'start' in data.files else 0
            e = int(data['end'])   if 'end'   in data.files else 0
            if s > 0 and e > 0:
                fall_count += 1
            else:
                normal_count += 1

    print()
    print("=" * 60)
    print(f"Augmentation complete.")
    print(f"  Original videos  : {len(orig_npz)}")
    print(f"    fall           : {fall_count}")
    print(f"    normal         : {normal_count}")
    print(f"    skipped (QC)   : {total_skipped}")
    print(f"  New .npz files   : {total_created}")
    print(f"  Total after aug  : {len(orig_npz) + total_created}")
    if args.dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        print()
        print("Next step — regenerate the CSV including augmented data:")
        print("  python build_dataset.py --skip-extract \\")
        print("      --csv-name datasets/le2i_keypoints_aug.csv \\")
        print("      --out-dir datasets")
        print()
        print("Then retrain:")
        print("  python train_gru.py --csv datasets/le2i_keypoints_aug.csv \\")
        print("      --out models/fall_gru_v4.pth [... same flags as v3 ...]")


if __name__ == '__main__':
    main()
