"""
build_dataset_mcfd.py
=====================
Extract keypoints from the Multiple Cameras Fall Dataset (MCFD) and save
as .npz files compatible with the existing training pipeline.

Dataset info
------------
  Name   : Multiple Cameras Fall Dataset
  Source : Kaggle – soumicksarker/multiple-cameras-fall-dataset
  Content: 24 fall scenarios × 8 cameras = 192 AVI videos
  FPS    : 120 fps,  Resolution: 720×480
  Annotation: data_tuple3.csv  (frame-level fall windows, 31 frames each)

CSV format
----------
  chute, cam, start, end, label
  label 0 = normal window, label 1 = fall window (one per chute+cam)
  start/end = frame indices within the video

Processing strategy
-------------------
For each (chute, cam) pair we:
  1. Find the fall window [fall_start, fall_end] from CSV.
  2. Extract a padded clip: [fall_start - PRE_FRAMES, fall_end + POST_FRAMES].
  3. Run MoveNet on each frame → (T, 17, 3) keypoints.
  4. Store .npz with start/end relative to clip start.

This avoids processing all 1562+ frames; we only read ~250 frames per video.

Label convention (consistent with v6 pre-fall training)
---------------------------------------------------------
  start = fall_start (relative to clip)  → falling motion begins
  end   = fall_end   (relative to clip)  → fall window ends
  Stage B with --include-prefall labels [start : end+50] as fall.

Quality filtering
-----------------
  - Skip if overall mean keypoint score < 0.20.
  - Skip clip if fewer than MIN_FRAMES frames were decoded.

Usage
-----
  python build_dataset_mcfd.py --mcfd-root datasets/mcfd

  # Then regenerate combined CSV:
  python build_dataset.py --skip-extract --recursive --include-prefall \\
      --out-dir ../datasets --csv-name le2i_urfd_mcfd_aug_prefall.csv
"""

import argparse
import csv
import glob
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

PRE_FRAMES       = 100   # frames to include before fall start
POST_FRAMES      = 100   # frames to include after fall end
MIN_FRAMES       = 30    # minimum decoded frames to keep sequence
MIN_KP_SCORE     = 0.20  # whole-clip mean keypoint score threshold
MCFD_OUT_SUBDIR  = 'MCFD'

# cam "55" appears in CSV but the file is cam5.avi – treat as cam5
CAM_ALIAS = {'55': '5'}


# ── MoveNet loader ────────────────────────────────────────────────────────────

def _load_movenet(weights_path: str):
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    movenet_dir = os.path.join(script_dir, 'movenet')
    weights_abs = os.path.abspath(weights_path)
    sys.path.insert(0, movenet_dir)
    old_dir = os.getcwd()
    os.chdir(movenet_dir)
    try:
        import torch
        from lib import init, MoveNet
        from config import cfg
        init(cfg)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MoveNet(num_classes=cfg['num_classes'],
                        width_mult=cfg['width_mult'], mode='train')
        ckpt = torch.load(weights_abs, map_location=device)
        sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        cleaned = {}
        for k, v in sd.items():
            nk = k
            for pfx in ('module.', 'model.'):
                if nk.startswith(pfx):
                    nk = nk[len(pfx):]
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=True)
        model = model.to(device).eval()
        print(f'[OK] MoveNet loaded  device={device}')
        return model, device, cfg['img_size'], torch
    finally:
        os.chdir(old_dir)


def _infer_frame(model, device, img_size, torch_mod, frame_bgr):
    h, w = frame_bgr.shape[:2]
    img  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    blob = np.transpose(img.astype(np.float32), (2, 0, 1))[None]
    tensor = torch_mod.from_numpy(blob).to(device)
    sx, sy = w / img_size, h / img_size
    with torch_mod.no_grad():
        outs = model(tensor)
    hm = outs[0].cpu().numpy();  ctr = outs[1].cpu().numpy()
    reg = outs[2].cpu().numpy(); off = outs[3].cpu().numpy()
    Hf, Wf = hm.shape[2], hm.shape[3]
    stride  = img_size / Hf
    cy, cx  = np.unravel_index(np.argmax(ctr[0, 0]), ctr[0, 0].shape)
    kpts = np.zeros((17, 3), dtype=np.float32)
    for k in range(17):
        rx = float(reg[0, 2*k,   cy, cx]); ry = float(reg[0, 2*k+1, cy, cx])
        kx = int(np.clip(round(cx + rx), 0, Wf-1))
        ky = int(np.clip(round(cy + ry), 0, Hf-1))
        ox = float(off[0, 2*k,   ky, kx]); oy = float(off[0, 2*k+1, ky, kx])
        kpts[k] = [(kx+ox)*stride*sx, (ky+oy)*stride*sy, float(hm[0, k, ky, kx])]
    return kpts


# ── Annotation parsing ────────────────────────────────────────────────────────

def _parse_annotations(csv_path: str) -> dict:
    """
    Returns {(chute_str, cam_str): {'fall_start': int, 'fall_end': int}}.
    Only stores the fall window (label == 1).
    cam '55' is aliased to '5'.
    """
    ann = {}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['label'].strip() != '1.0':
                continue
            chute = str(int(float(row['chute'].strip())))
            cam   = str(int(float(row['cam'].strip())))
            cam   = CAM_ALIAS.get(cam, cam)
            ann[(chute, cam)] = {
                'fall_start': int(float(row['start'].strip())),
                'fall_end':   int(float(row['end'].strip())),
            }
    return ann


# ── Video clip extraction ─────────────────────────────────────────────────────

def _extract_clip(video_path: str, clip_start: int, clip_end: int):
    """
    Read frames [clip_start, clip_end] inclusive from an AVI.
    Returns list of BGR frames (may be shorter if video ends early).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_start = max(0, clip_start)
    clip_end   = min(total - 1, clip_end)

    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    frames = []
    for _ in range(clip_end - clip_start + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ── Main processing ───────────────────────────────────────────────────────────

def process_mcfd(mcfd_root: str, out_npz_dir: str,
                 movenet_weights: str, force: bool = False):
    model, device, img_size, torch_mod = _load_movenet(movenet_weights)
    os.makedirs(out_npz_dir, exist_ok=True)

    csv_path   = os.path.join(mcfd_root, 'data_tuple3.csv')
    video_root = os.path.join(mcfd_root, 'dataset', 'dataset')
    annotations = _parse_annotations(csv_path)
    print(f'[INFO] Loaded {len(annotations)} fall annotations')

    stats = {'processed': 0, 'skipped_qc': 0,
             'skipped_cache': 0, 'errors': 0}

    # Sort by (chute, cam) for reproducibility
    for (chute, cam), ann in sorted(annotations.items(),
                                    key=lambda x: (int(x[0][0]), int(x[0][1]))):
        video_path = os.path.join(video_root, f'chute{int(chute):02d}',
                                  f'cam{cam}.avi')
        if not os.path.isfile(video_path):
            print(f'  [MISS] chute{chute} cam{cam} – file not found: {video_path}')
            stats['errors'] += 1
            continue

        fall_start = ann['fall_start']
        fall_end   = ann['fall_end']
        clip_s     = max(0, fall_start - PRE_FRAMES)
        clip_e     = fall_end + POST_FRAMES

        # relative indices inside the extracted clip
        rel_start = fall_start - clip_s
        rel_end   = fall_end   - clip_s

        video_id = f'MCFD_chute{int(chute):02d}_cam{cam}'
        out_npz  = os.path.join(out_npz_dir, f'{video_id}.npz')

        if os.path.exists(out_npz) and not force:
            stats['skipped_cache'] += 1
            continue

        print(f'  chute{int(chute):02d} cam{cam:<3s}  '
              f'fall=[{fall_start},{fall_end}]  '
              f'clip=[{clip_s},{clip_e}] ... ', end='', flush=True)

        t0 = time.time()
        frames = _extract_clip(video_path, clip_s, clip_e)

        if len(frames) < MIN_FRAMES:
            print(f'SKIP (only {len(frames)} frames decoded)')
            stats['errors'] += 1
            continue

        # Run MoveNet
        kpts_list = []
        try:
            for frame in frames:
                kpts_list.append(_infer_frame(model, device, img_size,
                                              torch_mod, frame))
        except Exception as e:
            print(f'FAIL ({e})')
            stats['errors'] += 1
            continue

        kp_seq = np.stack(kpts_list, axis=0).astype(np.float32)

        # Quality check
        mean_score = float(kp_seq[:, :, 2].mean())
        if mean_score < MIN_KP_SCORE:
            print(f'SKIP (low kp_score={mean_score:.3f})')
            stats['skipped_qc'] += 1
            continue

        np.savez_compressed(
            out_npz,
            keypoints=kp_seq,
            video_id=video_id,
            start=rel_start,
            end=rel_end,
            malformed=False,
            source_video=video_path,
        )
        print(f'OK  {time.time()-t0:.1f}s  '
              f'T={len(frames)}  kp_mean={mean_score:.3f}')
        stats['processed'] += 1

    print()
    print('=' * 60)
    print('MCFD processing complete.')
    print(f'  Processed  : {stats["processed"]}')
    print(f'  Skipped QC : {stats["skipped_qc"]}')
    print(f'  Cached     : {stats["skipped_cache"]}')
    print(f'  Errors     : {stats["errors"]}')
    print(f'  Output dir : {out_npz_dir}')


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mcfd-root', required=True,
                   help='Path to MCFD root (contains data_tuple3.csv + dataset/).')
    p.add_argument('--out-dir', default='datasets/raw_keypoints',
                   help='Parent of MCFD/ output sub-folder.')
    p.add_argument('--weights', default='movenet/output/movenet.pth',
                   help='MoveNet .pth weights path.')
    p.add_argument('--force', action='store_true',
                   help='Re-process already-completed sequences.')
    args = p.parse_args()

    if not os.path.isdir(args.mcfd_root):
        print(f'[ERROR] MCFD root not found: {args.mcfd_root}')
        sys.exit(1)
    if not os.path.isfile(args.weights):
        print(f'[ERROR] Weights not found: {args.weights}')
        sys.exit(1)

    out_npz_dir = os.path.join(args.out_dir, MCFD_OUT_SUBDIR)
    process_mcfd(args.mcfd_root, out_npz_dir, args.weights, args.force)

    print()
    print('Next: regenerate CSV with pre-fall labels:')
    print('  python build_dataset.py --skip-extract --recursive --include-prefall \\')
    print('      --out-dir ../datasets --csv-name le2i_urfd_mcfd_aug_prefall.csv')


if __name__ == '__main__':
    main()
