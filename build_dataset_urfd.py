"""
build_dataset_urfd.py
=====================
Extract keypoints from the UR Fall Detection Dataset (URFD) and save
them as .npz files compatible with build_dataset.py / train_gru.py.

Dataset info
------------
  Name   : UR Fall Detection Dataset
  URL    : http://fenix.ur.edu.pl/mkepski/ds/uf.html
  License: Free for academic research

Expected directory structure (after running download_urfd.py --cam 0):
  <urfd-root>/
    falls/
      fall-01-cam0-rgb/      <- PNG image sequence
      fall-02-cam0-rgb/
      ...
      fall-30-cam0-rgb/
    adl/
      adl-01-cam0-rgb/
      ...
      adl-40-cam0-rgb/
    urfall-cam0-falls.csv    <- per-frame fall annotations
    urfall-cam0-adls.csv     <- ADL annotations (all normal)

Annotation format (urfall-cam0-falls.csv)
-----------------------------------------
  Columns: seq_name, frame_no, label, [sensor features...]
  label = -1  : normal / pre-fall
  label =  1  : falling / post-fall

  We extract: start = first frame with label==1
               end   = last  frame with label==1

Label convention
-----------------
Consistent with v5 training (pre-fall labels):
  label=1 for frames in [start, end+post_fall_frames].
  start/end are stored in .npz so Stage B --include-prefall applies correctly.

Quality filtering
-----------------
- Skip if MoveNet overall mean keypoint score < MIN_OVERALL_KP_SCORE (0.20).
- Skip sequences shorter than MIN_SEQUENCE_FRAMES (20).

Usage
-----
  python build_dataset_urfd.py --urfd-root datasets/urfd

  # Then regenerate CSV:
  python build_dataset.py --skip-extract --recursive --include-prefall \\
      --out-dir ../datasets --csv-name le2i_urfd_aug_prefall.csv
"""

import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

MIN_OVERALL_KP_SCORE = 0.20
MIN_SEQUENCE_FRAMES  = 20
URFD_OUT_SUBDIR      = 'URFD'


# ── MoveNet loader (mirrors build_dataset.py) ────────────────────────────────

def _load_movenet(movenet_weights: str):
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    movenet_dir = os.path.join(script_dir, 'movenet')
    # Resolve weights path to absolute BEFORE chdir
    weights_abs = os.path.abspath(movenet_weights)
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
        cleaned = {k.lstrip('module.').lstrip('model.'): v for k, v in sd.items()}
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
    heatmap = outs[0].cpu().numpy()
    center  = outs[1].cpu().numpy()
    regs    = outs[2].cpu().numpy()
    offsets = outs[3].cpu().numpy()
    Hf, Wf = heatmap.shape[2], heatmap.shape[3]
    stride  = img_size / Hf
    cy, cx  = np.unravel_index(np.argmax(center[0, 0]), center[0, 0].shape)
    kpts = np.zeros((17, 3), dtype=np.float32)
    for k in range(17):
        rx = float(regs[0, 2*k,   cy, cx]);  ry = float(regs[0, 2*k+1, cy, cx])
        kx = int(np.clip(round(cx + rx), 0, Wf-1))
        ky = int(np.clip(round(cy + ry), 0, Hf-1))
        ox = float(offsets[0, 2*k,   ky, kx]); oy = float(offsets[0, 2*k+1, ky, kx])
        kpts[k] = [(kx+ox)*stride*sx, (ky+oy)*stride*sy,
                   float(heatmap[0, k, ky, kx])]
    return kpts


# ── Annotation parsing ───────────────────────────────────────────────────────

def _parse_urfd_falls_csv(csv_path: str) -> dict:
    """
    Parse urfall-cam0-falls.csv.
    Returns {seq_name: {'start': int, 'end': int, 'total': int}}.
    label==1 means fall; label==-1 means normal.
    start = first frame with label==1, end = last frame with label==1.
    """
    from collections import defaultdict
    rows = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            seq, frame_str, label_str = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                rows[seq].append((int(frame_str), int(float(label_str))))
            except ValueError:
                pass

    result = {}
    for seq, frames in rows.items():
        frames.sort(key=lambda x: x[0])
        total = frames[-1][0] if frames else 0
        fall_frames = [f for f, lbl in frames if lbl == 1]
        if fall_frames:
            result[seq] = {
                'start': min(fall_frames),
                'end':   max(fall_frames),
                'total': total,
            }
        else:
            result[seq] = {'start': 0, 'end': 0, 'total': total}
    return result


def _parse_urfd_adl_csv(csv_path: str) -> dict:
    """ADL are all normal sequences: start=0, end=0."""
    from collections import defaultdict
    rows = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                rows[row[0].strip()].append(int(row[1].strip()))
    result = {}
    for seq, frames in rows.items():
        result[seq] = {'start': 0, 'end': 0, 'total': max(frames) if frames else 0}
    return result


# ── Image sequence loader ────────────────────────────────────────────────────

def _load_frames(seq_dir: str):
    """Yield BGR frames sorted by filename."""
    files = sorted(
        glob.glob(os.path.join(seq_dir, '*.png')) +
        glob.glob(os.path.join(seq_dir, '*.jpg'))
    )
    for fpath in files:
        img = cv2.imread(fpath)
        if img is not None:
            yield img


# ── Main processing ──────────────────────────────────────────────────────────

def process_urfd(urfd_root: str, out_npz_dir: str,
                 movenet_weights: str, force: bool = False):
    model, device, img_size, torch_mod = _load_movenet(movenet_weights)
    os.makedirs(out_npz_dir, exist_ok=True)

    # Load annotations
    falls_csv = os.path.join(urfd_root, 'urfall-cam0-falls.csv')
    adls_csv  = os.path.join(urfd_root, 'urfall-cam0-adls.csv')
    fall_ann  = _parse_urfd_falls_csv(falls_csv) if os.path.exists(falls_csv) else {}
    adl_ann   = _parse_urfd_adl_csv(adls_csv)  if os.path.exists(adls_csv)  else {}
    print(f'[INFO] Fall annotations loaded: {len(fall_ann)} sequences')
    print(f'[INFO] ADL  annotations loaded: {len(adl_ann)} sequences')

    stats = {'processed': 0, 'skipped_qc': 0, 'skipped_cache': 0, 'errors': 0}

    for split, ann_dict, split_dir in [
        ('falls', fall_ann, os.path.join(urfd_root, 'falls')),
        ('adl',   adl_ann,  os.path.join(urfd_root, 'adl')),
    ]:
        if not os.path.isdir(split_dir):
            print(f'[WARN] {split_dir} not found, skipping {split}')
            continue
        print(f'\n=== URFD {split} ===')

        seq_dirs = sorted(d for d in glob.glob(os.path.join(split_dir, '*'))
                          if os.path.isdir(d))

        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)  # e.g. fall-01-cam0-rgb

            # Match annotation: strip '-cam0-rgb' suffix to get 'fall-01'
            ann_key = seq_name.replace('-cam0-rgb', '').replace('-cam1-rgb', '')
            ann = ann_dict.get(ann_key, {'start': 0, 'end': 0, 'total': 0})
            start, end = ann['start'], ann['end']

            tag = 'fall  ' if (start > 0 and end > 0) else 'normal'
            video_id = f'URFD_{seq_name}'
            out_npz  = os.path.join(out_npz_dir, f'{seq_name}.npz')

            if os.path.exists(out_npz) and not force:
                stats['skipped_cache'] += 1
                continue

            frames = list(_load_frames(seq_dir))
            if not frames:
                print(f'  [SKIP] {seq_name}  (no images)')
                stats['errors'] += 1
                continue

            print(f'  {seq_name:<40s}  [{tag} start={start} end={end}]  '
                  f'{len(frames)} frames ... ', end='', flush=True)

            if len(frames) < MIN_SEQUENCE_FRAMES:
                print(f'SKIP (too short: {len(frames)} frames)')
                stats['skipped_qc'] += 1
                continue

            # Run MoveNet
            t0 = time.time()
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
            if mean_score < MIN_OVERALL_KP_SCORE:
                print(f'SKIP (low kp score: {mean_score:.3f})')
                stats['skipped_qc'] += 1
                continue

            np.savez_compressed(
                out_npz,
                keypoints=kp_seq,
                video_id=video_id,
                start=start,
                end=end,
                malformed=False,
                source_dir=seq_dir,
            )
            print(f'OK  {time.time()-t0:.1f}s  kp_mean={mean_score:.3f}')
            stats['processed'] += 1

    print(f'\n{"="*60}')
    print(f'URFD processing complete.')
    print(f'  Processed  : {stats["processed"]}')
    print(f'  Skipped QC : {stats["skipped_qc"]}')
    print(f'  Cached     : {stats["skipped_cache"]}')
    print(f'  Errors     : {stats["errors"]}')
    print(f'  Output dir : {out_npz_dir}')


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--urfd-root', required=True,
                   help='Path to URFD root (contains falls/ adl/ and CSV files).')
    p.add_argument('--out-dir', default='datasets/raw_keypoints',
                   help='Parent of the URFD/ output sub-folder.')
    p.add_argument('--weights', default='movenet/output/movenet.pth',
                   help='Path to MoveNet .pth weights.')
    p.add_argument('--force', action='store_true',
                   help='Re-process sequences that already have .npz files.')
    args = p.parse_args()

    if not os.path.isdir(args.urfd_root):
        print(f'[ERROR] URFD root not found: {args.urfd_root}')
        sys.exit(1)
    if not os.path.isfile(args.weights):
        print(f'[ERROR] MoveNet weights not found: {args.weights}')
        sys.exit(1)

    out_npz_dir = os.path.join(args.out_dir, URFD_OUT_SUBDIR)
    process_urfd(args.urfd_root, out_npz_dir, args.weights, args.force)

    print()
    print('Next: regenerate CSV with pre-fall labels:')
    print('  python build_dataset.py --skip-extract --recursive --include-prefall \\')
    print('      --out-dir ../datasets --csv-name le2i_urfd_aug_prefall.csv')


if __name__ == '__main__':
    main()
