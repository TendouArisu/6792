"""
download_urfd.py
================
自动从 URFD 官网下载 RGB 图像序列和标注文件。

只下载我们需要的文件（跳过深度图和加速度计）：
  - fall-{01-30}-cam0-rgb.zip  (30 个)
  - fall-{01-30}-cam1-rgb.zip  (30 个)
  - adl-{01-40}-cam0-rgb.zip   (40 个，ADL 只有 cam0)
  - urfall-cam0-falls.csv      (汇总标注，记录每段跌倒的起止帧)
  - urfall-cam0-adls.csv       (ADL 汇总)
  合计: 100 个 zip + 2 个 csv

用法:
  python download_urfd.py --out-dir datasets/urfd
  python download_urfd.py --out-dir datasets/urfd --cam 0   # 只下 cam0（更快）
"""

import argparse
import os
import sys
import time

import requests

BASE_URL  = "http://fenix.ur.edu.pl/mkepski/ds/data/"
CHUNK     = 1024 * 512   # 512 KB per chunk


def _sizeof_fmt(num):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def download_file(url, dest_path, retries=3, timeout=30):
    """
    下载单个文件，支持断点续传和重试。
    返回 True=成功，False=跳过（已存在），raises=失败。
    """
    # 检查已存在且完整（有 Content-Length 则比较大小）
    if os.path.exists(dest_path):
        local_size = os.path.getsize(dest_path)
        if local_size > 0:
            print(f"  [缓存] {os.path.basename(dest_path)}  ({_sizeof_fmt(local_size)})")
            return False

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    fname = os.path.basename(dest_path)

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            if resp.status_code == 404:
                print(f"  [404]  {fname}  (服务器上不存在，跳过)")
                return False
            resp.raise_for_status()

            total = int(resp.headers.get('content-length', 0))
            downloaded = 0
            t0 = time.time()

            with open(dest_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=CHUNK):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100 * downloaded / total
                            speed = downloaded / max(time.time() - t0, 1e-3)
                            print(f"\r  [下载] {fname}  {pct:5.1f}%  "
                                  f"{_sizeof_fmt(downloaded)}/{_sizeof_fmt(total)}  "
                                  f"{_sizeof_fmt(speed)}/s     ", end='', flush=True)

            print(f"\r  [完成] {fname}  {_sizeof_fmt(downloaded)}  "
                  f"({time.time()-t0:.1f}s)                    ")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"\r  [重试 {attempt}/{retries}] {fname}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if attempt == retries:
                print(f"  [失败] {fname}  放弃下载")
                return False
            time.sleep(2 ** attempt)

    return False


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--out-dir', default='datasets/urfd',
                   help='下载目标目录（会自动创建 falls/ 和 adl/ 子目录）')
    p.add_argument('--cam', type=int, default=None, choices=[0, 1],
                   help='只下载指定摄像头（0 或 1）。不指定则下载两个。'
                        'cam0 约 800 MB，两个共约 1.5 GB')
    p.add_argument('--falls-only', action='store_true',
                   help='只下载跌倒序列，不下载 ADL')
    p.add_argument('--dry-run', action='store_true',
                   help='只打印将要下载的文件列表，不实际下载')
    args = p.parse_args()

    falls_dir = os.path.join(args.out_dir, 'falls')
    adl_dir   = os.path.join(args.out_dir, 'adl')
    os.makedirs(falls_dir, exist_ok=True)
    os.makedirs(adl_dir,   exist_ok=True)

    tasks = []   # (url, dest_path)

    # ── 汇总标注 CSV（必须下载）────────────────────────────────────────
    for csv_name in ('urfall-cam0-falls.csv', 'urfall-cam0-adls.csv'):
        tasks.append((BASE_URL + csv_name,
                      os.path.join(args.out_dir, csv_name)))

    # ── Fall 序列 RGB zip ──────────────────────────────────────────────
    cams = [0, 1] if args.cam is None else [args.cam]
    for seq in range(1, 31):
        for cam in cams:
            fname = f"fall-{seq:02d}-cam{cam}-rgb.zip"
            tasks.append((BASE_URL + fname,
                          os.path.join(falls_dir, fname)))

    # ── ADL 序列 RGB zip（只有 cam0）──────────────────────────────────
    if not args.falls_only:
        for seq in range(1, 41):
            # ADL 只有 cam0 rgb
            fname = f"adl-{seq:02d}-cam0-rgb.zip"
            tasks.append((BASE_URL + fname,
                          os.path.join(adl_dir, fname)))

    print(f"=== URFD 下载计划 ===")
    print(f"目标目录 : {os.path.abspath(args.out_dir)}")
    print(f"摄像头   : {'cam0 + cam1' if args.cam is None else f'cam{args.cam} 只'}")
    print(f"文件总数 : {len(tasks)}")
    if args.dry_run:
        print("\n[DRY RUN] 将下载以下文件：")
        for url, dest in tasks:
            print(f"  {url}  →  {dest}")
        return

    print("\n开始下载...\n")
    success = skipped = failed = 0
    t_start = time.time()

    for i, (url, dest) in enumerate(tasks, 1):
        print(f"[{i:3d}/{len(tasks)}] ", end='')
        try:
            result = download_file(url, dest)
            if result:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [错误] {url}: {e}")
            failed += 1

    elapsed = time.time() - t_start
    print(f"\n=== 下载完成 ===")
    print(f"  成功: {success}  跳过(已存在): {skipped}  失败: {failed}")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")

    # ── 解压 zip ────────────────────────────────────────────────────────
    import zipfile, glob

    print("\n开始解压 zip 文件...")
    zips = (sorted(glob.glob(os.path.join(falls_dir, '*.zip'))) +
            sorted(glob.glob(os.path.join(adl_dir,   '*.zip'))))

    for zi, zpath in enumerate(zips, 1):
        folder = zpath.replace('.zip', '')
        if os.path.isdir(folder) and len(os.listdir(folder)) > 0:
            print(f"  [{zi:3d}/{len(zips)}] 已解压，跳过: {os.path.basename(zpath)}")
            continue
        print(f"  [{zi:3d}/{len(zips)}] 解压: {os.path.basename(zpath)} ... ", end='', flush=True)
        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                zf.extractall(os.path.dirname(zpath))
            print("完成")
        except Exception as e:
            print(f"失败: {e}")

    print("\n全部完成！目录结构：")
    for item in sorted(os.listdir(args.out_dir)):
        print(f"  {args.out_dir}/{item}")
    print()
    print("下一步运行：")
    print(f"  python build_dataset_urfd.py --urfd-root {args.out_dir}")


if __name__ == '__main__':
    main()
