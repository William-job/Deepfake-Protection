#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baseline 数据归档脚本

将 baseline 训练数据和评估结果复制到带时间戳的归档目录，
确保后续模型改进实验不会覆盖 baseline 数据。

归档内容:
  - results/baselines/<model>/  (best_model.pt, checkpoint_latest.pt, training.log, results.json)
  - results/table2/             (comparison_results.json, <model>/raw_predictions.npz, <model>/metrics.json)

使用方法:
  python scripts/archive_baselines.py                    # 归档到 archives/baselines_<timestamp>/
  python scripts/archive_baselines.py --output archives/my_archive/  # 指定归档目录
  python scripts/archive_baselines.py --skip_missing     # 跳过不存在的文件（默认会警告但继续）

严谨性:
  - 计算每个文件的 MD5 校验和，写入 manifest.json
  - 归档完成后验证文件数量和总大小
  - manifest.json 记录归档时间、源路径、文件列表、校验和
  - 不删除源文件（仅复制）
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

# baseline 模型列表（manifest 的 "models" 字段使用此列表）
BASELINE_MODELS = ["mesonet", "efficientnet_b0", "efficientnet_b3", "xception"]
# table2 中除 baseline 外还会出现的评估目录（CAR 模型评估结果）
TABLE2_EXTRA_MODELS = ["car_v5"]

# MD5 计算分块大小（8MB，平衡内存占用与读取速度）
MD5_CHUNK_SIZE = 8 * 1024 * 1024

ARCHIVE_REASON = "baseline 数据持久化保存，防止后续模型改进实验覆盖"

# 项目根目录（scripts/ 的父目录），保证脚本从任意工作目录运行都能定位 results/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_md5(filepath):
    """计算文件的 MD5 校验和。

    采用分块读取方式，避免大文件一次性载入内存。
    返回: 16 进制 MD5 字符串
    """
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(MD5_CHUNK_SIZE), b""):
            md5.update(chunk)
    return md5.hexdigest()


def copy_with_checksum(src, dst, with_md5=True):
    """复制单个文件到目标路径，并返回文件大小与（可选）MD5。

    使用 shutil.copy2 保留元数据（mtime 等）。
    参数名刻意使用 with_md5 而非 compute_md5，避免遮蔽同名函数。
    返回: (size_bytes, md5_hex or None)
    """
    dst_dir = os.path.dirname(dst)
    if dst_dir and not os.path.isdir(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, dst)
    size = os.path.getsize(dst)
    md5 = compute_md5(filepath=dst) if with_md5 else None
    return size, md5


def _norm_rel(path):
    """将本机路径分隔符统一为正斜杠，便于 manifest 跨平台可读。"""
    return path.replace(os.sep, "/")


def _record_missing(missing_files, rel_path, skip_missing):
    """记录缺失文件/目录。skip_missing 为 True 时不打印警告。"""
    missing_files.append(rel_path)
    if not skip_missing:
        print("  [警告] 缺失: {}".format(rel_path))


def archive_file(src, dst, rel_path, files_info, missing_files,
                 skip_missing, with_md5):
    """归档单个文件。

    rel_path: 在 manifest 中记录的相对路径（正斜杠分隔）。
    成功复制返回 True，源文件不存在或复制失败返回 False（并记录缺失/警告）。
    """
    if not os.path.isfile(src):
        _record_missing(missing_files, rel_path, skip_missing)
        return False
    try:
        size, md5 = copy_with_checksum(src, dst, with_md5=with_md5)
    except Exception as e:
        # 单个文件复制失败不中断整个归档
        print("  [警告] 复制失败 {}: {}".format(rel_path, e))
        missing_files.append(rel_path + " (复制失败)")
        return False
    files_info.append({"path": rel_path, "size_bytes": size, "md5": md5})
    return True


def archive_dir(src_dir, dst_dir, rel_prefix, files_info, missing_files,
                skip_missing, with_md5):
    """递归复制 src_dir 下所有文件到 dst_dir，保持目录结构。

    src_dir: 源目录绝对路径
    dst_dir: 目标目录绝对路径
    rel_prefix: manifest 中记录的相对路径前缀（例如 "baselines/mesonet"）
    返回: 成功复制的文件数量
    """
    if not os.path.isdir(src_dir):
        _record_missing(missing_files, rel_prefix + "/", skip_missing)
        return 0

    count = 0
    for root, dirs, files in os.walk(src_dir):
        dirs.sort()  # 保证遍历顺序稳定
        for name in sorted(files):
            src_file = os.path.join(root, name)
            rel = os.path.relpath(src_file, src_dir)
            dst_file = os.path.join(dst_dir, rel)
            manifest_path = _norm_rel(os.path.join(rel_prefix, rel))
            if archive_file(src_file, dst_file, manifest_path, files_info,
                            missing_files, skip_missing, with_md5):
                count += 1
    return count


def discover_table2_models(table2_src):
    """发现 table2 目录下的模型子目录。

    合并已知列表（baseline + car_v5）与实际存在的子目录，
    保证既能覆盖预期模型，也能捕获新增的评估目录。
    """
    known = list(BASELINE_MODELS) + list(TABLE2_EXTRA_MODELS)
    models = set(known)
    if os.path.isdir(table2_src):
        for name in os.listdir(table2_src):
            if os.path.isdir(os.path.join(table2_src, name)):
                models.add(name)
    # 保持已知模型优先排序，其余按字母序追加
    ordered = [m for m in known if m in models]
    extras = sorted(m for m in models if m not in known)
    return ordered + extras


def write_manifest(archive_root, files_info, missing_files, source_dirs,
                   models, total_size, total_files):
    """写入 manifest.json，并返回 manifest 字典。"""
    manifest = {
        "archive_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "archive_reason": ARCHIVE_REASON,
        "source_dirs": source_dirs,
        "models": models,
        "files": files_info,
        "total_size_bytes": total_size,
        "total_files": total_files,
    }
    # missing_files 为扩展字段：记录未能归档的预期项，便于审计
    if missing_files:
        manifest["missing_files"] = missing_files
    manifest_path = os.path.join(archive_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def verify_archive(archive_root, manifest):
    """验证归档完整性：检查每个文件存在、大小匹配，并核对总数与总大小。

    返回: 全部通过返回 True，否则返回 False。
    """
    ok = True
    checked = 0
    size_sum = 0
    for entry in manifest.get("files", []):
        rel = entry["path"]
        dst = os.path.join(archive_root, rel)
        if not os.path.isfile(dst):
            print("  [校验失败] 文件不存在: {}".format(rel))
            ok = False
            continue
        actual = os.path.getsize(dst)
        if actual != entry["size_bytes"]:
            print("  [校验失败] 大小不匹配 {}: 记录={}, 实际={}".format(
                rel, entry["size_bytes"], actual))
            ok = False
        checked += 1
        size_sum += actual

    if checked != manifest.get("total_files", 0):
        print("  [校验失败] 文件数量不匹配: manifest 记录 {}, 实际校验 {}".format(
            manifest.get("total_files", 0), checked))
        ok = False
    if size_sum != manifest.get("total_size_bytes", 0):
        print("  [校验失败] 总大小不匹配: manifest 记录 {}, 实际 {}".format(
            manifest.get("total_size_bytes", 0), size_sum))
        ok = False
    return ok


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline 数据归档脚本：将 baseline 训练/评估数据复制到带时间戳的归档目录。")
    parser.add_argument(
        "--output", default=None,
        help="归档输出目录（默认 archives/baselines_<YYYYMMDD_HHMMSS>/）")
    parser.add_argument(
        "--skip_missing", action="store_true",
        help="跳过不存在的文件且不打印警告（默认会打印警告但继续）")
    parser.add_argument(
        "--no_md5", action="store_true",
        help="跳过 MD5 计算（加速大文件归档，默认计算）")
    return parser.parse_args()


def main():
    args = parse_args()
    with_md5 = not args.no_md5

    # 确定归档输出目录
    if args.output:
        archive_root = os.path.abspath(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_root = os.path.join(PROJECT_ROOT, "archives",
                                    "baselines_{}".format(timestamp))
    os.makedirs(archive_root, exist_ok=True)

    baselines_src = os.path.join(PROJECT_ROOT, "results", "baselines")
    table2_src = os.path.join(PROJECT_ROOT, "results", "table2")

    print("=" * 60)
    print("Baseline 数据归档")
    print("=" * 60)
    print("归档目录: {}".format(archive_root))
    print("源目录:")
    print("  baselines: {}".format(baselines_src))
    print("  table2:    {}".format(table2_src))
    print("MD5 校验:   {}".format("开启" if with_md5 else "关闭"))
    print("-" * 60)

    files_info = []
    missing_files = []

    # 1. 归档 baseline 训练数据: results/baselines/<model>/ -> baselines/<model>/
    print("[1/3] 归档 baseline 训练数据...")
    for model in BASELINE_MODELS:
        src = os.path.join(baselines_src, model)
        dst = os.path.join(archive_root, "baselines", model)
        rel_prefix = "baselines/{}".format(model)
        n = archive_dir(src, dst, rel_prefix, files_info, missing_files,
                        args.skip_missing, with_md5)
        print("  {}: {} 个文件".format(model, n))

    # 2. 归档 table2 评估数据
    print("[2/3] 归档 table2 评估数据...")
    # 2a. comparison_results.json
    cmp_src = os.path.join(table2_src, "comparison_results.json")
    cmp_dst = os.path.join(archive_root, "table2", "comparison_results.json")
    if archive_file(cmp_src, cmp_dst, "table2/comparison_results.json",
                    files_info, missing_files, args.skip_missing, with_md5):
        print("  comparison_results.json: 已归档")

    # 2b. 各模型评估目录
    table2_models = discover_table2_models(table2_src)
    for model in table2_models:
        src = os.path.join(table2_src, model)
        dst = os.path.join(archive_root, "table2", model)
        rel_prefix = "table2/{}".format(model)
        n = archive_dir(src, dst, rel_prefix, files_info, missing_files,
                        args.skip_missing, with_md5)
        print("  table2/{}: {} 个文件".format(model, n))

    # 3. 汇总并写入 manifest
    print("[3/3] 写入 manifest 并校验...")
    total_files = len(files_info)
    total_size = sum(e["size_bytes"] for e in files_info)

    source_dirs = {
        "baselines": "results/baselines/",
        "table2": "results/table2/",
    }
    manifest = write_manifest(
        archive_root, files_info, missing_files, source_dirs,
        list(BASELINE_MODELS), total_size, total_files)
    manifest_path = os.path.join(archive_root, "manifest.json")

    # 4. 验证归档完整性
    verified = verify_archive(archive_root, manifest)

    # 5. 打印汇总
    print("-" * 60)
    print("归档完成: {}".format(archive_root))
    print("文件数量: {}".format(total_files))
    print("总大小:   {} 字节 ({:.2f} MB)".format(
        total_size, total_size / (1024 * 1024)))
    print("manifest: {}".format(manifest_path))
    if missing_files:
        print("缺失项:   {} 个（详见 manifest.missing_files）".format(len(missing_files)))
    print("完整性校验: {}".format("通过 ✅" if verified else "失败 ❌"))
    print("=" * 60)

    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
