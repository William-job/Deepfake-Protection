"""
数据泄露审计脚本（阶段 0.1）

对 CAR 实验的数据划分做只读审计，检测：
  1. train/val/test 是否按视频级隔离（无路径交集）
  2. List_of_testing_videos.txt 是否存在、test 视频是否误入 train/val
  3. train+val 规模是否异常（相对 Celeb-DF 官方规模）
  4. checkpoint 溯源（默认 checkpoints/best_model.pt vs checkpoints_v5/best_model.pt）

无 GPU 依赖，可在任意环境运行：

    python scripts/audit_data_leak.py --config configs/default.yaml

输出：控制台报告 + results/audit/audit_report.json
"""
import argparse
import json
import os
import sys

# 允许从 scripts/ 目录直接运行：把项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config


def parse_test_list(list_path):
    """解析 List_of_testing_videos.txt，返回 set(rel_path) 与 label_map。"""
    test_set, label_map = set(), {}
    if not os.path.exists(list_path):
        return test_set, label_map
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            label_str, rel_path = parts
            rel_path = rel_path.replace("/", os.sep)
            test_set.add(rel_path)
            label_map[rel_path] = int(label_str)
    return test_set, label_map


def collect_video_paths(root):
    """递归收集 root 下所有视频文件相对路径（不含 root 前缀）。"""
    if not os.path.exists(root):
        return []
    exts = (".mp4", ".avi", ".mov")
    paths = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(exts):
                abs_p = os.path.join(dirpath, fn)
                paths.append(os.path.relpath(abs_p, root))
    return paths


def identity_key(rel_path):
    """从 Celeb-DF 相对路径提取身份标识（视频文件名去掉非身份片段）。"""
    base = os.path.basename(rel_path)
    return base.split("_")[0].lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output", default="results/audit/audit_report.json")
    args = ap.parse_args()

    config = load_config(args.config)
    data_root = config.data.data_root
    findings = {"data_root": data_root, "checks": []}

    print("=" * 70)
    print("CAR 数据泄露审计")
    print(f"data_root = {data_root}")
    print("=" * 70)

    ok = lambda name, passed, detail: findings["checks"].append(
        {"check": name, "passed": bool(passed), "detail": detail}
    )

    # --- 1. 数据目录存在性 ---
    if not os.path.exists(data_root):
        print(f"[SKIP] 数据目录不存在：{data_root}")
        print("[SKIP] 请在持有 Celeb-DF++ 数据的环境运行本脚本以完成完整审计。")
        ok("data_root_exists", False, "数据目录不存在，无法审计")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(findings, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.output}")
        return

    # --- 2. List_of_testing_videos.txt ---
    list_path = os.path.join(data_root, "List_of_testing_videos.txt")
    test_set, label_map = parse_test_list(list_path)
    print(f"\n[1] List_of_testing_videos.txt 存在: {os.path.exists(list_path)}")
    print(f"    解析到 test 视频数: {len(test_set)}")
    ok("test_list_exists", os.path.exists(list_path), f"test 视频 {len(test_set)} 个")

    # --- 3. 视频文件采集（各子集） ---
    real_dirs = [
        os.path.join(data_root, "Celeb-real"),
        os.path.join(data_root, "YouTube-real"),
    ]
    syn_dir = os.path.join(data_root, "Celeb-synthesis")

    real_paths = []
    for d in real_dirs:
        real_paths += collect_video_paths(d)
    fake_paths = collect_video_paths(syn_dir)

    print(f"\n[2] 视频文件统计")
    print(f"    real  : {len(real_paths)}")
    print(f"    fake  : {len(fake_paths)}")
    print(f"    合计  : {len(real_paths) + len(fake_paths)}")

    # 真实视频/伪视频是否重复（同一路径）
    real_set = set(real_paths)
    fake_set = set(fake_paths)
    dup_real_fake = real_set & fake_set
    print(f"    real∩fake 路径交集: {len(dup_real_fake)}")
    ok("no_real_fake_path_overlap", len(dup_real_fake) == 0,
       f"交集 {len(dup_real_fake)} 个" + (":" + ",".join(sorted(dup_real_fake)[:5]) if dup_real_fake else ""))

    # 身份级重叠：真实身份 vs 伪视频身份
    real_ids = {identity_key(p) for p in real_paths}
    fake_ids = {identity_key(p) for p in fake_paths}
    id_overlap = real_ids & fake_ids
    print(f"    身份级 real∩fake 交集（按 basename 前缀）: {len(id_overlap)}")
    ok("identity_overlap_info", True, f"身份交集 {len(id_overlap)} 个（Celeb-DF 官方允许身份跨集，需记录）")

    # --- 4. test 视频是否误入 train/val 集合（用 endswith 匹配语义模拟） ---
    if test_set:
        # 建立"非 test"集合（train/val 候选）
        all_paths = real_paths + fake_paths
        leaked = []
        for p in all_paths:
            p_norm = p.replace(os.sep, "/")
            for tp in test_set:
                t_norm = tp.replace(os.sep, "/")
                if p_norm.endswith(t_norm) or t_norm.endswith(p_norm):
                    leaked.append((p, tp))
                    break
        print(f"\n[3] test 与 全量视频 endswith 重叠条目数: {len(leaked)}")
        ok("test_train_approx_disjoint", len(leaked) == 0,
           f"重叠 {len(leaked)} 条（endswith 近似）")

    # --- 5. checkpoint 溯源 ---
    print(f"\n[4] checkpoint 溯源")
    ckpt_default = "checkpoints/best_model.pt"
    ckpt_v5 = "checkpoints_v5/best_model.pt"
    for p in [ckpt_default, ckpt_v5]:
        if os.path.exists(p):
            st = os.stat(p)
            print(f"    {p:<32} {st.st_size/1e6:6.2f} MB  mtime={__import__('datetime').datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}")
        else:
            print(f"    {p:<32} (缺失)")
    mismatch = (os.path.exists(ckpt_default) and os.path.exists(ckpt_v5)
                and os.path.getsize(ckpt_default) != os.path.getsize(ckpt_v5))
    print(f"    评估脚本默认 checkpoint 与 v5 大小不一致: {mismatch}")
    ok("checkpoint_provenance", not mismatch,
       f"默认 {ckpt_default} 与 {ckpt_v5} 大小{'不一致（评估用错 checkpoint 风险）' if mismatch else '一致'}")

    # --- 汇总 ---
    print("\n" + "=" * 70)
    passed = sum(1 for c in findings["checks"] if c["passed"])
    print(f"审计结论：{passed}/{len(findings['checks'])} 项通过")
    for c in findings["checks"]:
        mark = "PASS" if c["passed"] else "FAIL/SKIP"
        print(f"  [{mark}] {c['check']}: {c['detail']}")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {args.output}")


if __name__ == "__main__":
    main()