"""批量训练 4 个 baseline 模型（Xception/EffNet-B0/B3/MesoNet）
依次训练，每个模型 15 epoch + early stopping patience=7
启用 num_workers=4 + AMP 加速（解决 Windows 视频解码瓶颈）
与 CAR 公平对比：相同数据划分/预处理/seed
"""
import subprocess
import sys
import os
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["mesonet", "efficientnet_b0", "efficientnet_b3", "xception"]
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "results", "baselines", "training_summary.json")


def train_one_model(model_name, epochs_limit=15):
    """训练单个模型，返回状态字典。

    完成检测: 若 results/baselines/<model>/results.json 存在且含 test_auc,
              视为已完成, 跳过训练（不重复训练, 不覆盖 baseline 数据）。
    断点续训: 若 results/baselines/<model>/best_model.pt 存在且 epoch+1 < epochs_limit,
              自动添加 --resume 参数从断点续训（warm restart）。
    """
    start_time = datetime.now()
    status = {
        "model": model_name,
        "start_time": start_time.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "status": "running",
        "return_code": None,
        "error": None,
        "resumed": False,
        "resumed_from_epoch": None,
    }

    # === 完成检测: 已训练完成的模型跳过（不重复训练, 保护 baseline 数据） ===
    results_json_path = os.path.join(
        PROJECT_ROOT, "results", "baselines", model_name, "results.json"
    )
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, "r", encoding="utf-8") as f:
                r = json.load(f)
            # 含 test_auc 视为已完成
            if "test_auc" in r or "test_metrics" in r or "test" in r:
                end_time = datetime.now()
                status["end_time"] = end_time.isoformat()
                status["duration_seconds"] = 0
                status["status"] = "skipped"
                status["return_code"] = 0
                status["skip_reason"] = "already_completed"
                test_auc = r.get("test_auc")
                if test_auc is None and "test_metrics" in r:
                    test_auc = r["test_metrics"].get("auc")
                if test_auc is None and "test" in r:
                    test_auc = r["test"].get("auc")
                print(f"\n{'='*70}")
                print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] {model_name} 已完成, 跳过")
                print(f"{'='*70}")
                print(f"[SKIP] 检测到 results.json (test_auc={test_auc}), 视为已完成训练")
                return status
        except Exception as e:
            print(f"[WARN] 读取 results.json 失败, 继续检查 resume: {e}")

    print(f"\n{'='*70}")
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] 开始训练: {model_name}")
    print(f"{'='*70}", flush=True)

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "baseline_full.py"),
        "--model", model_name,
        "--epochs", str(epochs_limit),
        "--patience", "7",
        "--num_workers", "4",
        "--config", "configs/default.yaml",
    ]

    # 断点续训检测：优先检测 checkpoint_latest.pt，回退到 best_model.pt
    latest_ckpt_path = os.path.join(PROJECT_ROOT, "results", "baselines", model_name, "checkpoint_latest.pt")
    best_ckpt_path = os.path.join(PROJECT_ROOT, "results", "baselines", model_name, "best_model.pt")

    # 优先使用 checkpoint_latest.pt（完整恢复），否则回退到 best_model.pt（warm restart）
    ckpt_path = None
    resume_mode = None
    if os.path.exists(latest_ckpt_path):
        ckpt_path = latest_ckpt_path
        resume_mode = "full"
    elif os.path.exists(best_ckpt_path):
        ckpt_path = best_ckpt_path
        resume_mode = "warm_restart"

    if ckpt_path:
        import torch as _torch
        try:
            ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_epoch = int(ckpt.get("epoch", -1))  # 0-based
            if ckpt_epoch + 1 < epochs_limit:
                cmd.extend(["--resume", ckpt_path])
                status["resumed"] = True
                status["resumed_from_epoch"] = ckpt_epoch + 1  # 1-based, 即下一个 epoch
                status["resume_mode"] = resume_mode
                mode_desc = "完整恢复 (full resume)" if resume_mode == "full" else "Warm restart"
                ckpt_name = "checkpoint_latest.pt" if resume_mode == "full" else "best_model.pt"
                print(f"[RESUME] 检测到 {ckpt_name} (epoch {ckpt_epoch}, 1-based: Epoch {ckpt_epoch+1})")
                print(f"[RESUME] 恢复模式: {mode_desc}")
                print(f"[RESUME] 将从 Epoch {ckpt_epoch + 2} 续训至 Epoch {epochs_limit}")
            else:
                print(f"[SKIP-RESUME] checkpoint epoch {ckpt_epoch} >= epochs_limit {epochs_limit}, 从头训练")
        except Exception as e:
            print(f"[WARN] 加载 checkpoint 失败, 从头训练: {e}")

    try:
        # 输出直接流到控制台，便于实时监控；训练日志另由 baseline_full.py 写入
        # results/baselines/<model>/training.log
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        end_time = datetime.now()
        status["end_time"] = end_time.isoformat()
        status["duration_seconds"] = (end_time - start_time).total_seconds()
        status["return_code"] = result.returncode
        if result.returncode == 0:
            status["status"] = "success"
            resume_tag = " (resumed from Epoch {})".format(status["resumed_from_epoch"]) if status["resumed"] else ""
            print(f"\n[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"{model_name} 训练成功{resume_tag} (耗时 {status['duration_seconds']:.0f}s)")
        else:
            status["status"] = "failed"
            print(f"\n[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"{model_name} 训练失败 (returncode={result.returncode})")
    except Exception as e:
        end_time = datetime.now()
        status["end_time"] = end_time.isoformat()
        status["duration_seconds"] = (end_time - start_time).total_seconds()
        status["status"] = "error"
        status["error"] = str(e)
        status["return_code"] = -1
        print(f"\n[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"{model_name} 训练异常: {e}")

    return status


def _write_summary(all_status, overall_start, overall_end=None, total_duration=None):
    """写入汇总文件（每个模型完成后即时更新，防止中途崩溃丢失进度）"""
    summary = {
        "overall_start_time": overall_start.isoformat(),
        "overall_end_time": overall_end.isoformat() if overall_end else None,
        "total_duration_seconds": total_duration,
        "last_updated": datetime.now().isoformat(),
        "total_models": len(MODELS),
        "completed": sum(1 for s in all_status if s["status"] == "success"),
        "failed": sum(1 for s in all_status if s["status"] != "success"),
        "models": all_status,
    }
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    """依次训练 4 个模型"""
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"待训练模型: {MODELS}")
    print(f"训练协议: epochs=15, patience=7, num_workers=4, AMP=on, config=configs/default.yaml")
    # 断点续训: 检测已有 checkpoint 并自动恢复
    print(f"断点续训: 优先检测 checkpoint_latest.pt（完整恢复），回退 best_model.pt（warm restart）")

    all_status = []
    overall_start = datetime.now()

    for i, model in enumerate(MODELS, 1):
        print(f"\n>>>>> 进度: [{i}/{len(MODELS)}] {model} <<<<<", flush=True)
        status = train_one_model(model, epochs_limit=15)
        all_status.append(status)
        # 每个模型完成后立即更新汇总文件
        _write_summary(all_status, overall_start)

    overall_end = datetime.now()
    total_duration = (overall_end - overall_start).total_seconds()

    summary = _write_summary(all_status, overall_start, overall_end, total_duration)

    print(f"\n{'='*70}")
    print(f"批量训练完成")
    print(f"{'='*70}")
    print(f"总耗时: {total_duration:.0f}s")
    print(f"成功: {summary['completed']}/{len(MODELS)}")
    print(f"失败: {summary['failed']}/{len(MODELS)}")
    print(f"汇总文件: {SUMMARY_PATH}")

    for s in all_status:
        dur = s['duration_seconds']
        dur_str = f"(耗时 {dur:.0f}s)" if dur is not None else ""
        print(f"  - {s['model']}: {s['status']} {dur_str}")


if __name__ == "__main__":
    main()
