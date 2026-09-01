# 诚实基线评估协议（阶段 0.3）

> 本文档是 CAR 与 baseline（EfficientNet-B0 / Xception / MesoNet）公平对比的**唯一权威协议**。
> 任何"架构优于 baseline"的结论，必须建立在遵循本协议、且 `summary.md` 中 `complete=true` 的指标之上。
> 生成日期：2026-08-23

---

## 1. 数据划分（已修复）

| 项 | 值 |
|----|----|
| 数据源 | `D:/Celeb-DF++/Celeb-DF-v3`（Celeb-DF-v3） |
| 划分方式 | 视频级隔离（非帧级） |
| test 归属 | `List_of_testing_videos.txt` **精确匹配**（已由 `endswith` 修复） |
| 样本数 | train 38,934 / val 9,734 / test 5,418 |
| test 构成 | 178 real + 5,240 fake |
| 泄露 | 无（`results/audit/leak_report.md`） |

## 2. 预处理（完全一致）

- 帧采样：每视频 8 帧，`frame_stride=2`（即第 0,2,4,...14 帧）
- 尺寸：`224 × 224`，RGB
- 归一化：`(x / 255.0 - 0.5) / 0.5` → 值域 [-1, 1]
- 训练期帧级增强：`VideoTransform`（水平翻转 p=0.5、旋转±5°、亮度±0.1、对比度±0.1）

## 3. 训练协议

| 项 | CAR | EfficientNet-B0 / Xception | MesoNet |
|----|-----|---------------------------|---------|
| 初始化 | EfficientNet-B0 stem（ImageNet） | ImageNet（timm pretrained） | 随机 |
| 采样 | WeightedRandomSampler（real:fake≈1:1） | 同左 | 同左 |
| 损失 | BCE + aux/balance/difficulty（CARLoss） | CrossEntropy | CrossEntropy |
| 优化器 | AdamW | Adam | Adam |
| 学习率 | 3e-4（stem 5e-5） | 1e-4 | 1e-4 |
| epoch 预算 | 早停（patience=28） | 早停（patience=7） | 早停（patience=7） |
| CutMix | p=0.15（CAR 特有 MoE 正则化） | 不适用 | 不适用 |
| AMP | 是 | 是 | 是 |

> **记录的控制变量差异**：CutMix 仅施加于 CAR（其设计目的是增强 MoE 的组合伪造识别）；baseline 为单流 CNN，不强加 CutMix。此差异在论文中显式声明，不隐瞒。

## 4. 评估协议（完全一致，已修复）

1. 用 **val 集** 以 Youden's J（最大化 TPR−FPR）确定阈值 `τ`；
2. 将 `τ` **冻结**后用于 **test 集**计算 Acc / F1（杜绝 test 集阈值过拟合）；
3. 阈值无关指标 **AUC / AP / EER / TPR@FPR=1% / TPR@FPR=0.1%** 直接在 test 集计算；
4. 强制 `model.eval()` + `torch.no_grad()`，评估时禁用一切随机增强（CutMix 关闭）；
5. 评分方式：baseline `softmax(logits)[:,1]`；CAR `sigmoid(logits[:,1])`；
6. 保存每个 seed 的 `raw_predictions.npz`（preds/labels/video_ids）与 `eval_metrics.json`。

## 5. 多 seed 与统计

- 每模型 **3 个 seed**：`42, 43, 44`；
- 报告 `mean ± std`（AUC/Acc/F1/AP/EER/TPR@1%/TPR@0.1% 七项）；
- 最终表与 bootstrap 95% CI 由 `scripts/aggregate_honest.py` 生成。

## 6. 输出目录

```
results/baseline_honest/
  protocol.md
  car/seed_42/{logs,checkpoints,results.json,eval_metrics.json,raw_predictions.npz}
  efficientnet_b0/seed_42/...
  xception/seed_42/...
  mesonet/seed_42/...
  summary.json / summary.md
```

## 7. 复现命令

```bash
# 训练（每 seed）
python train.py --seed 42 --output-dir results/baseline_honest/car/seed_42
python baseline_full.py --model efficientnet_b0 --seed 42 --output-dir results/baseline_honest/efficientnet_b0/seed_42

# 评估（val→test）
python scripts/eval_honest.py --model car --checkpoint results/baseline_honest/car/seed_42/checkpoints/best_model.pt --seed 42 --output-dir results/baseline_honest/car/seed_42

# 汇总
python scripts/aggregate_honest.py
```

## 8. 变更记录

- 2026-08-23：初版。
- 2026-08-24：剔除 EfficientNet-B3。队列固定为 **4 模型 × 3 seed = 12 个任务**（CAR / EfficientNet-B0 / Xception / MesoNet）；同步从 `scripts/aggregate_honest.py` 与 `scripts/eval_honest.py` 中移除 B3（此前误列入，`run_honest_queue.py` 从未包含它）。B3 旧结果仅存于 `archives/baselines_*`，不参与本协议对比。