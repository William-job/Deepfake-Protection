# 数据泄露与评估协议审计报告（阶段 0.1）

> 审计日期：2026-08-23
> 结论：**当前 `results/` 下所有已发布指标（robustness / ablation_study / cross_dataset_ff）均不可信，未修复前不得用于任何论文结论。**

---

## 0. 审计对象

| 文件 | 关键指标 | 判定 |
|------|----------|------|
| `results/robustness.json` | baseline AUC=0.9996, EER=0.0, TPR@FPR=0.1%=0.997 | **疑似泄露** |
| `results/ablation_study.json` | Full=0.9341, -Temporal/-Flow/-Blending≈0.065 | **harness 已损坏** |
| `results/cross_dataset_ff.json` | FF++ c23 AUC=0.4917（随机水平） | 泛化失效 |

---

## 1. P0 缺陷一：checkpoint / 测试集不匹配，test AUC 远超 val AUC

### 证据
- 最新训练 `logs_v5/training.log` 全程最佳 **val AUC = 0.8625（epoch 23）**，train AUC ≈ 0.90（epoch 23/25/27）。
- 但 `robustness.json` 的 baseline **test AUC = 0.9996**，EER=0.0。
- test AUC（0.9996）比 val AUC（0.8625）高出 **+0.137**，且方向相反（固定划分下 test ≤ val 才正常）。这在统计上不可能，除非存在泄露或用了错误的 checkpoint。

### 根因（两点叠加）
1. **checkpoint 用错**：所有评估脚本（`robustness_experiment.py` / `ablation_experiment.py` / `expert_ablation.py` / `cross_dataset_eval*.py`）默认 `--checkpoint checkpoints/best_model.pt`。
   - `checkpoints/best_model.pt`：**71.9 MB，2026-05-26**（旧 `logs/` 时代产物）。
   - 当前模型的正确 checkpoint 是 `checkpoints_v5/best_model.pt`：**81.8 MB，2026-06-26**（对应 logs_v5）。
   - 两文件大小（71.9 vs 81.8 MB）不同 → **架构/训练不一致**，评估用的是旧 checkpoint。
2. 旧 `checkpoints/best_model.pt` 的 train/val/test 划分与当前 `DeepfakeDataset` 可能不一致，且疑似早前版本存在 train/test 重叠。

### 修复
- 评估必须显式指向 `checkpoints_v5/best_model.pt`（或任意唯一声明的 checkpoint），并在结果 JSON 记录 `checkpoint` 字段（目前 `cross_dataset_ff.json` 已记、`robustness/ablation` 未记）。
- 用同一 checkpoint 复现：若 val≈0.86 而 test≈0.9996 仍成立 → 判定为 test/train 重叠，直接修复划分。

---

## 2. P0 缺陷二：消融 harness 已损坏（==已修复==）

### 根因
`expert_ablation.py` 的 `ExpertAblation.forward` 把 `out["head_outputs"]`（**head 特征，非 2 类 logits**）直接当作 expert logits 叠加：

```python
expert_logits = out["head_outputs"]          # 错：head 特征 (B, 128)
stacked = torch.stack(logit_list, dim=1)      # (B, 4, 128)
y_combined = (w.unsqueeze(-1) * stacked).sum(dim=1)  # (B, 128)
# 下游 sigmoid(logits[:, 1]) 取的是第 1 个特征通道，几乎随机
```

`car.py` 正确的 expert logits 应经 `self.experts[name](head_outputs[name])` 得到 `(B, 2)`。

### 现象
`-Temporal/-Flow/-Blending` 三配置 AUC 坍缩到 0.065–0.069（比随机 0.5 还低），acc≈0.5，f1=0 —— 实为随机切片，非真实"专家边际贡献"。

### 修复（已应用 `expert_ablation.py`）
- 重算 `self.base.experts[name](head_outputs[name])`，复刻 `CAR.forward` 的 batch-size 归一化与 `clamp(-100,100)`。
- 新增消融有效性断言：`AUC<0.6 且 acc≈0.5` → 标记 `harness_error=True`，结果 JSON 写入 `valid/harness_error` 字段。

---

## 3. P1 缺陷三：阈值在 test 集上确定（Youden）

### 根因
`src/utils/metrics.py::compute_all_metrics` 调用 `find_optimal_threshold(preds, labels)`，用 **test 集的 Youden's J** 选阈值，再算 acc / f1 / EER。这是"test-set 阈值"造成的轻度信息泄露。

- 影响：**acc / f1 / EER**（阈值相关）被高估；**AUC / AP**（阈值无关）不受影响。
- `robustness.json` 中 `threshold=0.9868` 即 test-optimal 阈值，`accuracy=0.998` 因此不可信。

### 修复
- 阈值必须在 **val 集**用 Youden 确定，冻结后套用于 test 集。
- `compute_all_metrics` 增加可选参数 `threshold=None`；评估脚本改为 `compute_all_metrics(preds, labels, threshold=val_threshold)`。

---

## 4. P1 缺陷四：`endswith` 路径匹配脆弱（==已修复==）

### 根因
`src/data/dataset.py::_load_samples` 用 `rel_path_normalized.endswith(test_path)` 判断 test 归属，当 `List_of_testing_videos.txt` 内出现短路径或前缀碰撞时会误配/漏配，且复杂度 O(n·m)。

### 修复（已应用 `src/data/dataset.py`）
- `_parse_test_list` 将 test 路径统一归一化为 forward-slash 存入 `set`。
- `_load_samples` 改为 O(1) 精确集合匹配（`rel_path_normalized in test_set`），替代 `endswith` 循环。
- 新增断言：test 列表每条都应精确命中一个已采集视频，未命中则打印 WARNING。
- 实测校验：test=5418（178 real + 5240 fake），无未匹配条目。

---

## 5. P2 已核实：数据规模异常（`scripts/audit_data_leak.py` 实测）

- 实测数据目录 `D:/Celeb-DF++/Celeb-DF-v3`：
  - real 视频 = **890**
  - fake 视频 = **53196**
  - 合计 = **54086**
  - `List_of_testing_videos.txt` 解析出 test = **5418**；train+val = 48668（与 logs_v5 一致）。
- Celeb-DF v2 官方规模约 6229（590 real + 5639 fake）。当前 real=890 与官方量级接近，但 **fake=53196 ≈ 官方 5639 的 9.4 倍**，强烈提示 `_collect_fake_videos`（`dataset.py`）的嵌套目录遍历存在重复计数，或 `Celeb-DF-v3` 确为更大规模变体。
- 待定项：需逐目录核对 fake 视频是否被重复入样（同一视频在 `Celeb-synthesis/<method>/<id>/` 与其 `Celeb-DF-v2/` 子目录下被双计）。

### 结论（已核对，非重复计数）
- 实测目录结构：`Celeb-synthesis/FaceReenact|FaceSwap|TalkingFace/<method>/<视频>.mp4`，为严格三层结构，无更深嵌套；`FaceSwap/Celeb-DF-v2/`（5639 个）是原 Celeb-DF v2 的面部交换子集，与 v3 新增方法并行存在、互不重叠。
- `os.walk` 递归计数 = 手工三层计数 = **53196**，二者一致 → **无漏采、无重复计数**。
- 53196 ≈ 官方 v2 5639 的 9.4 倍，是因为 Celeb-DF-v3 新增了 FaceReenact / TalkingFace 两个类别与数十种生成方法，数据规模本就更庞大。
- 已将 `_collect_fake_videos` 由硬编码三层遍历改为 `os.walk` 递归，保证任意嵌套下的正确性。

---

## 6. P1 已实测：checkpoint 溯源确认

- `checkpoints/best_model.pt`：71.93 MB，2026-05-26（评估脚本默认值）。
- `checkpoints_v5/best_model.pt`：81.80 MB，2026-06-26（logs_v5 实际产物，val AUC 0.8625）。
- 两者大小与时间戳均不一致 → **评估脚本使用的是旧 checkpoint**，与 v5 训练产物无关。

---

## 7. 修复优先级与后续动作

| 优先级 | 动作 | 状态 |
|--------|------|------|
| P0 | 消融 harness 修复 + 有效性断言 | ✅ 已修复 `expert_ablation.py` |
| P0 | 统一 checkpoint 指向 + 结果记录 checkpoint 字段 | 待改评估脚本 |
| P0 | 用 `checkpoints_v5/best_model.pt` 复现 test/val 指标，定位泄露 | 待运行 |
| P1 | 阈值改为 val 确定 | ✅ 已改 `metrics.py`（支持 `threshold` 参数） |
| P1 | `endswith` 改精确匹配 | ✅ 已改 `dataset.py`（精确匹配 + os.walk 递归遍历） |
| P2 | 审计数据目录真实视频数 | ✅ 已核对：fake=53196 属实、非重复计数 |

> 在 P0 全部完成并产出「诚实基线」前，不启动阶段 1–5 架构重定义。