# R1/R2/R3 初始化对照（阶段 2.4）

> 说明：val_auc 为预训练后（未做下游 Celeb-DF 微调）的零样本 val AUC，用于对照初始化对下游判别力的影响。

| Mode | 初始化 | Router KL | val AUC (零样本) |
|---|---|---|---|
| R1 | 随机初始化（无 ImageNet）+ Level2/3 | 0.8177 | 0.6152 |
| R2 | ImageNet stem（无 artifact 预训练） | - | 0.2450 |
| R3 | ImageNet stem + 完整 Level2/3（CAR-aware） | 0.7101 | 0.3702 |

## Level 2 各专家伪任务 AUC（CAR-aware 预训练质量探针）

### R1
| 专家 | acc | auc | batches |
|---|---|---|---|
| motion | 0.500 | 0.500 | 30 |
| temporal | 0.625 | 0.672 | 30 |
| spectral | 0.625 | 0.734 | 30 |
| boundary | 0.562 | 0.672 | 30 |

### R3
| 专家 | acc | auc | batches |
|---|---|---|---|
| motion | 0.500 | 0.484 | 30 |
| temporal | 0.625 | 0.734 | 30 |
| spectral | 1.000 | 1.000 | 30 |
| boundary | 1.000 | 1.000 | 30 |
