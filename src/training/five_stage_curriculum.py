"""阶段 4.1：五阶段渐进课程（Progressive Mixed Curriculum）。

与四专家重定义一致的渐进课程（非"简单→困难"朴素课程）：
  Stage 1 single           → 单 artifact（ControlledSBI 单类）
  Stage 2 dual             → 双 artifact 混合
  Stage 3 multi            → 多 artifact 混合
  Stage 4 distribution shift → 质量退化（JPEG/blur/resize/H.264/noise/color）
  Stage 5 unseen generator → 未见过生成器/操纵（用 FF++ 或保留方法模拟）

每类返回 dict(name, artifact_mix, quality_aug_p, top_k, description)，
供训练循环按 epoch 切换，并记录日志。
"""


class FiveStageCurriculum:
    def __init__(self, total_epochs=28):
        self.total_epochs = max(5, total_epochs)
        # 五阶段按总 epoch 等比划分（每段至少 1 epoch）
        seg = max(1, self.total_epochs // 5)
        self.stage_epochs = [0, seg, 2 * seg, 3 * seg, 4 * seg, self.total_epochs]

    def get_stage(self, epoch):
        if epoch < self.stage_epochs[1]:
            return {
                "name": "single_artifact",
                "stage_idx": 1,
                "artifact_mix": 1,           # 单 artifact
                "quality_aug_p": 0.0,
                "top_k": 2,
                "description": "单 artifact：每类 ControlledSBI 独立，专家专业化",
            }
        if epoch < self.stage_epochs[2]:
            return {
                "name": "dual_artifact",
                "stage_idx": 2,
                "artifact_mix": 2,           # 双 artifact 混合
                "quality_aug_p": 0.1,
                "top_k": 2,
                "description": "双 artifact 混合：学习组合伪造",
            }
        if epoch < self.stage_epochs[3]:
            return {
                "name": "multi_artifact",
                "stage_idx": 3,
                "artifact_mix": 3,           # 多 artifact 混合
                "quality_aug_p": 0.2,
                "top_k": 3,
                "description": "多 artifact 混合：提升难度，专家协同",
            }
        if epoch < self.stage_epochs[4]:
            return {
                "name": "distribution_shift",
                "stage_idx": 4,
                "artifact_mix": 3,
                "quality_aug_p": 0.5,        # 质量退化为主
                "top_k": 3,
                "description": "分布偏移：质量退化（JPEG/blur/resize/H.264/noise/color），鲁棒性",
            }
        return {
            "name": "unseen_generator",
            "stage_idx": 5,
            "artifact_mix": 4,
            "quality_aug_p": 0.6,
            "top_k": 4,
            "description": "未见生成器：最强退化与混合，验证学的是 artifact 而非 dataset label",
        }