class ProgressiveMixedCurriculum:
    def __init__(self, total_epochs=30, pretrain_ratio=0.3, progressive_ratio=0.2):
        self.total_epochs = max(1, total_epochs)
        self.pretrain_epochs = max(1, int(self.total_epochs * pretrain_ratio))
        self.progressive_epochs = max(1, int(self.total_epochs * progressive_ratio))
        self.mixed_precision_start = self.pretrain_epochs + self.progressive_epochs

    def get_stage(self, epoch):
        if epoch < self.pretrain_epochs:
            return {
                "name": "artifact_pretraining",
                "target_top_k": 1,
                "difficulty_loss_weight": 0.0,
                "qat_enabled": False,
                "qat_loss_weight": 0.0,
                "anti_collapse_weight": 0.1,
            }

        if epoch < self.mixed_precision_start:
            progress = (epoch - self.pretrain_epochs) / max(1, self.progressive_epochs)
            return {
                "name": "progressive_curriculum",
                "target_top_k": 2,
                "difficulty_loss_weight": min(0.3, progress * 0.3),
                "qat_enabled": False,
                "qat_loss_weight": 0.0,
                "anti_collapse_weight": min(0.2, progress * 0.2),
            }

        progress = min(1.0, (epoch - self.mixed_precision_start) / max(1, self.total_epochs - self.mixed_precision_start))
        return {
            "name": "mixed_precision_finetuning",
            "target_top_k": 2,
            "difficulty_loss_weight": 0.3,
            "qat_enabled": True,
            "qat_loss_weight": 0.02 + 0.03 * progress,
            "anti_collapse_weight": 0.3 + 0.2 * progress,
        }
