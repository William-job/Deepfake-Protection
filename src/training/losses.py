import torch
import torch.nn as nn
import torch.nn.functional as F


class CARLoss(nn.Module):
    def __init__(self, aux_loss_weight=0.1, load_balance_weight=0.01,
                 difficulty_loss_weight=0.0, min_expert_weight=0.0,
                 min_expert_threshold=0.05, label_smoothing=0.0):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight
        self.load_balance_weight = load_balance_weight
        self.difficulty_loss_weight = difficulty_loss_weight
        self.min_expert_weight = min_expert_weight
        self.min_expert_threshold = min_expert_threshold
        self.label_smoothing = label_smoothing
        self.proxy_weight = 1.0
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels, artifact_gt=None):
        logits = outputs["logits"]
        z = outputs["z"]
        w = outputs["w"]

        logits_bce = torch.clamp(logits, -10.0, 10.0)
        if self.label_smoothing > 0:
            smoothed = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            if logits_bce.size(1) > 1:
                loss_bce = self.bce(logits_bce[:, 1], smoothed)
            else:
                loss_bce = self.bce(logits_bce.squeeze(-1), smoothed)
        else:
            if logits_bce.size(1) > 1:
                loss_bce = self.bce(logits_bce[:, 1], labels)
            else:
                loss_bce = self.bce(logits_bce.squeeze(-1), labels)

        total_loss = loss_bce

        if artifact_gt is not None and self.aux_loss_weight > 0:
            loss_aux = F.mse_loss(z, artifact_gt.to(z.device))
            total_loss = total_loss + self.aux_loss_weight * loss_aux
        else:
            loss_aux = torch.tensor(0.0, device=logits.device)

        if self.load_balance_weight > 0:
            # 作用在 dense softmax 概率（pre-topk）上，而非 post-topk sparse 权重。
            # 依据: Switch Transformer (Fedus et al., 2022) load balance loss 作用在
            # router dense 概率上，确保所有专家（含未被 topk 选中的）都能收到梯度。
            w_dense = outputs.get("w_dense")
            if w_dense is None:
                w_dense = w  # fallback
            # 转 fp32 并 clamp 防止 AMP fp16 下数值下溢：
            # fp16 下 1e-8 下溢为 0，log(0)=-inf，0*-inf=NaN
            w_mean = w_dense.float().clamp(min=1e-6).mean(dim=0)
            loss_balance = (w_mean * torch.log(w_mean)).sum() + torch.log(torch.tensor(4.0))
            loss_balance = torch.clamp(loss_balance, min=0.0)
            total_loss = total_loss + self.load_balance_weight * loss_balance
        else:
            loss_balance = torch.tensor(0.0, device=logits.device)

        difficulty = outputs.get("difficulty")
        if difficulty is not None and self.difficulty_loss_weight > 0:
            confidence = torch.sigmoid(logits[:, 1] if logits.size(1) > 1 else logits.squeeze(-1)).detach()
            proxy = 1.0 - torch.abs(confidence - 0.5) * 2.0
            proxy = proxy.clamp(0.2, 0.8).view_as(difficulty)
            loss_proxy = F.smooth_l1_loss(difficulty, proxy)
            mean_floor = torch.relu(torch.tensor(0.25, device=logits.device) - difficulty.mean()).pow(2)
            std_floor = torch.relu(torch.tensor(0.08, device=logits.device) - difficulty.std(unbiased=False)).pow(2)
            loss_difficulty = self.proxy_weight * loss_proxy + mean_floor + 2.0 * std_floor
            total_loss = total_loss + self.difficulty_loss_weight * loss_difficulty
        else:
            loss_difficulty = torch.tensor(0.0, device=logits.device)

        loss_anti_collapse = torch.tensor(0.0, device=logits.device)
        if self.min_expert_weight > 0:
            # 作用在 dense softmax 概率（pre-topk）上，对未激活专家产生梯度，
            # 防止门控坍缩。与 load_balance 同理。转 fp32 保证数值稳定性。
            w_dense = outputs.get("w_dense")
            if w_dense is None:
                w_dense = w  # fallback
            w_mean_batch = w_dense.float().mean(dim=0)
            shortfall = torch.relu(self.min_expert_threshold - w_mean_batch)
            loss_anti_collapse = shortfall.sum()
            total_loss = total_loss + self.min_expert_weight * loss_anti_collapse

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_bce": loss_bce.item(),
            "loss_aux": loss_aux.item() if isinstance(loss_aux, torch.Tensor) else loss_aux,
            "loss_balance": loss_balance.item() if isinstance(loss_balance, torch.Tensor) else loss_balance,
            "loss_difficulty": loss_difficulty.item() if isinstance(loss_difficulty, torch.Tensor) else loss_difficulty,
            "loss_anti_collapse": loss_anti_collapse.item() if isinstance(loss_anti_collapse, torch.Tensor) else 0.0,
        }
