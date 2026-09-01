import numpy as np
import torch
import torch.nn.functional as F

from src.utils.metrics import compute_accuracy, compute_auc, compute_f1, find_optimal_threshold


class BoundaryLogitCalibrator:
    def __init__(self, max_iter=300, lr=0.03, distill_weight=1.0, label_weight=0.2):
        self.max_iter = max_iter
        self.lr = lr
        self.distill_weight = distill_weight
        self.label_weight = label_weight
        self.alpha = 1.0
        self.beta = 0.0

    def fit(self, quantized_logits, fp32_logits, labels=None):
        q_logits = self._to_tensor(quantized_logits).float().flatten()
        teacher_logits = self._to_tensor(fp32_logits).float().flatten()

        alpha = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        beta = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        optimizer = torch.optim.Adam([alpha, beta], lr=self.lr)

        label_tensor = None
        if labels is not None:
            label_tensor = self._to_tensor(labels).float().flatten()

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            calibrated_logits = alpha * q_logits + beta
            distill_loss = F.mse_loss(torch.sigmoid(calibrated_logits), torch.sigmoid(teacher_logits))
            loss = self.distill_weight * distill_loss
            if label_tensor is not None:
                label_loss = F.binary_cross_entropy_with_logits(calibrated_logits, label_tensor)
                loss = loss + self.label_weight * label_loss
            loss.backward()
            optimizer.step()

        self.alpha = float(alpha.detach().cpu().item())
        self.beta = float(beta.detach().cpu().item())
        return self

    def transform_logits(self, logits):
        logits_tensor = self._to_tensor(logits).float()
        return logits_tensor * self.alpha + self.beta

    def predict_proba(self, logits):
        return torch.sigmoid(self.transform_logits(logits))

    def evaluate(self, logits, labels):
        probs = torch.sigmoid(self._to_tensor(logits).float()).detach().cpu().numpy().flatten()
        labels_np = np.array(labels).flatten()
        return self._metrics_from_probs(probs, labels_np)

    def evaluate_calibrated(self, logits, labels):
        probs = self.predict_proba(logits).detach().cpu().numpy().flatten()
        labels_np = np.array(labels).flatten()
        return self._metrics_from_probs(probs, labels_np)

    def state_dict(self):
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "max_iter": self.max_iter,
            "lr": self.lr,
            "distill_weight": self.distill_weight,
            "label_weight": self.label_weight,
        }

    def load_state_dict(self, state_dict):
        self.alpha = float(state_dict["alpha"])
        self.beta = float(state_dict["beta"])
        self.max_iter = state_dict.get("max_iter", self.max_iter)
        self.lr = state_dict.get("lr", self.lr)
        self.distill_weight = state_dict.get("distill_weight", self.distill_weight)
        self.label_weight = state_dict.get("label_weight", self.label_weight)
        return self

    @staticmethod
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        return torch.tensor(x)

    @staticmethod
    def _metrics_from_probs(probs, labels):
        threshold = find_optimal_threshold(probs, labels)
        return {
            "accuracy": compute_accuracy(probs, labels, threshold),
            "auc": compute_auc(probs, labels),
            "f1": compute_f1(probs, labels, threshold),
            "threshold": threshold,
        }
