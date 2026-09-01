import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocessing.multi_scale import MultiScalePreprocessor
from src.models.stem import SharedStem
from src.models.heads.temporal_head import TemporalHead
from src.models.heads.motion_head import MotionHead
from src.models.heads.spectral_head import SpectralHead
from src.models.heads.boundary_head import BoundaryHead
from src.models.fusion import ArtifactFusion
from src.models.gating import DifficultyEstimator, GatingNetwork
from src.models.experts.redefined_experts import (
    MotionExpert,
    TemporalExpert,
    SpectralExpert,
    BoundaryExpert,
)


class CAR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.preprocessor = MultiScalePreprocessor()

        self.stem = SharedStem(
            stem_type=config.model.stem,
            pretrained=True,
            freeze=False,
        )

        # Get actual output channels from stem
        stem_channels = self.stem.out_channels

        num_frames = config.data.num_frames
        latent_dim = config.model.latent_dim
        latent_k = config.model.latent_k

        self.temporal_head = TemporalHead(stem_channels, latent_dim, num_frames)
        self.motion_head = MotionHead(stem_channels, latent_dim, num_frames)
        self.spectral_head = SpectralHead(stem_channels, latent_dim, num_frames)
        self.boundary_head = BoundaryHead(stem_channels, latent_dim, num_frames)

        self.fusion = ArtifactFusion(
            latent_dim=latent_dim,
            latent_k=latent_k,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_transformer_layers,
        )

        self.gating = GatingNetwork(
            latent_k=latent_k,
            top_k=config.model.top_k,
            temperature=config.training.temperature_start,
            min_temperature=getattr(config.training, "difficulty_temperature_min", getattr(config.training, "temperature_end", 0.1)),
            max_temperature=getattr(config.training, "difficulty_temperature_max", config.training.temperature_start),
            difficulty_conditioned=getattr(config.training, "gating_difficulty_conditioned", False),
        )

        self.difficulty_estimator = None  # 由 forward 中 expert disagreement 直接计算，无需独立模块

        self.head_norms = nn.ModuleDict({
            "temporal": nn.LayerNorm(latent_dim),
            "motion": nn.LayerNorm(latent_dim),
            "spectral": nn.LayerNorm(latent_dim),
            "boundary": nn.LayerNorm(latent_dim),
        })

        self.experts = nn.ModuleDict({
            "temporal": TemporalExpert(latent_dim, latent_dim * 2),
            "motion": MotionExpert(latent_dim, latent_dim * 2),
            "spectral": SpectralExpert(latent_dim, latent_dim * 2),
            "boundary": BoundaryExpert(latent_dim, latent_dim * 2),
        })

        self.expert_names = ["motion", "temporal", "spectral", "boundary"]

    def forward(self, x):
        preprocessed = self.preprocessor(x)
        x_rgb = preprocessed["rgb"]

        features = self.stem(x_rgb)

        h_motion = self.motion_head(features)
        h_temporal = self.temporal_head(features)
        h_spectral = self.spectral_head(features)
        h_boundary = self.boundary_head(features)

        head_outputs = {
            "motion": h_motion,
            "temporal": h_temporal,
            "spectral": h_spectral,
            "boundary": h_boundary,
        }

        h_motion_norm = self.head_norms["motion"](h_motion)
        h_temporal_norm = self.head_norms["temporal"](h_temporal)
        h_spectral_norm = self.head_norms["spectral"](h_spectral)
        h_boundary_norm = self.head_norms["boundary"](h_boundary)

        z = self.fusion(h_motion_norm, h_temporal_norm, h_spectral_norm, h_boundary_norm)

        # difficulty = expert disagreement（专家间分歧熵），替代 confidence 代理
        head_stack = torch.stack([h_motion_norm, h_temporal_norm, h_spectral_norm, h_boundary_norm], dim=1)
        head_stack_normed = F.normalize(head_stack, p=2, dim=-1)
        cos_matrix = torch.bmm(head_stack_normed, head_stack_normed.transpose(1, 2))
        mask = torch.triu(torch.ones(4, 4, device=cos_matrix.device, dtype=torch.bool), diagonal=1)
        pair_cos = cos_matrix[:, mask]                      # (B, 6)
        disagreement = (1.0 - pair_cos).clamp(min=0.0)
        difficulty = disagreement.mean(dim=-1, keepdim=True)  # (B, 1) in [0,1]

        w, active_set, gate_info = self.gating(z, difficulty=difficulty, return_active_set=True, return_aux=True)

        expert_logits = []
        for i, name in enumerate(self.expert_names):
            expert_out = self.experts[name](head_outputs[name])
            # Ensure all outputs have the same batch size
            if expert_out.dim() == 2 and expert_out.size(0) != z.size(0):
                # Average over temporal dimension if needed
                if expert_out.size(0) % z.size(0) == 0:
                    T = expert_out.size(0) // z.size(0)
                    expert_out = expert_out.view(z.size(0), T, -1).mean(dim=1)
                else:
                    # Repeat to match batch size
                    repeat_factor = z.size(0) // expert_out.size(0)
                    if repeat_factor > 0:
                        expert_out = expert_out.repeat(repeat_factor, 1)
            expert_logits.append(expert_out)

        expert_logits = torch.stack(expert_logits, dim=1)
        # 防止 0*inf=NaN：当 top_k 未选中某专家（w=0）但该专家 logits 在 fp16 下溢出为 inf 时，
        # 0*inf=NaN 会污染 y_combined。clamp 到 [-100, 100] 消除 inf，对学习影响可忽略。
        expert_logits = expert_logits.clamp(-100.0, 100.0)

        y_combined = (w.unsqueeze(-1) * expert_logits).sum(dim=1)

        return {
            "logits": y_combined,
            "z": z,
            "w": w,
            "w_dense": gate_info.get("w_dense"),
            "active_set": active_set,
            "difficulty": difficulty,
            "gate_temperature": gate_info["temperature"],
            "gate_logits": gate_info.get("gate_logits", None),
            "head_outputs": head_outputs,
        }

    def predict(self, x):
        outputs = self.forward(x)
        logits = outputs["logits"]
        pred = torch.sigmoid(logits[:, 1] if logits.size(1) > 1 else logits.squeeze(-1))

        w = outputs["w"]
        active_set = outputs["active_set"]
        z = outputs["z"]
        difficulty = outputs["difficulty"]

        artifact_labels = ["motion_warping_inconsistency", "temporal_dynamics_anomaly",
                          "spectral_structural_inconsistency", "boundary_identity_inconsistency"]

        composition = {}
        for i, label in enumerate(artifact_labels):
            composition[label] = round(float(w[0, i].item()), 4)

        active_names = [self.expert_names[idx.item()] for idx in active_set[0]]

        report = {
            "confidence": round(float(pred[0].item()), 4),
            "prediction": "fake" if pred[0].item() >= 0.5 else "real",
            "artifact_composition": composition,
            "active_experts": active_names,
            "difficulty": round(float(difficulty[0].item()), 4),
            "route": self._edge_cloud_route(float(difficulty[0].item())),
            "latent_vector": z[0].detach().cpu().tolist(),
            "verdict": self._generate_verdict(pred[0].item(), composition, active_names),
        }

        return pred, report

    def _generate_verdict(self, confidence, composition, active_experts):
        if confidence < 0.5:
            return "Low confidence of manipulation. Video appears authentic."

        primary_artifact = max(composition, key=composition.get)
        primary_value = composition[primary_artifact]

        verdict_parts = []
        if primary_value > 0.5:
            verdict_parts.append(f"High confidence fake. Primary artifact: {primary_artifact}")

        if "motion" in active_experts:
            verdict_parts.append("unnatural motion warping residuals between frames")
        if "temporal" in active_experts:
            verdict_parts.append("incoherent temporal dynamics and jitter")
        if "spectral" in active_experts:
            verdict_parts.append("spectral band-ratio and compression artifacts")
        if "boundary" in active_experts:
            verdict_parts.append("boundary seam and identity inconsistencies")

        if not verdict_parts:
            return "Suspicious patterns detected but no dominant artifact identified."

        return ". ".join(verdict_parts) + "."

    def _edge_cloud_route(self, difficulty):
        if difficulty < 0.3:
            return "edge_exit"
        if difficulty < 0.6:
            return "edge_specialist"
        return "cloud_upload"

    def get_parameter_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def set_temperature(self, temperature):
        self.gating.temperature = temperature

    def freeze_stem(self):
        self.stem.freeze()

    def unfreeze_stem(self):
        self.stem.unfreeze()
