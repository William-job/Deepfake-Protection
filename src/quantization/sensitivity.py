import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class SensitivityAnalyzer:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _get_expert_layers(self, expert):
        layers = OrderedDict()
        for name, module in expert.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                layers[name] = module
        return layers

    def compute_sensitivity(self, calibration_loader, num_samples=100):
        expert_names = ["temporal", "flow", "frequency", "blending"]
        all_layers = []

        for name in expert_names:
            layers = self._get_expert_layers(self.model.experts[name])
            all_layers.append(layers)

        num_layers = max(len(l) for l in all_layers)
        sensitivity_matrix = np.zeros((4, num_layers))

        self.model.eval()
        sample_count = 0

        with torch.no_grad():
            for batch in calibration_loader:
                if sample_count >= num_samples:
                    break

                frames = batch["frames"].to(self.device)

                for i, name in enumerate(expert_names):
                    expert = self.model.experts[name]
                    layers = all_layers[i]

                    full_output = expert(self.model.temporal_head(
                        self.model.stem(self.model.preprocessor(frames)["rgb"])
                    ))

                    for j, (layer_name, layer) in enumerate(layers.items()):
                        original_weight = layer.weight.data.clone()

                        noise = torch.randn_like(original_weight) * 0.01
                        layer.weight.data = original_weight + noise

                        noisy_output = expert(self.model.temporal_head(
                            self.model.stem(self.model.preprocessor(frames)["rgb"])
                        ))

                        layer.weight.data = original_weight

                        diff = (full_output - noisy_output).abs().mean().item()
                        sensitivity_matrix[i, j] += diff

                sample_count += frames.size(0)

        sensitivity_matrix /= max(sample_count, 1)
        sensitivity_matrix = sensitivity_matrix / (sensitivity_matrix.max(axis=1, keepdims=True) + 1e-8)

        return sensitivity_matrix

    def save_sensitivity(self, sensitivity_matrix, path):
        np.save(path, sensitivity_matrix)

    def load_sensitivity(self, path):
        return np.load(path)