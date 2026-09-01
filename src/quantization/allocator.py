import numpy as np
import torch


class DynamicBitAllocator:
    def __init__(self, sensitivity_matrix, b_base=4, b_max=16, bit_widths=None):
        self.sensitivity_matrix = sensitivity_matrix
        self.b_base = b_base
        self.b_max = b_max
        self.bit_widths = bit_widths or [4, 8, 16]

    def allocate(self, w, difficulty=None):
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy()

        if isinstance(difficulty, torch.Tensor):
            difficulty = difficulty.detach().cpu().numpy()

        if w.ndim == 2:
            w = w[0]

        if difficulty is not None:
            difficulty = float(np.asarray(difficulty).reshape(-1)[0])

        s_j = np.dot(w, self.sensitivity_matrix)

        if difficulty is not None:
            s_j = s_j * (1.0 + np.clip(difficulty, 0.0, 1.0))

        mu = np.mean(s_j)
        sigma = np.std(s_j) + 1e-8

        normalized = (s_j - mu) / sigma

        bit_ratios = 1.0 / (1.0 + np.exp(-normalized))

        bit_widths = np.zeros_like(s_j, dtype=int)
        for i, ratio in enumerate(bit_ratios):
            idx = int(ratio * (len(self.bit_widths) - 1) + 0.5)
            idx = np.clip(idx, 0, len(self.bit_widths) - 1)
            bit_widths[i] = self.bit_widths[idx]

        return bit_widths

    def get_quantization_config(self, w, difficulty=None):
        bit_widths = self.allocate(w, difficulty=difficulty)

        config = {
            "stem": "FP16" if difficulty is not None and float(np.asarray(difficulty).reshape(-1)[0]) >= 0.6 else "INT8",
            "temporal_expert": {},
            "flow_expert": {},
            "frequency_expert": {},
            "blending_expert": {},
        }

        expert_names = ["temporal_expert", "flow_expert", "frequency_expert", "blending_expert"]

        for i, name in enumerate(expert_names):
            layers_per_expert = len(bit_widths) // 4
            start_idx = i * layers_per_expert
            end_idx = start_idx + layers_per_expert

            layer_config = {}
            for j in range(start_idx, min(end_idx, len(bit_widths))):
                bw = int(bit_widths[j])
                if bw <= 4:
                    layer_config[f"layer_{j - start_idx}"] = "INT4"
                elif bw <= 8:
                    layer_config[f"layer_{j - start_idx}"] = "INT8"
                else:
                    layer_config[f"layer_{j - start_idx}"] = "FP16"

            if layer_config:
                config[name] = layer_config

        return config
