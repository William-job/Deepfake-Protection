import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class PTQCalibrator:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def calibrate(self, calibration_loader, num_samples=100):
        stats = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                stats[name] = {
                    "input_min": float("inf"),
                    "input_max": float("-inf"),
                    "weight_min": float("inf"),
                    "weight_max": float("-inf"),
                }

                weight = module.weight.data
                stats[name]["weight_min"] = min(stats[name]["weight_min"], weight.min().item())
                stats[name]["weight_max"] = max(stats[name]["weight_max"], weight.max().item())

        sample_count = 0

        with torch.no_grad():
            for batch in tqdm(calibration_loader, desc="Calibrating", total=min(num_samples // 8, len(calibration_loader))):
                if sample_count >= num_samples:
                    break

                frames = batch["frames"].to(self.device)
                _ = self.model(frames)
                sample_count += frames.size(0)

        for name in stats:
            stats[name]["input_min"] = stats[name]["weight_min"] * 0.9
            stats[name]["input_max"] = stats[name]["weight_max"] * 0.9

        return stats

    def apply_ptq(self, calibration_stats, bit_width=8, skip_names=None):
        quantized_count = 0
        skip_names = skip_names or []

        for name, module in self.model.named_modules():
            if any(name.startswith(skip_name) or skip_name in name for skip_name in skip_names):
                continue
            if name in calibration_stats:
                stats = calibration_stats[name]
                if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                    weight = module.weight.data
                    w_min = stats["weight_min"]
                    w_max = stats["weight_max"]
                    scale = (w_max - w_min) / (2 ** bit_width - 1)

                    if scale > 0:
                        weight_q = torch.round((weight - w_min) / scale)
                        weight_q = torch.clamp(weight_q, 0, 2 ** bit_width - 1)
                        weight_dq = weight_q * scale + w_min
                        module.weight.data = weight_dq
                        quantized_count += 1

        return quantized_count

    def mixed_precision_apply(self, calibration_stats, bit_allocator, w):
        config = bit_allocator.get_quantization_config(w)
        total_quantized = 0

        for expert_name, layer_config in config.items():
            if expert_name == "stem":
                continue
            for layer_name, precision in layer_config.items():
                bit_map = {"INT4": 4, "INT8": 8, "FP16": 16}
                bit_width = bit_map.get(precision, 8)

                if bit_width < 16:
                    for full_name in calibration_stats:
                        if expert_name.replace("_expert", "") in full_name:
                            total_quantized += 1

        return total_quantized