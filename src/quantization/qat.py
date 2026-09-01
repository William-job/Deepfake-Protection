import torch
import torch.nn as nn
import torch.nn.functional as F


class QATRegularizer:
    def __init__(self, model, min_bit_width=4, max_bit_width=8,
                 noise_scale=0.02, stem_protect=True):
        self.model = model
        self.min_bit_width = min_bit_width
        self.max_bit_width = max_bit_width
        self.noise_scale = noise_scale
        self.stem_protect = stem_protect

    def compute_regularization_loss(self, difficulty=None):
        loss = 0.0
        count = 0

        if difficulty is not None and torch.is_tensor(difficulty):
            bit_width = self.min_bit_width + (self.max_bit_width - self.min_bit_width) * (1.0 - difficulty.mean())
            bit_width = max(self.min_bit_width, min(self.max_bit_width, int(bit_width + 0.5)))
        else:
            bit_width = self.max_bit_width

        levels = 2 ** bit_width

        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                continue
            if self.stem_protect and "stem" in name:
                continue

            w = module.weight
            w_min, w_max = w.min(), w.max()
            scale = (w_max - w_min) / (levels - 1) if levels > 1 else 1.0

            if scale > 1e-8:
                w_q = torch.round((w - w_min) / scale) * scale + w_min
                layer_loss = F.mse_loss(w, w_q)
                loss = loss + layer_loss
                count += 1

        if count > 0:
            loss = loss / count

        return loss, bit_width

    def inject_noise(self):
        frozen = {}
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if not isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                    continue
                if self.stem_protect and "stem" in name:
                    continue

                w = module.weight
                frozen[name] = w.data.clone()

                noise = torch.randn_like(w) * self.noise_scale * w.std()
                w.data.add_(noise)

        return frozen

    def restore_weights(self, frozen):
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name in frozen:
                    module.weight.data.copy_(frozen[name])

    def forward_with_noise(self, input_batch_fn):
        frozen = self.inject_noise()
        try:
            outputs = input_batch_fn()
        finally:
            self.restore_weights(frozen)
        return outputs
