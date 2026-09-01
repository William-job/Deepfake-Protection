import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SharedStem(nn.Module):
    def __init__(self, stem_type="efficientnet-b0", pretrained=True, freeze=False,
                 out_channels=None):
        super().__init__()
        self.stem_type = stem_type

        if stem_type == "efficientnet-b0":
            self.backbone = self._build_efficientnet_b0(pretrained)
            # Actual output channels from EfficientNet-B0 features_only
            self.out_channels = [16, 24, 40, 112]
        elif stem_type == "mobilenetv4-small":
            self.backbone = self._build_mobilenetv4_small(pretrained)
            self.out_channels = [16, 24, 48, 96]
        else:
            raise ValueError(f"Unknown stem type: {stem_type}")

        if freeze:
            self.freeze()

    def _build_efficientnet_b0(self, pretrained):
        model = timm.create_model("efficientnet_b0", pretrained=pretrained, features_only=True)
        return model

    def _build_mobilenetv4_small(self, pretrained):
        try:
            model = timm.create_model("mobilenetv4_conv_small", pretrained=pretrained, features_only=True)
        except Exception:
            model = self._build_efficientnet_b0(pretrained)
        return model

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            is_video = True
        else:
            is_video = False

        features = self.backbone(x)

        if isinstance(features, list):
            feat_dict = {}
            for i, feat in enumerate(features[:4]):
                if is_video:
                    B = feat.shape[0] // T
                    feat = feat.view(B, T, *feat.shape[1:])
                feat_dict[f"f_{i+1}"] = feat
        else:
            feat_dict = {"f_1": features}

        return feat_dict

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        return self
