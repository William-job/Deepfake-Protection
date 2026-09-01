import torch
import torch.nn as nn
import torch.nn.functional as F


class ArtifactFusion(nn.Module):
    def __init__(self, latent_dim=128, latent_k=4, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_k = latent_k

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.compress = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, latent_k),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for module in self.compress:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, h_t, h_f, h_phi, h_b):
        x = torch.stack([h_t, h_f, h_phi, h_b], dim=1)

        x = self.transformer(x)

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)

        z = self.compress(x)

        return z