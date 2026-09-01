import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedSSM(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.x_proj = nn.Linear(d_inner, d_state + 1)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)

        A = torch.arange(1, d_state + 1).float().view(1, -1)
        self.register_buffer("A_log", torch.log(A))

    def forward(self, x):
        # 注：FlowExpert.forward 已全程禁用 autocast，此处输入应为 fp32。
        # SSM 的 softplus/exp/循环累积在 fp16 下会溢出，依赖外层 autocast 保护。
        B, L, D = x.shape
        d_inner = D * self.expand

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        A = -torch.exp(self.A_log.float())
        dt = F.softplus(self.dt_proj(x))
        Bx = self.x_proj(x)
        B_val, C_val = Bx[:, :, :self.d_state], Bx[:, :, self.d_state:self.d_state + 1]

        y = torch.zeros(B, L, d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(L):
            dt_t = dt[:, t:t+1, :]
            B_t = B_val[:, t:t+1, :]
            C_t = C_val[:, t:t+1, :]
            x_t = x[:, t:t+1, :]

            dA = torch.exp(torch.einsum("bd,ds->bds", dt_t.squeeze(1), A))
            dB = torch.einsum("bd,bn->bdn", dt_t.squeeze(1), B_t.squeeze(1))
            h = h * dA + torch.einsum("bd,bdn->bdn", x_t.squeeze(1), dB)
            y_t = torch.einsum("bdn,bn->bd", h, C_t.squeeze(1))
            y[:, t, :] = y_t

        y = y * F.silu(z)
        y = self.out_proj(y)
        return y


class FlowExpert(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=256, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.ssm = SimplifiedSSM(d_model=hidden_dim, d_state=16, expand=2)

        self.norm = nn.LayerNorm(hidden_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 整个 FlowExpert 在 fp32 下计算：SSM 的 softplus/exp/循环累积在 fp16 下易溢出，
        # 且 SSM 输出转回 fp16 时可能超出 fp16 范围（>65504）导致 inf。
        # 与 FrequencyExpert 对 FFT 的处理一致，全程禁用 autocast。
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            # x can be (B, C) or (B, T, C) or (B, C, H, W)
            if x.dim() == 2:
                # (B, C) -> (B, 1, C)
                x = x.unsqueeze(1)
            elif x.dim() == 4:
                # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
                B, C, H, W = x.shape
                x = x.view(B, C, H * W).transpose(1, 2)
            elif x.dim() == 3:
                pass  # Already (B, T, C)
            else:
                raise ValueError(f"Unexpected input dimension: {x.dim()}")

            x = self.input_proj(x)
            x = self.ssm(x)

            x = self.norm(x)
            x = x.transpose(1, 2)
            x = self.pool(x).squeeze(-1)
            x = self.dropout(x)
            x = self.fc(x)
        return x
