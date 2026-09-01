import torch
import torch.nn as nn
import torch.nn.functional as F


class DifficultyEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, use_head_std=False, use_pairwise_cos=False, use_z_norm=False):
        super().__init__()
        self.use_head_std = use_head_std
        self.use_pairwise_cos = use_pairwise_cos
        self.use_z_norm = use_z_norm
        effective_dim = input_dim
        if use_head_std:
            effective_dim += 1
        if use_pairwise_cos:
            effective_dim += 6
        if use_z_norm:
            effective_dim += 1
        self.net = nn.Sequential(
            nn.Linear(effective_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, head_std=None, head_pairwise_cos=None, z_norm=None):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.shape[0]
        x = [z]
        if self.use_head_std and head_std is not None:
            if head_std.dim() == 0:
                head_std = head_std.unsqueeze(0).unsqueeze(0).expand(B, -1)
            elif head_std.dim() == 1:
                head_std = head_std.unsqueeze(-1)
            x.append(head_std)
        if self.use_pairwise_cos and head_pairwise_cos is not None:
            if head_pairwise_cos.dim() == 1:
                head_pairwise_cos = head_pairwise_cos.unsqueeze(0).expand(B, -1)
            x.append(head_pairwise_cos)
        if self.use_z_norm and z_norm is not None:
            if z_norm.dim() == 0:
                z_norm = z_norm.unsqueeze(0).unsqueeze(0).expand(B, -1)
            elif z_norm.dim() == 1:
                z_norm = z_norm.unsqueeze(-1)
            x.append(z_norm)
        return self.net(torch.cat(x, dim=-1))


class GatingNetwork(nn.Module):
    def __init__(self, latent_k=4, top_k=2, temperature=1.0, use_bias=True, min_temperature=0.2, max_temperature=2.0, difficulty_conditioned=False):
        super().__init__()
        self.latent_k = latent_k
        self.top_k = top_k
        self.use_bias = use_bias
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.difficulty_conditioned = difficulty_conditioned

        # Xavier uniform 初始化: bound = sqrt(6 / (fan_in + fan_out))
        # fan_in = latent_k, fan_out = 4。打破 softmax 对称性，避免 logits 恒为 0。
        # 依据: Glorot & Bengio (2010)，与 PyTorch nn.Linear 默认初始化一致。
        bound = (6.0 / (latent_k + 4)) ** 0.5
        self.weight = nn.Parameter(torch.empty(4, latent_k).uniform_(-bound, bound))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(4))
        else:
            self.register_buffer("bias", torch.zeros(4))

        self.beta = nn.Parameter(torch.tensor(1.0))
        self._temperature = temperature

        if difficulty_conditioned:
            self.difficulty_proj = nn.Linear(1, 4)
            # 使用 nn.Linear 默认初始化（Kaiming uniform），保证 difficulty_proj
            # 有非零输出，能在训练初期对 logits 产生差异化影响。

    def reset_parameters(self):
        """课程切换时重新初始化门控参数，打破可能积累的死状态。

        top_k=1 阶段门控梯度为 0，参数保持初始值不变。切换到 top_k=2 时若不重置，
        零初始化的死状态会被带入新阶段，导致未激活专家永远无法进入激活集。
        """
        bound = (6.0 / (self.latent_k + 4)) ** 0.5
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if isinstance(self.bias, nn.Parameter):
                self.bias.zero_()
            if self.difficulty_conditioned:
                self.difficulty_proj.reset_parameters()

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    def _resolve_temperature(self, difficulty):
        base = torch.as_tensor(self._temperature, device=self.weight.device, dtype=self.weight.dtype)
        if difficulty is None:
            return base

        if not torch.is_tensor(difficulty):
            difficulty = torch.tensor(difficulty, device=self.weight.device, dtype=self.weight.dtype)

        difficulty = difficulty.to(device=self.weight.device, dtype=self.weight.dtype)
        if difficulty.dim() == 0:
            difficulty = difficulty.unsqueeze(0)

        # difficulty（expert disagreement）驱动温度：分歧大=难样本→高温（平滑、多专家），
        # 分歧小=易样本→低温（锐化、少专家）。无需手工阈值。
        temperature = self.min_temperature + (self.max_temperature - self.min_temperature) * difficulty
        temperature = torch.clamp(temperature, min=self.min_temperature, max=self.max_temperature)
        return temperature

    def forward(self, z, difficulty=None, return_active_set=True, return_aux=False):
        logits = F.linear(z, self.weight, self.bias) * self.beta
        if difficulty is not None and self.difficulty_conditioned:
            logits = logits + self.difficulty_proj(difficulty)

        temperature = self._resolve_temperature(difficulty)
        if temperature.dim() == 0:
            temperature = temperature.view(1)
        if temperature.dim() == 1:
            temperature = temperature.unsqueeze(-1)

        w = F.softmax(logits / temperature, dim=-1)

        if return_active_set:
            # 难度自适应 Top-K（向量化实现，避免逐样本 Python 循环）：
            # difficulty（专家分歧）越大参与专家越多，无需手工阈值。
            # K_i = 1 + round(d_i * (Kmax-1))
            if difficulty is None:
                k_eff = torch.full((z.size(0),), self.top_k, device=z.device, dtype=torch.long)
            else:
                d = difficulty.view(-1).clamp(0.0, 1.0)
                k_eff = (1 + torch.round(d * (self.top_k - 1))).long()
            k_eff = k_eff.clamp(min=1, max=self.latent_k)

            # rank[b, j] = w[b, j] 在样本 b 内按从大到小的名次（0=最大）
            order = w.argsort(dim=-1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, torch.arange(self.latent_k, device=z.device).expand_as(order))
            # 保留名次 < k_eff[b] 的专家
            keep = rank < k_eff.unsqueeze(-1)
            w_sparse = w * keep.float()
            w_sparse = w_sparse / (w_sparse.sum(dim=-1, keepdim=True) + 1e-8)
            _, indices = torch.topk(w_sparse, k=self.top_k, dim=-1)  # 日志用 active_set
            if return_aux:
                return w_sparse, indices, {"temperature": temperature.squeeze(-1), "gate_logits": logits, "w_dense": w, "k_eff": k_eff}
            return w_sparse, indices

        if return_aux:
            return w, None, {"temperature": temperature.squeeze(-1), "gate_logits": logits, "w_dense": w}

        return w

    def get_artifact_composition(self, z):
        logits = F.linear(z, self.weight, self.bias)
        w = F.softmax(logits / self._temperature, dim=-1)
        return w.detach().cpu().tolist()

    def forward_with_temperature(self, z, temperature):
        self._temperature = temperature
        return self.forward(z)


class TemperatureScheduler:
    def __init__(self, start_temp=1.0, end_temp=0.1, total_epochs=30):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_epochs = total_epochs

    def get_temperature(self, epoch):
        if epoch >= self.total_epochs:
            return self.end_temp

        progress = epoch / self.total_epochs
        temperature = self.start_temp + (self.end_temp - self.start_temp) * progress
        return temperature

    def step(self, gating_network, epoch):
        temp = self.get_temperature(epoch)
        gating_network.temperature = temp
        return temp
