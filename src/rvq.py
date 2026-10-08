"""
Residual Vector Quantizer — modality-agnostic.

Wraps N sequential SoftVectorQuantizer layers where each level quantizes the
residual left over by the previous levels.  The forward pass returns the summed
quantized output (straight-through to the encoder) plus per-level losses,
perplexities, and code indices.

Works for 2-D features (images: shape (B, D, H, W)) and 1-D features
(audio: shape (B, D, T)) — the underlying SoftVectorQuantizer already handles
both because it operates on the flattened (N, D) tensor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftVectorQuantizerBase(nn.Module):
    """
    Same hybrid quantizer as vqvae_model.SoftVectorQuantizer, but exposes an
    additional `residual_forward()` method so a stack of them can be composed
    into an RVQ.  Kept as a fresh copy here so the RVQ file is self-contained.
    """
    def __init__(self, num_embeddings, embedding_dim,
                 commitment_cost=0.25, soft_weight=0.1,
                 decay=0.99, epsilon=1e-5, dead_thresh=0.01):
        super().__init__()
        self.K, self.D = num_embeddings, embedding_dim
        self.commitment_cost = commitment_cost
        self.soft_weight = soft_weight
        self.decay, self.epsilon, self.dead_thresh = decay, epsilon, dead_thresh
        embed = torch.randn(num_embeddings, embedding_dim) * 0.1
        self.register_buffer('embedding', embed)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed', embed.clone())

    def _step(self, z_flat, temperature):
        """One quantization step on a flat (N, D) tensor. Returns hard z_q and
        auxiliary quantities. No straight-through applied here — that's done at
        the RVQ boundary."""
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.embedding.t()
                + self.embedding.pow(2).sum(1))
        idx = dist.argmin(1)
        one_hot = F.one_hot(idx, self.K).type(z_flat.dtype)
        z_q_hard = self.embedding[idx]                        # (N, D)
        soft_assign = F.softmax(-dist / temperature, dim=1)

        if self.training:
            with torch.no_grad():
                z32   = z_flat.float()
                oh32  = one_hot.float()
                cs    = oh32.sum(0)
                ema_w = oh32.t() @ z32
                self.ema_cluster_size.mul_(self.decay).add_(cs,    alpha=1 - self.decay)
                self.ema_embed       .mul_(self.decay).add_(ema_w, alpha=1 - self.decay)
                n = self.ema_cluster_size.sum()
                stab = (self.ema_cluster_size + self.epsilon) \
                       / (n + self.K * self.epsilon) * n
                new_embed = self.ema_embed / stab.unsqueeze(1)
                dead = self.ema_cluster_size < self.dead_thresh
                if dead.any():
                    rand_idx = torch.randint(0, z32.shape[0], (int(dead.sum()),),
                                             device=z_flat.device)
                    new_embed[dead] = z32[rand_idx]
                    self.ema_cluster_size[dead] = 1.0
                self.embedding.data.copy_(new_embed)

        commit = self.commitment_cost * F.mse_loss(z_q_hard.detach(), z_flat)
        soft_target = soft_assign @ self.embedding
        soft = self.soft_weight * F.mse_loss(soft_target.detach(), z_flat)
        vq_loss = commit + soft
        avg = one_hot.mean(0)
        ppl = torch.exp(-(avg * (avg + 1e-10).log()).sum())
        return z_q_hard, vq_loss, ppl, idx


class ResidualSoftVQ(nn.Module):
    """
    N-level Residual Vector Quantizer.

    Forward:
      z            → shape (B, D, H, W) for 2-D or (B, D, T) for 1-D
      temperature  → soft-assignment temperature (same for every level)

    Returns:
      z_q_st       — straight-through output for the decoder (same shape as z)
      vq_loss      — mean of per-level VQ losses
      perplexities — list of per-level perplexities
      indices      — list of per-level index tensors, each shape matches
                     spatial layout of z (e.g. (B, H, W) or (B, T_lat))
    """
    def __init__(self, num_levels, num_embeddings, embedding_dim, **kw):
        super().__init__()
        self.levels = nn.ModuleList([
            SoftVectorQuantizerBase(num_embeddings, embedding_dim, **kw)
            for _ in range(num_levels)
        ])
        self.num_levels = num_levels
        self.K = num_embeddings

    def forward(self, z, temperature=1.0):
        # Detect layout: 2-D (B,D,H,W) or 1-D (B,D,T)
        if z.dim() == 4:
            B, D, H, W = z.shape
            spatial_shape = (B, H, W)
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)
        elif z.dim() == 3:
            B, D, T = z.shape
            spatial_shape = (B, T)
            z_flat = z.permute(0, 2, 1).contiguous().view(-1, D)
        else:
            raise ValueError(f"unexpected z.dim() = {z.dim()}")

        residual = z_flat.clone()
        total_hard = torch.zeros_like(z_flat)
        loss_sum = 0.0
        perplexities = []
        indices = []

        for vq in self.levels:
            q_hard, loss, ppl, idx = vq._step(residual, temperature)
            total_hard = total_hard + q_hard
            residual = residual - q_hard.detach()          # move to next level's target
            loss_sum = loss_sum + loss
            perplexities.append(ppl)
            indices.append(idx.view(*spatial_shape))       # reshape back to grid

        # Straight-through from total_hard to encoder output z_flat
        z_q_st_flat = z_flat + (total_hard - z_flat).detach()

        # Back to the original layout
        if z.dim() == 4:
            z_q_st = z_q_st_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        else:
            z_q_st = z_q_st_flat.view(B, T, D).permute(0, 2, 1).contiguous()

        return z_q_st, loss_sum / self.num_levels, perplexities, indices
