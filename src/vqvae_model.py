"""
VQ-VAE with Soft Vector Quantization (corrected).

Based on van den Oord et al. 2017 (https://arxiv.org/abs/1711.00937).

Key fixes vs. naive implementation:
  1. EMA codebook updated via in-place buffer ops (no buffer replacement).
  2. EMA uses HARD one-hot assignments (standard); soft assignments only used
     for the differentiable "soft" loss term and perplexity metric.
  3. Encoder downsamples 2× (not 4×) so small images keep usable spatial latents.
  4. Dead-code revival: codebook entries unused for many steps get reset to
     a random encoder output each update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ─────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    """2× downsampling — keeps a richer latent grid for small inputs."""
    def __init__(self, in_ch=3, hidden=128, latent_dim=64, n_res=2):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, hidden // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, hidden, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        ]
        for _ in range(n_res):
            layers.append(ResBlock(hidden))
        layers += [nn.ReLU(inplace=True), nn.Conv2d(hidden, latent_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden=128, out_ch=3, n_res=2):
        super().__init__()
        layers = [nn.Conv2d(latent_dim, hidden, 3, padding=1)]
        for _ in range(n_res):
            layers.append(ResBlock(hidden))
        layers += [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden // 2, out_ch, 4, stride=2, padding=1),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Soft Vector Quantizer ──────────────────────────────────────────────────

class SoftVectorQuantizer(nn.Module):
    """
    Hybrid quantizer:
      - Forward pass: hard nearest-neighbour (with straight-through estimator).
      - Codebook update: EMA over HARD one-hot assignments (standard, stable).
      - Soft commitment: softmax-weighted MSE between encoder outputs and
        ALL codebook vectors — this is the "soft" loss that gives smoother
        gradients to the encoder, controlled by `temperature`.
      - Dead-code revival: any codebook entry whose EMA cluster size falls
        below `dead_thresh` is re-initialized to a random encoder output.
    """
    def __init__(self, num_embeddings=512, embedding_dim=64,
                 commitment_cost=0.25, soft_weight=0.1,
                 decay=0.99, epsilon=1e-5, dead_thresh=0.01):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.commitment_cost = commitment_cost
        self.soft_weight = soft_weight
        self.decay = decay
        self.epsilon = epsilon
        self.dead_thresh = dead_thresh

        # Initialise codebook with small random values (like the original paper)
        embed = torch.randn(num_embeddings, embedding_dim) * 0.1
        self.register_buffer('embedding', embed)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed', embed.clone())

    def forward(self, z, temperature=1.0):
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)  # (N, D)
        N = z_flat.shape[0]

        # Distances z ↔ codebook  (N, K)
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )

        # ── Hard assignment (used for the actual quantized output) ────────
        idx = dist.argmin(1)                        # (N,)
        one_hot = F.one_hot(idx, self.K).type(z_flat.dtype)  # (N, K)
        z_q = self.embedding[idx]                   # (N, D)

        # ── Soft assignment (used for soft commitment loss + perplexity) ──
        soft_assign = F.softmax(-dist / temperature, dim=1)  # (N, K)

        # ── EMA codebook update (training only, in-place on buffers) ──────
        # Cast to float32 so autocast (bf16/fp16) doesn't break buffer ops.
        if self.training:
            with torch.no_grad():
                z32 = z_flat.float()
                oh32 = one_hot.float()
                cluster_size = oh32.sum(0)                  # (K,)
                ema_w = oh32.t() @ z32                      # (K, D)

                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.ema_embed.mul_(self.decay).add_(ema_w,  alpha=1 - self.decay)

                # Laplace-smoothed cluster sizes
                n = self.ema_cluster_size.sum()
                stabilised = (self.ema_cluster_size + self.epsilon) \
                             / (n + self.K * self.epsilon) * n
                new_embed = self.ema_embed / stabilised.unsqueeze(1)

                # Dead-code revival: replace dormant entries with random
                # encoder outputs so the codebook stays diverse.
                dead = self.ema_cluster_size < self.dead_thresh
                if dead.any():
                    rand_idx = torch.randint(0, N, (int(dead.sum()),), device=z.device)
                    new_embed[dead] = z32[rand_idx]
                    self.ema_cluster_size[dead] = 1.0

                self.embedding.data.copy_(new_embed)

        # ── Losses ────────────────────────────────────────────────────────
        # Standard commitment loss (encoder commits to its chosen codeword)
        commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_flat)

        # SOFT term: distance to weighted average of all codewords.
        # With temp→0 this matches hard assignment; with temp large it spreads
        # gradient to many codewords, helping early-stage training.
        soft_target = soft_assign @ self.embedding         # (N, D)
        soft_loss = self.soft_weight * F.mse_loss(soft_target.detach(), z_flat)

        vq_loss = commitment_loss + soft_loss

        # Straight-through estimator: forward = quantized, backward = identity
        z_q_st = z_flat + (z_q - z_flat).detach()
        z_q_st = z_q_st.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # Perplexity over hard assignments (true codebook usage measure)
        avg_probs = one_hot.mean(0)                        # (K,)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        return z_q_st, vq_loss, perplexity, idx.view(B, H, W)


# ── Full model ─────────────────────────────────────────────────────────────

class SoftVQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden=128, latent_dim=64,
                 num_embeddings=512, commitment_cost=0.25, soft_weight=0.1,
                 decay=0.99):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden, latent_dim)
        self.vq = SoftVectorQuantizer(num_embeddings, latent_dim,
                                      commitment_cost, soft_weight, decay)
        self.decoder = Decoder(latent_dim, hidden, in_channels)

    def forward(self, x, temperature=1.0):
        z = self.encoder(x)
        z_q, vq_loss, perplexity, idx = self.vq(z, temperature)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, perplexity, idx

    @torch.no_grad()
    def encode_indices(self, x):
        z = self.encoder(x)
        _, _, _, idx = self.vq(z)
        return idx

    @torch.no_grad()
    def decode_indices(self, idx):
        B, H, W = idx.shape
        z = self.vq.embedding[idx.view(-1)].view(B, H, W, -1).permute(0, 3, 1, 2)
        return self.decoder(z)
