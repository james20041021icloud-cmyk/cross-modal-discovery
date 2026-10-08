"""
Soft-VQ-VAE for raw audio (1D version).
Same hard-EMA + soft-commitment + dead-code-revival quantizer as the image model.
Encoder/decoder use 1D convs over waveform samples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class AudioEncoder(nn.Module):
    """4× downsampling on the time axis (16 kHz → 1 kHz effective code rate)."""
    def __init__(self, in_ch=1, hidden=128, latent_dim=64, n_res=3):
        super().__init__()
        c = hidden
        layers = [
            nn.Conv1d(in_ch, c//4, 4, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv1d(c//4, c//2, 4, stride=2, padding=1),   # /4
            nn.ReLU(inplace=True),
            nn.Conv1d(c//2, c,    4, stride=2, padding=1),   # /8
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c, 4, stride=2, padding=1),         # /16
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c, 3, padding=1),
        ]
        for _ in range(n_res):
            layers.append(ResBlock1D(c))
        layers += [nn.ReLU(inplace=True), nn.Conv1d(c, latent_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AudioDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden=128, out_ch=1, n_res=3):
        super().__init__()
        c = hidden
        layers = [nn.Conv1d(latent_dim, c, 3, padding=1)]
        for _ in range(n_res):
            layers.append(ResBlock1D(c))
        layers += [
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c, c, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c, c//2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c//2, c//4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c//4, out_ch, 4, stride=2, padding=1),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SoftVectorQuantizer1D(nn.Module):
    """Same hybrid quantizer as the image model, operating on (B, D, T)."""
    def __init__(self, num_embeddings=512, embedding_dim=64,
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

    def forward(self, z, temperature=1.0):
        B, D, T = z.shape
        z_flat = z.permute(0, 2, 1).contiguous().view(-1, D)
        N = z_flat.shape[0]

        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.embedding.t()
                + self.embedding.pow(2).sum(1))

        idx = dist.argmin(1)
        one_hot = F.one_hot(idx, self.K).type(z_flat.dtype)
        z_q = self.embedding[idx]
        soft_assign = F.softmax(-dist / temperature, dim=1)

        if self.training:
            with torch.no_grad():
                z32 = z_flat.float()
                oh32 = one_hot.float()
                cluster_size = oh32.sum(0)
                ema_w = oh32.t() @ z32
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.ema_embed.mul_(self.decay).add_(ema_w, alpha=1 - self.decay)
                n = self.ema_cluster_size.sum()
                stabilised = (self.ema_cluster_size + self.epsilon) \
                             / (n + self.K * self.epsilon) * n
                new_embed = self.ema_embed / stabilised.unsqueeze(1)
                dead = self.ema_cluster_size < self.dead_thresh
                if dead.any():
                    rand_idx = torch.randint(0, N, (int(dead.sum()),), device=z.device)
                    new_embed[dead] = z32[rand_idx]
                    self.ema_cluster_size[dead] = 1.0
                self.embedding.data.copy_(new_embed)

        commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_flat)
        soft_target = soft_assign @ self.embedding
        soft_loss = self.soft_weight * F.mse_loss(soft_target.detach(), z_flat)
        vq_loss = commitment_loss + soft_loss

        z_q_st = z_flat + (z_q - z_flat).detach()
        z_q_st = z_q_st.view(B, T, D).permute(0, 2, 1).contiguous()

        avg_probs = one_hot.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        return z_q_st, vq_loss, perplexity, idx.view(B, T)


class AudioSoftVQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden=128, latent_dim=64,
                 num_embeddings=512, commitment_cost=0.25, soft_weight=0.1):
        super().__init__()
        self.encoder = AudioEncoder(in_channels, hidden, latent_dim)
        self.vq = SoftVectorQuantizer1D(num_embeddings, latent_dim,
                                        commitment_cost, soft_weight)
        self.decoder = AudioDecoder(latent_dim, hidden, in_channels)

    def forward(self, x, temperature=1.0):
        z = self.encoder(x)
        z_q, vq_loss, perplexity, idx = self.vq(z, temperature)
        x_recon = self.decoder(z_q)
        # Match output length to input length if rounding caused a mismatch
        if x_recon.shape[-1] != x.shape[-1]:
            x_recon = x_recon[..., :x.shape[-1]]
        return x_recon, vq_loss, perplexity, idx
