"""
Wrapper models: encoder + ResidualSoftVQ + decoder for image and audio.
Encoders/decoders reused from the original modules to keep architecture identical
apart from the RVQ swap.
"""
import torch
import torch.nn as nn

from vqvae_model  import Encoder      as ImgEncoder
from vqvae_model  import Decoder      as ImgDecoder
from vqvae_audio  import AudioEncoder, AudioDecoder
from rvq import ResidualSoftVQ


class ImageRVQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden=128, latent_dim=64,
                 num_levels=2, codes_per_level=256,
                 commitment_cost=0.25, soft_weight=0.1):
        super().__init__()
        self.encoder = ImgEncoder(in_channels, hidden, latent_dim)
        self.vq = ResidualSoftVQ(num_levels, codes_per_level, latent_dim,
                                 commitment_cost=commitment_cost,
                                 soft_weight=soft_weight)
        self.decoder = ImgDecoder(latent_dim, hidden, in_channels)
        self.num_levels     = num_levels
        self.codes_per_level = codes_per_level

    def forward(self, x, temperature=1.0):
        z = self.encoder(x)
        z_q, vq_loss, ppls, idxs = self.vq(z, temperature)
        recon = self.decoder(z_q)
        return recon, vq_loss, ppls, idxs


class AudioRVQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden=128, latent_dim=64,
                 num_levels=2, codes_per_level=128,
                 commitment_cost=0.25, soft_weight=0.1):
        super().__init__()
        self.encoder = AudioEncoder(in_channels, hidden, latent_dim)
        self.vq = ResidualSoftVQ(num_levels, codes_per_level, latent_dim,
                                 commitment_cost=commitment_cost,
                                 soft_weight=soft_weight)
        self.decoder = AudioDecoder(latent_dim, hidden, in_channels)
        self.num_levels      = num_levels
        self.codes_per_level = codes_per_level

    def forward(self, x, temperature=1.0):
        z = self.encoder(x)
        z_q, vq_loss, ppls, idxs = self.vq(z, temperature)
        recon = self.decoder(z_q)
        if recon.shape[-1] != x.shape[-1]:
            recon = recon[..., :x.shape[-1]]
        return recon, vq_loss, ppls, idxs
