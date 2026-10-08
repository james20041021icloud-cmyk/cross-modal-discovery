"""
Reconstruction quality verification for both Soft-VQ-VAEs.

Loads trained checkpoints, reconstructs random frames + audio windows,
and reports:
  - per-sample MSE on a held-out set
  - side-by-side visual reconstruction sheets
  - orig / recon WAV pairs
  - code-usage frequency histogram (any concentration = mode collapse)
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import soundfile as sf
import matplotlib.pyplot as plt
import os
from pathlib import Path

from vqvae_model import SoftVQVAE
from vqvae_audio import AudioSoftVQVAE

FRAMES = Path(os.environ.get("MMVQ_ROOT", ".")) / "data/frames"
AUDIO  = Path(os.environ.get("MMVQ_ROOT", ".")) / "data/audio.wav"
V_CKPT = Path(os.environ.get("MMVQ_ROOT", ".")) / "models/visual/visual_vqvae.pt"
A_CKPT = Path(os.environ.get("MMVQ_ROOT", ".")) / "models/audio/audio_vqvae.pt"
TOK    = Path(os.environ.get("MMVQ_ROOT", ".")) / "tokens"
OUT    = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/verify"
OUT.mkdir(parents=True, exist_ok=True)

SR = 16000; CLIP = 16000  # 1-sec audio


def load_visual(device):
    ckpt = torch.load(V_CKPT, map_location=device, weights_only=False)
    a = ckpt["args"]
    m = SoftVQVAE(in_channels=3, hidden=a["hidden"], latent_dim=a["latent"],
                  num_embeddings=a["codebook"]).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, a


def load_audio(device):
    ckpt = torch.load(A_CKPT, map_location=device, weights_only=False)
    a = ckpt["args"]
    m = AudioSoftVQVAE(in_channels=1, hidden=a["hidden"], latent_dim=a["latent"],
                       num_embeddings=a["codebook"]).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, a


@torch.no_grad()
def visual_verify(device):
    print("\n══ Visual VQ-VAE verification ══════════════════════════════")
    model, args = load_visual(device)
    size = args["size"]
    K    = args["codebook"]

    tf = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    paths = sorted(FRAMES.glob("frame_*.jpg"))
    T = len(paths)
    print(f"  Model: hidden={args['hidden']}, latent={args['latent']}, K={K}")
    print(f"  Frames available: {T}")

    # ── Reconstruction on 12 diverse frames evenly spaced across the video
    idxs = np.linspace(0, T - 1, 12, dtype=int)
    xs = torch.stack([tf(Image.open(paths[i]).convert("RGB")) for i in idxs]).to(device)
    recon, vq_loss, ppl, tok = model(xs, temperature=0.1)
    per_sample_mse = ((recon - xs) ** 2).mean(dim=[1, 2, 3])
    print(f"\n  Reconstruction MSE (12 evenly-spaced test frames):")
    for k, (i, m) in enumerate(zip(idxs, per_sample_mse.cpu())):
        print(f"    frame {i:4d}  t={i/4:.1f}s   MSE = {m.item():.4f}")
    print(f"  Mean MSE across sample = {per_sample_mse.mean().item():.4f}")
    print(f"  Batch perplexity        = {ppl.item():.1f} / {K} "
          f"({100*ppl.item()/K:.1f}%)")

    # Side-by-side grid: originals top row, recon bottom row
    grid = torch.cat([xs, recon.float()])
    save_image(grid * 0.5 + 0.5, OUT / "visual_recon_grid.png", nrow=12)
    print(f"  → grid saved: {OUT/'visual_recon_grid.png'}")

    # ── Code usage from the pre-computed tokens (real evidence, not sampled)
    v_tok = np.load(TOK / "visual_tokens.npy")   # (T, 32, 32)
    usage = np.bincount(v_tok.flatten(), minlength=K)
    usage_frac = usage / usage.sum()
    sorted_frac = np.sort(usage_frac)[::-1]
    top10_share  = sorted_frac[:10].sum()
    top50_share  = sorted_frac[:50].sum()
    codes_used   = int((usage > 0).sum())
    print(f"\n  Code usage across {T} × 32 × 32 = {T*1024:,} tokens:")
    print(f"    codes used at least once : {codes_used}/{K} ({100*codes_used/K:.1f}%)")
    print(f"    top-10 codes share      : {100*top10_share:.1f}%  "
          f"(uniform would be {100*10/K:.1f}%)")
    print(f"    top-50 codes share      : {100*top50_share:.1f}%  "
          f"(uniform would be {100*50/K:.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(K), np.sort(usage_frac)[::-1] * 100)
    ax.set_xlabel("code rank (sorted)")
    ax.set_ylabel("% of all tokens")
    ax.set_title(f"Visual codebook usage — K={K} entries "
                 f"({codes_used} non-empty; top-10 = {100*top10_share:.1f}%)")
    ax.set_yscale("log"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "visual_code_usage.png", dpi=140)
    plt.close()
    print(f"  → usage plot: {OUT/'visual_code_usage.png'}")
    return per_sample_mse.mean().item(), codes_used, K


@torch.no_grad()
def audio_verify(device):
    print("\n══ Audio VQ-VAE verification ═══════════════════════════════")
    model, args = load_audio(device)
    K = args["codebook"]

    wav, sr = sf.read(AUDIO); wav = wav.astype(np.float32)
    if wav.ndim > 1: wav = wav.mean(1)
    peak = np.abs(wav).max()
    if peak > 0: wav = wav / peak * 0.95
    print(f"  Model: hidden={args['hidden']}, latent={args['latent']}, K={K}")
    print(f"  Audio duration: {len(wav)/SR:.1f} s")

    # 8 evenly-spaced 1-sec clips
    n = 8
    starts = np.linspace(0, len(wav) - CLIP, n, dtype=int)
    xs = torch.stack([torch.from_numpy(wav[s:s+CLIP]) for s in starts]).unsqueeze(1).to(device)
    recon, vq_loss, ppl, tok = model(xs, temperature=0.1)
    per_sample_mse = ((recon - xs) ** 2).mean(dim=[1, 2])

    print(f"\n  Reconstruction MSE ({n} evenly-spaced 1-sec clips):")
    for i, (s, m) in enumerate(zip(starts, per_sample_mse.cpu())):
        print(f"    clip start={s/SR:5.1f}s   MSE = {m.item():.5f}")
    print(f"  Mean MSE across sample = {per_sample_mse.mean().item():.5f}")
    print(f"  Batch perplexity        = {ppl.item():.1f} / {K} "
          f"({100*ppl.item()/K:.1f}%)")

    # Dump orig + recon wavs
    audio_dir = OUT / "audio_pairs"; audio_dir.mkdir(exist_ok=True)
    for i, s in enumerate(starts):
        sf.write(audio_dir / f"orig_{i+1:02d}_t{s/SR:.0f}s.wav",
                 xs[i, 0].float().cpu().numpy(), SR)
        sf.write(audio_dir / f"recon_{i+1:02d}_t{s/SR:.0f}s.wav",
                 recon[i, 0].float().cpu().numpy(), SR)
    print(f"  → {n} orig/recon wav pairs in {audio_dir}/")

    # Waveform overlay plot for first 3 clips
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for k in range(3):
        ax = axes[k]
        t = np.arange(CLIP) / SR
        ax.plot(t, xs[k, 0].cpu().numpy(), lw=0.5, alpha=0.7, label="orig")
        ax.plot(t, recon[k, 0].float().cpu().numpy(), lw=0.5, alpha=0.7, label="recon")
        ax.set_title(f"clip start={starts[k]/SR:.0f}s  MSE={per_sample_mse[k].item():.5f}",
                     fontsize=10)
        ax.set_ylim(-1.1, 1.1); ax.grid(alpha=0.3)
        if k == 0: ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(OUT / "audio_recon_waveforms.png", dpi=140)
    plt.close()
    print(f"  → waveform overlay: {OUT/'audio_recon_waveforms.png'}")

    # Code usage
    a_tok = np.load(TOK / "audio_tokens.npy")    # (T, 1000)
    usage = np.bincount(a_tok.flatten(), minlength=K)
    usage_frac = usage / usage.sum()
    sorted_frac = np.sort(usage_frac)[::-1]
    codes_used = int((usage > 0).sum())
    print(f"\n  Code usage across {a_tok.shape[0]} × 1000 = {a_tok.size:,} tokens:")
    print(f"    codes used at least once : {codes_used}/{K} ({100*codes_used/K:.1f}%)")
    print(f"    top-10 codes share       : {100*sorted_frac[:10].sum():.1f}%  "
          f"(uniform would be {100*10/K:.1f}%)")
    print(f"    top-50 codes share       : {100*sorted_frac[:50].sum():.1f}%  "
          f"(uniform would be {100*50/K:.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(K), np.sort(usage_frac)[::-1] * 100)
    ax.set_xlabel("code rank"); ax.set_ylabel("% of all tokens")
    ax.set_title(f"Audio codebook usage — K={K} entries "
                 f"({codes_used} non-empty; top-10 = {100*sorted_frac[:10].sum():.1f}%)")
    ax.set_yscale("log"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "audio_code_usage.png", dpi=140)
    plt.close()
    print(f"  → usage plot: {OUT/'audio_code_usage.png'}")
    return per_sample_mse.mean().item(), codes_used, K


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    v_mse, v_used, v_K = visual_verify(device)
    a_mse, a_used, a_K = audio_verify(device)

    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    print(f"  Visual : recon MSE = {v_mse:.4f}   codes used {v_used}/{v_K} "
          f"({100*v_used/v_K:.1f}%)")
    print(f"  Audio  : recon MSE = {a_mse:.5f}   codes used {a_used}/{a_K} "
          f"({100*a_used/a_K:.1f}%)")
    print(f"\n  All verification artifacts in {OUT}/")


if __name__ == "__main__":
    main()
