"""
Step 3: Encode every frame + every audio window into codebook indices, and
build time-aligned histograms suitable for Graphical Lasso.

Outputs (all under out_dir/):
  visual_tokens.npy   (T, H_lat, W_lat) — codebook index per latent position
  audio_tokens.npy    (T, L_lat)         — codebook index per latent position
  visual_hist.npy     (T, K_v)           — code count histogram per frame
  audio_hist.npy      (T, K_a)           — code count histogram per audio-second
  timeline.csv        frame-index → time-sec bookkeeping
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import soundfile as sf

from vqvae_model import SoftVQVAE
from vqvae_audio import AudioSoftVQVAE


SAMPLE_RATE = 16000
CLIP_LEN    = 16000  # 1-second window


def load_visual(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = SoftVQVAE(in_channels=3, hidden=a["hidden"],
                      latent_dim=a["latent"], num_embeddings=a["codebook"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, a


def load_audio(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = AudioSoftVQVAE(in_channels=1, hidden=a["hidden"],
                           latent_dim=a["latent"], num_embeddings=a["codebook"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, a


@torch.no_grad()
def encode_frames(model, size, frames_dir, device):
    tf = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
    all_tokens = []
    for p in paths:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        z = model.encoder(x)
        _, _, _, idx = model.vq(z, temperature=0.1)
        all_tokens.append(idx[0].cpu().numpy())    # (H_lat, W_lat)
    return np.stack(all_tokens, axis=0)


@torch.no_grad()
def encode_audio_windows(model, wav_path, device, hop=CLIP_LEN):
    """Slide a CLIP_LEN window across the audio with `hop` stride."""
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1: wav = wav.mean(1)
    assert sr == SAMPLE_RATE
    wav = wav.astype(np.float32)
    peak = np.abs(wav).max()
    if peak > 0: wav = wav / peak * 0.95

    tokens = []
    for start in range(0, max(1, len(wav) - CLIP_LEN + 1), hop):
        clip = wav[start:start+CLIP_LEN]
        if len(clip) < CLIP_LEN:
            clip = np.pad(clip, (0, CLIP_LEN - len(clip)))
        x = torch.from_numpy(clip[None, None]).to(device)  # (1,1,T)
        z = model.encoder(x)
        _, _, _, idx = model.vq(z, temperature=0.1)
        tokens.append(idx[0].cpu().numpy())               # (L_lat,)
    return np.stack(tokens, axis=0)


def to_histogram(tokens, K):
    """tokens: (T, ...) integer array in [0,K)  →  (T, K) count histogram."""
    T = tokens.shape[0]
    flat = tokens.reshape(T, -1)
    hist = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        vals, cnts = np.unique(flat[t], return_counts=True)
        hist[t, vals] = cnts
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir",  required=True)
    ap.add_argument("--audio_path",  required=True)
    ap.add_argument("--visual_ckpt", required=True)
    ap.add_argument("--audio_ckpt",  required=True)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--fps",         type=float, default=2.0,
                    help="fps used when frames were extracted")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Loading models ...")
    vmodel, va = load_visual(args.visual_ckpt, device)
    amodel, aa = load_audio (args.audio_ckpt,  device)
    K_v, K_a  = va["codebook"], aa["codebook"]
    size_v    = va["size"]

    print("Encoding frames ...")
    v_tok = encode_frames(vmodel, size_v, args.frames_dir, device)
    print(f"  visual tokens: {v_tok.shape}  ({v_tok.dtype})")

    # For per-frame audio histograms we hop by (1/fps) seconds not by 1 s.
    hop = max(1, int(SAMPLE_RATE / args.fps))
    print(f"Encoding audio windows (hop = {hop/SAMPLE_RATE:.2f} s) ...")
    a_tok = encode_audio_windows(amodel, args.audio_path, device, hop=hop)
    print(f"  audio  tokens: {a_tok.shape}")

    # Align lengths
    T = min(v_tok.shape[0], a_tok.shape[0])
    v_tok, a_tok = v_tok[:T], a_tok[:T]
    print(f"Aligned length: {T} time bins")

    # Histograms
    v_hist = to_histogram(v_tok, K_v)
    a_hist = to_histogram(a_tok, K_a)
    print(f"Visual histogram: {v_hist.shape}   (K_v = {K_v})")
    print(f"Audio  histogram: {a_hist.shape}   (K_a = {K_a})")

    # Timeline
    times = np.arange(T) / args.fps
    pd.DataFrame({"time_sec": times}).to_csv(out/"timeline.csv", index=False)

    np.save(out/"visual_tokens.npy", v_tok)
    np.save(out/"audio_tokens.npy",  a_tok)
    np.save(out/"visual_hist.npy",   v_hist)
    np.save(out/"audio_hist.npy",    a_hist)

    # Summary
    print("\n── Codebook usage summary ──")
    print(f"Visual codes used: {(v_hist.sum(0) > 0).sum()} / {K_v} "
          f"({100*(v_hist.sum(0)>0).mean():.1f}%)")
    print(f"Audio codes used : {(a_hist.sum(0) > 0).sum()} / {K_a} "
          f"({100*(a_hist.sum(0)>0).mean():.1f}%)")
    print(f"\n✅ Saved token streams to {out}/")


if __name__ == "__main__":
    main()
