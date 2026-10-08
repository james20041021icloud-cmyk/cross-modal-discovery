"""
RVQ token extractor: produces per-level tokens AND the summed histograms.

Outputs in tokens_rvq/:
  visual_tokens_L{i}.npy   (T, H, W)   per-level indices
  audio_tokens_L{i}.npy    (T, T_lat)  per-level indices
  visual_hist.npy          (T, num_levels * codes_per_level)  summed histogram
  audio_hist.npy           (T, num_levels * codes_per_level)  summed histogram
  timeline.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import soundfile as sf

from vqvae_rvq_models import ImageRVQVAE, AudioRVQVAE


SR, CLIP = 16000, 16000


def load(kind, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    if kind == "image":
        m = ImageRVQVAE(3, a["hidden"], a["latent"],
                        a["num_levels"], a["codes_per_level"]).to(device)
    else:
        m = AudioRVQVAE(1, a["hidden"], a["latent"],
                        a["num_levels"], a["codes_per_level"]).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, a


@torch.no_grad()
def encode_frames(model, size, frames_dir, device):
    tf = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
    per_level_lists = None
    for p in paths:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        z = model.encoder(x)
        _, _, _, idxs = model.vq(z, temperature=0.1)   # list of (1, H, W)
        if per_level_lists is None:
            per_level_lists = [[] for _ in idxs]
        for i, idx in enumerate(idxs):
            per_level_lists[i].append(idx[0].cpu().numpy())
    return [np.stack(lst, axis=0) for lst in per_level_lists]   # list of (T, H, W)


@torch.no_grad()
def encode_audio(model, wav_path, device, hop):
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1: wav = wav.mean(1)
    assert sr == SR
    wav = wav.astype(np.float32)
    peak = np.abs(wav).max()
    if peak > 0: wav = wav / peak * 0.95

    per_level_lists = None
    for start in range(0, max(1, len(wav) - CLIP + 1), hop):
        clip = wav[start:start+CLIP]
        if len(clip) < CLIP:
            clip = np.pad(clip, (0, CLIP - len(clip)))
        x = torch.from_numpy(clip[None, None]).to(device)
        z = model.encoder(x)
        _, _, _, idxs = model.vq(z, temperature=0.1)   # list of (1, T_lat)
        if per_level_lists is None:
            per_level_lists = [[] for _ in idxs]
        for i, idx in enumerate(idxs):
            per_level_lists[i].append(idx[0].cpu().numpy())
    return [np.stack(lst, axis=0) for lst in per_level_lists]   # list of (T, T_lat)


def to_hist(tokens, K):
    """(T, ...) → (T, K) count histogram."""
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
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Loading models ...")
    vmodel, va = load("image", args.visual_ckpt, device)
    amodel, aa = load("audio", args.audio_ckpt,  device)
    K_v = va["codes_per_level"]; L_v = va["num_levels"]
    K_a = aa["codes_per_level"]; L_a = aa["num_levels"]

    print(f"Visual: {L_v} levels × {K_v} codes each")
    print(f"Audio : {L_a} levels × {K_a} codes each")

    print("\nEncoding frames ...")
    v_levels = encode_frames(vmodel, va["size"], args.frames_dir, device)
    for i, arr in enumerate(v_levels):
        print(f"  visual level {i+1}: {arr.shape}")
        np.save(out / f"visual_tokens_L{i+1}.npy", arr)

    print("\nEncoding audio ...")
    hop = max(1, int(SR / args.fps))
    a_levels = encode_audio(amodel, args.audio_path, device, hop=hop)
    for i, arr in enumerate(a_levels):
        print(f"  audio  level {i+1}: {arr.shape}")
        np.save(out / f"audio_tokens_L{i+1}.npy", arr)

    T = min(min(a.shape[0] for a in v_levels), min(a.shape[0] for a in a_levels))
    v_levels = [a[:T] for a in v_levels]
    a_levels = [a[:T] for a in a_levels]
    print(f"\nAligned length: {T} bins")

    # Concatenate per-level histograms
    v_hist_parts = [to_hist(a, K_v) for a in v_levels]
    a_hist_parts = [to_hist(a, K_a) for a in a_levels]
    v_hist = np.concatenate(v_hist_parts, axis=1)   # (T, L_v * K_v)
    a_hist = np.concatenate(a_hist_parts, axis=1)   # (T, L_a * K_a)
    print(f"Visual concatenated hist: {v_hist.shape}")
    print(f"Audio  concatenated hist: {a_hist.shape}")

    np.save(out / "visual_hist.npy", v_hist)
    np.save(out / "audio_hist.npy",  a_hist)

    times = np.arange(T) / args.fps
    pd.DataFrame({"time_sec": times}).to_csv(out/"timeline.csv", index=False)

    print("\nCode usage per level:")
    for i, arr in enumerate(v_levels):
        used = len(np.unique(arr))
        print(f"  visual L{i+1}: {used} / {K_v} codes used")
    for i, arr in enumerate(a_levels):
        used = len(np.unique(arr))
        print(f"  audio  L{i+1}: {used} / {K_a} codes used")

    print(f"\n✅ Saved to {out}/")


if __name__ == "__main__":
    main()
