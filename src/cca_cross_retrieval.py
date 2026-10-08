"""
CCA-based cross-modal retrieval / prediction.

Given a visual embedding V_t (from some target frame like an explosion),
CCA lets us:
  1) predict the audio embedding A_pred that should co-occur
  2) find nearest actual audio bin in the data → retrieve its 1-sec audio
  3) find which audio codebook entries best match A_pred (frequency proxy)
  4) run FFT on the retrieved audio to see the frequency content

Same in reverse: given an audio embedding → predict visual → retrieve frame.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import os
from pathlib import Path
from sklearn.cross_decomposition import CCA
import shutil

from vqvae_rvq_models import ImageRVQVAE, AudioRVQVAE

# ── Paths ───────────────────────────────────────────────────────────────
EMB    = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/embedding_glasso"
FRAMES = Path(os.environ.get("MMVQ_ROOT", ".")) / "data/frames"
AUDIO  = Path(os.environ.get("MMVQ_ROOT", ".")) / "data/audio.wav"
V_CKPT = Path(os.environ.get("MMVQ_ROOT", ".")) / "models_rvq/visual/image_rvq.pt"
A_CKPT = Path(os.environ.get("MMVQ_ROOT", ".")) / "models_rvq/audio/audio_rvq.pt"
OUT    = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/cca_retrieval"
OUT.mkdir(parents=True, exist_ok=True)

SR, FPS = 16000, 4.0


# ── Load everything ──────────────────────────────────────────────────────
print("Loading embeddings + models ...")
V_emb = np.load(EMB / "V_embeddings.npy")   # (T, 64)
A_emb = np.load(EMB / "A_embeddings.npy")   # (T, 64)
T, D = V_emb.shape

V = (V_emb - V_emb.mean(0)) / (V_emb.std(0) + 1e-9)
A = (A_emb - A_emb.mean(0)) / (A_emb.std(0) + 1e-9)

# Full-rank CCA
N_COMP = 32
cca = CCA(n_components=N_COMP, max_iter=1500, tol=1e-5)
cca.fit(V, A)
U, S = cca.transform(V, A)
corrs = np.array([np.corrcoef(U[:, i], S[:, i])[0, 1] for i in range(N_COMP)])

# Load audio codebook for the "which codes match predicted embedding"
ckpt = torch.load(A_CKPT, map_location="cpu", weights_only=False)
aa   = ckpt["args"]
a_model = AudioRVQVAE(1, aa["hidden"], aa["latent"],
                      aa["num_levels"], aa["codes_per_level"])
a_model.load_state_dict(ckpt["model"]); a_model.eval()
a_codebooks = [lvl.embedding.detach().cpu().numpy() for lvl in a_model.vq.levels]  # list of (K, D)

# Load raw audio for playback / FFT
wav_full, sr = sf.read(AUDIO)
if wav_full.ndim > 1: wav_full = wav_full.mean(1)
assert sr == SR
peak = np.abs(wav_full).max()
if peak > 0: wav_full = wav_full / peak * 0.95


# ── CCA-based V → A predictor ───────────────────────────────────────────
def predict_audio_from_visual(V_query_std, use_top_k=12):
    """V_query_std : (D,)  standardized visual embedding
       returns     : (D,)  predicted standardized audio embedding
    """
    # Project into visual canonical space
    U_q = V_query_std @ cca.x_weights_[:, :use_top_k]         # (K,)
    # Scale by canonical correlations (attenuated prediction)
    S_q = U_q * corrs[:use_top_k]                              # (K,)
    # Back-project into audio embedding space via pseudoinverse
    # y_pred ≈ S_q @ y_weights.T (using x-loadings-y-loadings link)
    y_weights = cca.y_weights_[:, :use_top_k]                  # (D, K)
    A_pred = y_weights @ S_q                                   # (D,)
    return A_pred


def predict_visual_from_audio(A_query_std, use_top_k=12):
    S_q = A_query_std @ cca.y_weights_[:, :use_top_k]
    U_q = S_q * corrs[:use_top_k]
    V_pred = cca.x_weights_[:, :use_top_k] @ U_q
    return V_pred


# ── Task 1: pick a "target explosion frame" ──────────────────────────────
# From v2 analysis we know frame at t=98.5s has V506 explosion pattern
# Also the audio-energy top spots were around t=10.5s, 70s, 86s, 100s
target_time = 100.0
target_idx  = int(round(target_time * FPS))
print(f"\nTarget frame index t={target_idx}  ({target_time:.1f}s)")

V_t = V[target_idx]                     # standardized visual embedding
A_actual = A[target_idx]                # standardized actual audio embedding

A_pred = predict_audio_from_visual(V_t, use_top_k=12)

# How similar is A_pred to A_actual?
cos_sim = float(np.dot(A_pred, A_actual) / (np.linalg.norm(A_pred)*np.linalg.norm(A_actual) + 1e-9))
print(f"  predicted-vs-actual cosine similarity : {cos_sim:+.3f}")

# Nearest actual audio embedding to A_pred (excluding target itself)
dists = np.linalg.norm(A - A_pred[None], axis=1)
dists[target_idx] = np.inf
top_neighbors = np.argsort(dists)[:6]
print(f"\n  Top-6 audio bins whose embedding is closest to predicted A_pred:")
for i in top_neighbors:
    print(f"    t = {i/FPS:6.1f}s   distance = {dists[i]:.3f}")

# Save target frame + retrieved audio clips
shutil.copy(FRAMES / f"frame_{target_idx+1:06d}.jpg",
            OUT / f"target_frame_t{target_time:.1f}s.jpg")

# Save the actual audio at target time
def save_audio_clip(bin_idx, name):
    start = int(bin_idx / FPS * SR)
    end   = min(start + SR, len(wav_full))
    sf.write(OUT / name, wav_full[start:end], SR)

save_audio_clip(target_idx, f"target_actual_audio_t{target_time:.1f}s.wav")
for rank, i in enumerate(top_neighbors):
    save_audio_clip(i, f"retrieved_audio_{rank+1}_t{i/FPS:.1f}s.wav")

# ── Task 2: which audio codebook entries best match A_pred ──────────────
# De-standardize A_pred first (multiply back by original std, add mean)
A_pred_raw = A_pred * A_emb.std(0) + A_emb.mean(0)

# Sum both RVQ level codebooks (since audio embeddings are sum of two levels)
# For each level, find which codes have embeddings closest to A_pred_raw
print("\n  ── Task 2: which audio codes best match the predicted A_pred? ──")
for L, cb in enumerate(a_codebooks):
    dists_cb = np.linalg.norm(cb - A_pred_raw[None] / 2, axis=1)  # /2 since RVQ sums two levels
    top_codes = np.argsort(dists_cb)[:8]
    print(f"\n    Audio codebook L{L+1} — top 8 codes matching predicted embedding:")
    for i, k in enumerate(top_codes):
        print(f"      #{i+1}  code A-L{L+1}-{k:03d}  distance = {dists_cb[k]:.4f}")

# ── Task 3: FFT of the retrieved audio  vs  a random baseline ────────────
print("\n  ── Task 3: FFT frequency content ──")
fft_target = np.abs(np.fft.rfft(wav_full[target_idx * int(SR/FPS):
                                          target_idx * int(SR/FPS) + SR]))
freqs = np.fft.rfftfreq(SR, 1/SR)

fft_retrieved = np.zeros_like(fft_target)
for i in top_neighbors:
    s = int(i / FPS * SR)
    clip = wav_full[s:s+SR]
    if len(clip) < SR:
        clip = np.pad(clip, (0, SR - len(clip)))
    fft_retrieved += np.abs(np.fft.rfft(clip))
fft_retrieved /= len(top_neighbors)

# Random baseline
rng = np.random.default_rng(0)
rand_idx = rng.choice(T, size=6, replace=False)
fft_random = np.zeros_like(fft_target)
for i in rand_idx:
    s = int(i / FPS * SR)
    clip = wav_full[s:s+SR]
    if len(clip) < SR:
        clip = np.pad(clip, (0, SR - len(clip)))
    fft_random += np.abs(np.fft.rfft(clip))
fft_random /= len(rand_idx)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
axes[0].semilogy(freqs, fft_target, color="black", lw=1.2, label="Target frame audio (actual)")
axes[0].semilogy(freqs, fft_retrieved, color="#F72585", lw=1.2,
                 label="Retrieved via CCA prediction (mean of top 6)")
axes[0].set_ylabel("|FFT|"); axes[0].legend()
axes[0].set_title(f"FFT spectra: target vs CCA-retrieved audio  (target t={target_time:.1f}s)")
axes[0].grid(alpha=0.3)

axes[1].semilogy(freqs, fft_retrieved, color="#F72585", lw=1.2,
                 label="CCA-retrieved (mean of top 6)")
axes[1].semilogy(freqs, fft_random, color="#666", lw=1.2, alpha=0.7,
                 label="Random-baseline audio (mean of 6)")
axes[1].set_xlabel("frequency (Hz)"); axes[1].set_ylabel("|FFT|"); axes[1].legend()
axes[1].set_title("FFT spectra: CCA-retrieved vs random baseline "
                  "(retrieved should look more like target)")
axes[1].grid(alpha=0.3)
axes[1].set_xlim(0, 8000)

plt.tight_layout()
plt.savefig(OUT / "fft_comparison.png", dpi=140)
plt.close()

# ── Task 4: reverse direction — given loud audio, predict visual ────────
print("\n  ── Task 4: REVERSE direction (audio → visual) ──")
# Pick a bin with high audio energy
audio_energy = np.zeros(T)
for t in range(T):
    s = int(t / FPS * SR)
    clip = wav_full[s:s+int(SR/FPS)]
    audio_energy[t] = np.sqrt(np.mean(clip**2)) if len(clip) else 0
loud_idx = int(np.argmax(audio_energy))
loud_time = loud_idx / FPS
print(f"    Loudest bin: t={loud_time:.1f}s  (RMS = {audio_energy[loud_idx]:.3f})")

V_pred_from_A = predict_visual_from_audio(A[loud_idx], use_top_k=12)
dists_V = np.linalg.norm(V - V_pred_from_A[None], axis=1)
dists_V[loud_idx] = np.inf
top_frames = np.argsort(dists_V)[:6]

# Save the retrieved frames
print(f"    Top-6 frames whose visual embedding matches predicted-from-loud-audio:")
for rank, i in enumerate(top_frames):
    print(f"      t = {i/FPS:6.1f}s   distance = {dists_V[i]:.3f}")
    shutil.copy(FRAMES / f"frame_{i+1:06d}.jpg",
                OUT / f"retrieved_frame_from_audio_{rank+1}_t{i/FPS:.1f}s.jpg")
save_audio_clip(loud_idx, f"query_loud_audio_t{loud_time:.1f}s.wav")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("SUMMARY — CCA cross-modal retrieval")
print("═" * 60)
print(f"  Target visual frame  : t = {target_time:.1f}s")
print(f"  Predicted A vs actual A cosine sim : {cos_sim:+.3f}")
print(f"  Retrieved similar audio at times    : "
      f"{[f'{i/FPS:.1f}s' for i in top_neighbors]}")
print(f"")
print(f"  Reverse query audio  : t = {loud_time:.1f}s (loudest bin)")
print(f"  Retrieved similar frames at times   : "
      f"{[f'{i/FPS:.1f}s' for i in top_frames]}")
print(f"\n  All artefacts in {OUT}/")
