"""
Glasso-based accuracy improvement (v2 of Stage 5).

For each pair that already has Glasso edges from Stage 2:

  1  UPSAMPLE Glasso-identified positions back to native latent resolution:
       spatial index (8×8 grid, 64 total) → 4×4 patch on 32×32 latent →
                                            16×16 pixel patch on 128×128 frame
       temporal index (50 bins) → 20 latent positions each →
from time import strftime
                                  320 audio samples (20 ms) each

  2  For each candidate co-occurrence frame, VALIDATE each Glasso edge:
       does the visual code actually fire in the specific 4×4 region?
       does the audio code actually fire in the specific 20-latent slice?

  3  Score each pair by:
       validation_rate = fraction of Glasso edges that hold in this frame
       show frames where validation_rate is highest → these are the
       frames the Glasso edges truly describe

  4  Draw red boxes ONLY for VALIDATED Glasso edges in the shown frames.
     This gives the "Glasso says X, and in this frame we can see X actually
     happening" evidence the professor asked for.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import soundfile as sf
from PIL import Image

warnings.filterwarnings("ignore")

ROOT   = Path(os.environ.get("MMVQ_ROOT", "."))
TOK    = ROOT / "tokens_rvq"
FRAMES = ROOT / "data/frames"
AUDIO  = ROOT / "data/audio.wav"
BASE   = ROOT / "results/pipeline_final"
OUT    = BASE / "pairs_glasso_v2"
OUT.mkdir(parents=True, exist_ok=True)

FPS       = 4.0
SR        = 16000
CLIP_LEN  = 16000

# Downsample factors used in Stage 2
SPATIAL_DS = 4          # 32×32 → 8×8, so each spatial idx covers 4×4 latent = 16×16 px
TEMPORAL_DS = 20        # 1000 → 50, so each temporal idx covers 20 latent = 320 samples

FRAME_H, FRAME_W = 128, 128


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def spatial_idx_to_latent_patch(p_8x8):
    """spatial idx 0..63 → (row_start, col_start, size) in 32×32 latent grid"""
    r_8, c_8 = divmod(p_8x8, 8)
    return r_8 * SPATIAL_DS, c_8 * SPATIAL_DS, SPATIAL_DS


def spatial_idx_to_pixel_box(p_8x8, H=FRAME_H, W=FRAME_W):
    """spatial idx 0..63 → (row_px, col_px, h_px, w_px) on H×W image"""
    r_8, c_8 = divmod(p_8x8, 8)
    r_px = r_8 * (H // 8)
    c_px = c_8 * (W // 8)
    return r_px, c_px, H // 8, W // 8


def temporal_idx_to_sample_range(t_50, samples_per_bin=None):
    """temporal idx 0..49 → (sample_start, sample_end) in 16000 samples"""
    spb = samples_per_bin or (CLIP_LEN // 50)
    return t_50 * spb, (t_50 + 1) * spb


def check_v_code_fires_in_patch(v_bin_frame, p_8x8):
    """
    v_bin_frame: (32, 32) binary map for one frame
    p_8x8:       Glasso spatial index in 8×8 grid
    Returns True if v-code fires anywhere in the corresponding 4×4 latent patch.
    """
    r0, c0, size = spatial_idx_to_latent_patch(p_8x8)
    return v_bin_frame[r0:r0+size, c0:c0+size].sum() > 0


def check_a_code_fires_in_range(a_bin_window, t_50):
    """
    a_bin_window: (1000,) binary sequence for one window
    t_50:         Glasso temporal index in [0, 50)
    Returns True if a-code fires anywhere in the 20-latent slice.
    """
    lo = t_50 * TEMPORAL_DS
    hi = lo + TEMPORAL_DS
    return a_bin_window[lo:hi].sum() > 0


# ── Main per-pair enhancement ───────────────────────────────────────────
def process_pair(pair, tokens, wav):
    """Validate Glasso edges per frame, render composite with only VALIDATED
    edges highlighted."""
    v_key = "V1" if pair["v_level"] == 1 else "V2"
    a_key = "A1" if pair["a_level"] == 1 else "A2"
    v_tok, a_tok = tokens[v_key], tokens[a_key]
    v_code, a_code = pair["v_code"], pair["a_code"]

    v_bin = (v_tok == v_code).astype(np.uint8)         # (T, 32, 32)
    a_bin = (a_tok == a_code).astype(np.uint8)         # (T, 1000)

    glasso_edges = pair.get("top_edges", [])
    if not glasso_edges:
        return None

    T = v_bin.shape[0]

    # ── Score each candidate frame by how many Glasso edges it validates
    v_fires = v_bin.sum(axis=(1, 2)) > 0
    a_fires = a_bin.sum(axis=1) > 0
    joint_mask = v_fires & a_fires
    joint_bins = np.where(joint_mask)[0]
    if len(joint_bins) == 0:
        return None

    validation_score = np.zeros(len(joint_bins))
    per_bin_validated_edges = {}
    for i, b in enumerate(joint_bins):
        n_valid = 0
        valid_edges = []
        for e in glasso_edges:
            sp = e["spatial_pos"]
            tp = e["temporal_pos"]
            v_ok = check_v_code_fires_in_patch(v_bin[b], sp)
            a_ok = check_a_code_fires_in_range(a_bin[b], tp)
            if v_ok and a_ok:
                n_valid += 1
                valid_edges.append(e)
        validation_score[i] = n_valid / len(glasso_edges)
        per_bin_validated_edges[b] = valid_edges

    # Pick top 4 bins by validation score
    order = np.argsort(-validation_score)
    top_bin_indices = joint_bins[order[:4]]
    top_scores = validation_score[order[:4]]

    # Mean validation rate across top 4 — the "grounding" of Glasso's edges
    grounding = float(np.mean(top_scores))

    fig = plt.figure(figsize=(20, 13))
    outer = fig.add_gridspec(3, 4, height_ratios=[1.2, 0.8, 0.9],
                             hspace=0.35, wspace=0.20,
                             left=0.03, right=0.99, top=0.90, bottom=0.05)

    per_frame_v_valid = []
    per_frame_a_valid = []

    for col, (b, sc) in enumerate(zip(top_bin_indices, top_scores)):
        valid_edges = per_bin_validated_edges[b]

        # ── Row 0: frame with Glasso-validated boxes (green) + code-only boxes (yellow)
        ax_img = fig.add_subplot(outer[0, col])
        img_path = FRAMES / f"frame_{b+1:06d}.jpg"
        img = np.array(Image.open(img_path).convert("RGB")) if img_path.exists() \
              else np.zeros((128, 128, 3), dtype=np.uint8)
        ax_img.imshow(img)
        # tokens were extracted with Resize(128) + CenterCrop(128), so map the
        # 128×128 model view back onto the full (wider) frame
        ih, iw = img.shape[:2]
        s = min(ih, iw) / 128
        ox, oy = (iw - min(ih, iw)) / 2, (ih - min(ih, iw)) / 2

        # 1) yellow (thin): everywhere V-code fires
        yellow_boxes = 0
        for (r, c) in np.argwhere(v_bin[b]):
            # thin yellow box per 4×4 latent → 16×16 px
            rect = Rectangle((ox + c * 4 * s, oy + r * 4 * s), 4 * s, 4 * s,
                             edgecolor="#FFC300", linewidth=0.4,
                             facecolor="none", alpha=0.7)
            ax_img.add_patch(rect)
            yellow_boxes += 1

        # 2) red (thick): Glasso edges that fire here (validated)
        red_boxes = 0
        drawn_positions = set()
        for e in valid_edges:
            sp = e["spatial_pos"]
            if sp in drawn_positions: continue
            drawn_positions.add(sp)
            r0, c0, h, w = spatial_idx_to_pixel_box(sp)
            rect = Rectangle((ox + c0 * s, oy + r0 * s), w * s, h * s, edgecolor="red",
                             linewidth=2.4, facecolor="none")
            ax_img.add_patch(rect)
            red_boxes += 1

        per_frame_v_valid.append(red_boxes / max(1, len(glasso_edges)))
        ax_img.set_title(
            f"frame #{b+1}   t={b/FPS:.1f}s\n"
            f"Glasso valid = {sc:.0%}   ({red_boxes} red boxes)\n"
            f"yellow = all code positions ({yellow_boxes})",
            fontsize=9)
        ax_img.axis("off")

        # ── Row 1: audio window with Glasso-validated bands (red)
        ax_wav = fig.add_subplot(outer[1, col])
        s0_frame = int(b / FPS * SR)
        wav_seg = wav[s0_frame:s0_frame + CLIP_LEN]
        if len(wav_seg) < CLIP_LEN:
            wav_seg = np.pad(wav_seg, (0, CLIP_LEN - len(wav_seg)))
        t_axis = np.arange(len(wav_seg)) / SR
        ax_wav.plot(t_axis, wav_seg, lw=0.4, color="#333")

        # yellow: all a-code positions (thin bands)
        yellow_bands = 0
        for tau in np.where(a_bin[b])[0]:
            ax_wav.axvspan(tau * 16 / SR, (tau * 16 + 16) / SR,
                           color="#FFC300", alpha=0.20)
            yellow_bands += 1

        # red: Glasso-validated bands
        red_bands = 0
        drawn_temporal = set()
        for e in valid_edges:
            tp = e["temporal_pos"]
            if tp in drawn_temporal: continue
            drawn_temporal.add(tp)
            lo, hi = temporal_idx_to_sample_range(tp)
            ax_wav.axvspan(lo / SR, hi / SR, color="red", alpha=0.45)
            red_bands += 1

        per_frame_a_valid.append(red_bands / max(1, len(glasso_edges)))
        ax_wav.set_xlim(0, 1)
        ax_wav.set_ylim(-1.05, 1.05)
        ax_wav.grid(alpha=0.2)
        ax_wav.set_title(
            f"waveform   red = {red_bands} Glasso-valid bands   "
            f"yellow = {yellow_bands} code activations",
            fontsize=9)
        if col == 0:
            ax_wav.set_xlabel("s within 1-sec window")
            ax_wav.set_ylabel("amp")

        # ── Row 2: FFT of the RED (Glasso-validated) audio bands
        ax_fft = fig.add_subplot(outer[2, col])
        for e in valid_edges:
            tp = e["temporal_pos"]
            lo, hi = temporal_idx_to_sample_range(tp)
            seg = wav_seg[lo:hi]
            if len(seg) < 16: continue
            F = np.abs(np.fft.rfft(seg))
            f_axis = np.fft.rfftfreq(len(seg), 1 / SR)
            ax_fft.semilogy(f_axis, F, lw=0.7, alpha=0.6)
        ax_fft.set_xlim(0, 8000)
        ax_fft.grid(alpha=0.2)
        ax_fft.set_title("FFT of Glasso-valid audio bands", fontsize=9)
        if col == 0:
            ax_fft.set_xlabel("Hz"); ax_fft.set_ylabel("|FFT|")

    mean_v = float(np.mean(per_frame_v_valid))
    mean_a = float(np.mean(per_frame_a_valid))

    fig.suptitle(
        f"Pair #{pair['pair_idx']+1}  ·  "
        f"V-L{pair['v_level']}-{pair['v_code']:03d}  ↔  "
        f"A-L{pair['a_level']}-{pair['a_code']:03d}   ·   "
        f"CCA r = {pair['r_canonical']:+.3f}   ·   Glasso α = {pair.get('alpha', 0):.2f}\n"
        f"Glasso edges: {len(glasso_edges)}    ·   "
        f"mean validation rate: {grounding:.0%}   "
        f"(V region hit-rate {mean_v:.0%}, A band hit-rate {mean_a:.0%})",
        fontsize=13, fontweight="bold")

    out_path = OUT / f"pair_{pair['pair_idx']+1:02d}_glasso_v2.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "pair_idx": pair["pair_idx"],
        "grounding": grounding,
        "n_glasso_edges": len(glasso_edges),
        "top_bins": [int(b) for b in top_bin_indices],
        "validation_per_bin": [float(s) for s in top_scores],
        "png": str(out_path.relative_to(BASE)),
    }


def build_index(results):
    n = len(results); n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 10, n_rows * 4.2 + 1.5))
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[0.4] + [1] * n_rows,
                          hspace=0.35, wspace=0.10,
                          left=0.03, right=0.98, top=0.97, bottom=0.02)

    ax_h = fig.add_subplot(gs[0, :]); ax_h.axis("off")
    ax_h.text(0.5, 0.75,
              "Glasso-Validated Cross-Modal Fragment Localization",
              ha="center", va="center", fontsize=20, fontweight="bold",
              color="#065A82", transform=ax_h.transAxes)
    ax_h.text(0.5, 0.20,
              "red boxes/bands = Glasso edges CONFIRMED to fire in this frame   ·   "
              "yellow = all positions where the code fires (context)",
              ha="center", va="center", fontsize=11, fontstyle="italic",
              color="#666", transform=ax_h.transAxes)

    ordered = sorted(results, key=lambda r: -r["grounding"])
    for i, r in enumerate(ordered):
        row = i // n_cols + 1
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        p = BASE / r["png"]
        if p.exists():
            ax.imshow(np.array(Image.open(p)))
        ax.axis("off")
        ax.set_title(
            f"pair #{r['pair_idx']+1}   "
            f"Glasso validation rate = {r['grounding']:.0%}   "
            f"({r['n_glasso_edges']} edges)",
            fontsize=11)

    for i in range(len(ordered), n_rows * n_cols):
        row = i // n_cols + 1
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col]); ax.axis("off")

    fig.savefig(BASE / "index_glasso_v2.png", dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main():
    log("Loading tokens + audio + previous Glasso results ...")
    tokens = {
        "V1": np.load(TOK / "visual_tokens_L1.npy"),
        "V2": np.load(TOK / "visual_tokens_L2.npy"),
        "A1": np.load(TOK / "audio_tokens_L1.npy"),
        "A2": np.load(TOK / "audio_tokens_L2.npy"),
    }
    T = min(v.shape[0] for v in tokens.values())
    for k in tokens: tokens[k] = tokens[k][:T]

    with open(BASE / "results.json") as f:
        R = json.load(f)
    ok_pairs = [r for r in R["results"] if r.get("status") == "ok"]
    log(f"  T = {T},  {len(ok_pairs)} pairs with Glasso edges")

    wav, sr = sf.read(AUDIO)
    if wav.ndim > 1: wav = wav.mean(1)
    peak = np.abs(wav).max()
    if peak > 0: wav = wav / peak * 0.95

    log(f"\n── Per-pair Glasso-validated rendering ─────────────────────")
    results = []
    for p in ok_pairs:
        r = process_pair(p, tokens, wav)
        if r is None:
            log(f"  pair #{p['pair_idx']+1}: no co-occurrence frames, skipped")
            continue
        results.append(r)
        log(f"  pair #{p['pair_idx']+1:02d}: "
            f"validation rate = {r['grounding']:.0%}   "
            f"({r['n_glasso_edges']} Glasso edges)")

    build_index(results)

    df = pd.DataFrame(results).sort_values("grounding", ascending=False)
    df.to_csv(BASE / "glasso_v2_summary.csv", index=False)

    log("")
    log("── Glasso validation-rate summary ─────────────────────")
    log(df.to_string(index=False))
    log(f"\n Mean validation rate: {df['grounding'].mean():.0%}")
    log(f" ≥ 50% pairs: {(df['grounding'] >= 0.5).sum()} / {len(df)}")
    log(f" ≥ 75% pairs: {(df['grounding'] >= 0.75).sum()} / {len(df)}")
    log(f"\n✅ Glasso-validated composites in {OUT}/")


if __name__ == "__main__":
    main()
