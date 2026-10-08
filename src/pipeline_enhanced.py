"""
Enhanced Stage 5 of the pipeline:
   - re-render composite figures with LOCAL-ONLY positions
     (only boxes/bands where the code actually fires in the chosen example)
   - use co-occurrence-based frame selection (V-code AND A-code both fire in
     the same time bin, ranked by joint activity)
   - show top-4 example frames per pair for cross-validation
   - compute grounding scores per pair (does the box land on something
     visually salient?  does the band land on high-energy audio?)

Reads the original pipeline outputs and writes new outputs to
  results/pipeline_final/pairs_enhanced/
without touching the original results.
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
OUT    = BASE / "pairs_enhanced"
OUT.mkdir(parents=True, exist_ok=True)

FPS       = 4.0
SR        = 16000
CLIP_LEN  = 16000
FRAME_H, FRAME_W = 128, 128
LAT_H, LAT_W     = 32, 32


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_all():
    log("Loading tokens + audio + previous results ...")
    with open(BASE / "results.json") as f:
        R = json.load(f)
    ok_pairs = [r for r in R["results"] if r.get("status") == "ok"]

    tokens = {
        "V1": np.load(TOK / "visual_tokens_L1.npy"),
        "V2": np.load(TOK / "visual_tokens_L2.npy"),
        "A1": np.load(TOK / "audio_tokens_L1.npy"),
        "A2": np.load(TOK / "audio_tokens_L2.npy"),
    }
    T = min(v.shape[0] for v in tokens.values())
    for k in tokens:
        tokens[k] = tokens[k][:T]

    wav, sr = sf.read(AUDIO)
    if wav.ndim > 1: wav = wav.mean(1)
    peak = np.abs(wav).max()
    if peak > 0: wav = wav / peak * 0.95

    log(f"  T = {T}, |audio| = {len(wav)/sr:.1f}s")
    return R, ok_pairs, tokens, wav


# ── Frame variance saliency (for grounding score) ───────────────────────
def frame_variance_map(img):
    """Approx saliency: local variance in 16x16 blocks of the frame."""
    img_gray = img.mean(axis=2) if img.ndim == 3 else img
    H, W = img_gray.shape
    Hs, Ws = H // 16, W // 16
    blocks = img_gray[:Hs*16, :Ws*16].reshape(Hs, 16, Ws, 16).swapaxes(1, 2)
    var = blocks.var(axis=(2, 3))
    return var                                                # (8, 8) variance per block


def spatial_grounding(v_positions_32, frame_img):
    """How much of the code's spatial positions land on high-variance patches?
    v_positions_32: list of (r, c) coords in 32×32 latent grid → each maps to
                    16×16 pixel region (r*4..r*4+4 in 32-grid → r*16..r*16+16 pixels).
                    We evaluate in the 8×8 downsampled block grid (16px each).
    """
    if len(v_positions_32) == 0: return 0.0
    var_map = frame_variance_map(frame_img)                   # (8, 8)
    var_flat = var_map.flatten()
    threshold = np.percentile(var_flat, 60)                   # top 40 % variance
    hits, total = 0, 0
    for (r, c) in v_positions_32:
        r8, c8 = r // 4, c // 4
        if 0 <= r8 < 8 and 0 <= c8 < 8:
            total += 1
            if var_map[r8, c8] >= threshold:
                hits += 1
    return hits / max(1, total)


def temporal_grounding(a_positions_1000, wav_seg):
    """How much of the code's temporal positions land on high-energy audio bins?
    a_positions_1000: list of integer positions in [0, 1000) latent → each maps
                      to 16 audio samples.
    """
    if len(a_positions_1000) == 0: return 0.0
    n_bins = 50
    samples_per_bin = SR // n_bins                            # 320 samples per bin
    rms = np.array([np.sqrt((wav_seg[i*samples_per_bin:(i+1)*samples_per_bin] ** 2).mean())
                    for i in range(n_bins)])
    threshold = np.percentile(rms, 60)
    hits, total = 0, 0
    for tau in a_positions_1000:
        b = tau // 20                                         # 20 latent = 1 of 50 bins
        if 0 <= b < n_bins:
            total += 1
            if rms[b] >= threshold:
                hits += 1
    return hits / max(1, total)


# ── Enhanced composite renderer ─────────────────────────────────────────
def render_enhanced(pair, tokens, wav):
    v_key = "V1" if pair["v_level"] == 1 else "V2"
    a_key = "A1" if pair["a_level"] == 1 else "A2"
    v_tok, a_tok = tokens[v_key], tokens[a_key]
    v_code, a_code = pair["v_code"], pair["a_code"]

    v_bin = (v_tok == v_code).astype(np.uint8)                # (T, 32, 32)
    a_bin = (a_tok == a_code).astype(np.uint8)                # (T, 1000)

    v_activity = v_bin.sum(axis=(1, 2))                       # (T,) count per frame
    a_activity = a_bin.sum(axis=1)                            # (T,) count per window

    joint_mask = (v_activity > 0) & (a_activity > 0)
    joint_bins = np.where(joint_mask)[0]

    if len(joint_bins) == 0:
        return None

    joint_score = v_activity * a_activity
    order = joint_bins[np.argsort(-joint_score[joint_bins])]
    top_bins = order[:4]                                       # top 4 examples

    fig = plt.figure(figsize=(18, 12))
    outer = fig.add_gridspec(3, 4, height_ratios=[1.1, 0.8, 0.9],
                             hspace=0.35, wspace=0.20,
                             left=0.04, right=0.98, top=0.92, bottom=0.05)

    v_scores, a_scores = [], []

    for col, b in enumerate(top_bins):
        # ── Row 0: frame with LOCAL-ONLY red boxes
        ax_img = fig.add_subplot(outer[0, col])
        img_path = FRAMES / f"frame_{b+1:06d}.jpg"
        if img_path.exists():
            img = np.array(Image.open(img_path).convert("RGB"))
        else:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        ax_img.imshow(img)

        H, W = img.shape[:2]
        cell_h, cell_w = H / 32, W / 32
        v_pos = np.argwhere(v_bin[b])                          # (n, 2)  in 32×32
        for (r, c) in v_pos:
            rect = Rectangle((c * cell_w, r * cell_h),
                             cell_w, cell_h,
                             edgecolor="red", linewidth=1.4, facecolor="none")
            ax_img.add_patch(rect)
        v_scores.append(spatial_grounding(v_pos, img))
        ax_img.set_title(f"frame #{b+1}   t={b/FPS:.1f} s\n"
                         f"{len(v_pos)} positions   "
                         f"grounding = {v_scores[-1]:.2f}",
                         fontsize=9)
        ax_img.axis("off")

        # ── Row 1: audio waveform with LOCAL-ONLY red bands
        ax_wav = fig.add_subplot(outer[1, col])
        s0_frame = int(b / FPS * SR)
        wav_seg = wav[s0_frame:s0_frame + CLIP_LEN]
        if len(wav_seg) < CLIP_LEN:
            wav_seg = np.pad(wav_seg, (0, CLIP_LEN - len(wav_seg)))
        t_axis = np.arange(len(wav_seg)) / SR
        ax_wav.plot(t_axis, wav_seg, lw=0.4, color="#333")

        a_pos = np.where(a_bin[b])[0]                          # positions in [0, 1000)
        for tau in a_pos:
            s_lo, s_hi = tau * 16, tau * 16 + 16
            ax_wav.axvspan(s_lo / SR, s_hi / SR, color="red", alpha=0.28)
        a_scores.append(temporal_grounding(a_pos, wav_seg))
        ax_wav.set_xlim(0, 1)
        ax_wav.set_ylim(-1.05, 1.05)
        ax_wav.set_title(f"audio window   {len(a_pos)} positions   "
                         f"grounding = {a_scores[-1]:.2f}",
                         fontsize=9)
        ax_wav.grid(alpha=0.2)
        if col == 0:
            ax_wav.set_ylabel("amp"); ax_wav.set_xlabel("s")

        # ── Row 2: FFT of the highlighted bands
        ax_fft = fig.add_subplot(outer[2, col])
        for tau in a_pos:
            seg = wav_seg[tau*16 : tau*16+16]
            if len(seg) < 8: continue
            F = np.abs(np.fft.rfft(seg))
            f_axis = np.fft.rfftfreq(len(seg), 1/SR)
            ax_fft.semilogy(f_axis, F, lw=0.6, alpha=0.5)
        ax_fft.set_xlim(0, 8000)
        ax_fft.grid(alpha=0.2)
        ax_fft.set_title("FFT of highlighted bands", fontsize=9)
        if col == 0:
            ax_fft.set_xlabel("Hz"); ax_fft.set_ylabel("|FFT|")

    mean_v = float(np.mean(v_scores))
    mean_a = float(np.mean(a_scores))

    fig.suptitle(
        f"Pair #{pair['pair_idx']+1}  ·  V-L{pair['v_level']}-{pair['v_code']:03d}  ↔  "
        f"A-L{pair['a_level']}-{pair['a_code']:03d}    ·    CCA r = {pair['r_canonical']:+.3f}    "
        f"·   spatial grounding {mean_v:.2f}   ·   temporal grounding {mean_a:.2f}",
        fontsize=13, fontweight="bold")

    out_path = OUT / f"pair_{pair['pair_idx']+1:02d}_enhanced.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "pair_idx": pair["pair_idx"],
        "n_joint_bins": int(len(joint_bins)),
        "top_bins": [int(b) for b in top_bins],
        "spatial_grounding": mean_v,
        "spatial_scores_per_frame": v_scores,
        "temporal_grounding": mean_a,
        "temporal_scores_per_frame": a_scores,
        "grounding_score": (mean_v + mean_a) / 2,             # combined
        "composite_png": str(out_path.relative_to(BASE)),
    }


# ── Enhanced index page ─────────────────────────────────────────────────
def build_enhanced_index(enhanced_results):
    n = len(enhanced_results)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 9, n_rows * 3.8 + 1.5))
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[0.4] + [1] * n_rows,
                          hspace=0.35, wspace=0.10,
                          left=0.03, right=0.98, top=0.97, bottom=0.02)

    ax_h = fig.add_subplot(gs[0, :]); ax_h.axis("off")
    ax_h.text(0.5, 0.75, "Enhanced Cross-Modal Fragment Localization",
              ha="center", va="center", fontsize=20, fontweight="bold",
              color="#065A82", transform=ax_h.transAxes)
    ax_h.text(0.5, 0.25,
              f"{n} pairs   ·   local-only boxes/bands   ·   4 co-occurrence frames per pair   ·   "
              f"grounding scores in headings",
              ha="center", va="center", fontsize=11, fontstyle="italic",
              color="#666", transform=ax_h.transAxes)

    ordered = sorted(enhanced_results, key=lambda r: -r["grounding_score"])
    for i, r in enumerate(ordered):
        row = i // n_cols + 1
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        p = BASE / r["composite_png"]
        if p.exists():
            ax.imshow(np.array(Image.open(p)))
        ax.axis("off")
        ax.set_title(
            f"pair #{r['pair_idx']+1}   "
            f"grounding {r['grounding_score']:.2f}   "
            f"(V {r['spatial_grounding']:.2f}, A {r['temporal_grounding']:.2f})   "
            f"joint bins = {r['n_joint_bins']}",
            fontsize=10)

    for i in range(len(ordered), n_rows * n_cols):
        row = i // n_cols + 1
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col]); ax.axis("off")

    fig.savefig(BASE / "index_enhanced.png", dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main():
    R, ok_pairs, tokens, wav = load_all()
    log(f"Processing {len(ok_pairs)} previously successful pairs with enhancements ...")

    enhanced = []
    for p in ok_pairs:
        r = render_enhanced(p, tokens, wav)
        if r is None:
            log(f"  pair #{p['pair_idx']+1}: no co-occurrence bins, skipped")
            continue
        enhanced.append(r)
        log(f"  pair #{p['pair_idx']+1:02d}: {r['n_joint_bins']:>3} joint bins, "
            f"grounding = {r['grounding_score']:.2f} "
            f"(V {r['spatial_grounding']:.2f}, A {r['temporal_grounding']:.2f})")

    build_enhanced_index(enhanced)

    # Merge into results.json in an enhanced_results field
    for e in enhanced:
        # find matching original result and attach
        for orig in R["results"]:
            if orig["pair_idx"] == e["pair_idx"]:
                orig["enhanced"] = e
                break
    with open(BASE / "results.json", "w") as f:
        json.dump(R, f, indent=2, default=str)

    # CSV of grounding scores
    df = pd.DataFrame([
        {"pair": r["pair_idx"] + 1,
         "v_grounding": r["spatial_grounding"],
         "a_grounding": r["temporal_grounding"],
         "grounding":   r["grounding_score"],
         "joint_bins":  r["n_joint_bins"],
         "png":         r["composite_png"]}
        for r in enhanced
    ]).sort_values("grounding", ascending=False)
    df.to_csv(BASE / "enhanced_summary.csv", index=False)

    log("")
    log("── Grounding summary ─────────────────────────────────────")
    log(df.to_string(index=False))
    log("")
    log(f"Mean spatial grounding: {df['v_grounding'].mean():.2f}")
    log(f"Mean temporal grounding: {df['a_grounding'].mean():.2f}")
    log(f"Mean combined grounding: {df['grounding'].mean():.2f}")
    log(f"\n✅ Enhanced outputs in {OUT}/")
    log(f"   index_enhanced.png, enhanced_summary.csv")


if __name__ == "__main__":
    main()
