"""
v3 improvement: filter Glasso edges to top-3 by |partial_corr| per pair.

Rationale: from v2 results, pairs with 1-2 edges validate at 75%,
pairs with 5-10 edges validate at 20-25%. Weaker Glasso edges are
often statistical noise. Limiting to top-3 (by |partial_corr|)
should raise mean validation rate significantly.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import json, warnings
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
OUT    = BASE / "pairs_glasso_v3"
OUT.mkdir(parents=True, exist_ok=True)

FPS, SR, CLIP_LEN = 4.0, 16000, 16000
SPATIAL_DS, TEMPORAL_DS = 4, 20
FRAME_H, FRAME_W = 128, 128
MAX_EDGES_PER_PAIR = 3          # ← THE KEY v3 CHANGE

def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def spatial_idx_to_pixel_box(p_8x8):
    r_8, c_8 = divmod(p_8x8, 8)
    return r_8 * (FRAME_H // 8), c_8 * (FRAME_W // 8), FRAME_H // 8, FRAME_W // 8


def check_v_fires(v_bin_frame, p_8x8):
    r_8, c_8 = divmod(p_8x8, 8)
    r0, c0 = r_8 * SPATIAL_DS, c_8 * SPATIAL_DS
    return v_bin_frame[r0:r0+SPATIAL_DS, c0:c0+SPATIAL_DS].sum() > 0


def check_a_fires(a_bin_win, t_50):
    lo = t_50 * TEMPORAL_DS
    return a_bin_win[lo:lo+TEMPORAL_DS].sum() > 0


def process_pair(pair, tokens, wav):
    v_key = "V1" if pair["v_level"] == 1 else "V2"
    a_key = "A1" if pair["a_level"] == 1 else "A2"
    v_bin = (tokens[v_key] == pair["v_code"]).astype(np.uint8)
    a_bin = (tokens[a_key] == pair["a_code"]).astype(np.uint8)

    all_edges = pair.get("top_edges", [])
    if not all_edges:
        return None

    # Keep only top-K by |partial_corr|
    edges = sorted(all_edges,
                   key=lambda e: -abs(e.get("partial_corr", 0)))[:MAX_EDGES_PER_PAIR]

    v_fires = v_bin.sum(axis=(1, 2)) > 0
    a_fires = a_bin.sum(axis=1) > 0
    joint_bins = np.where(v_fires & a_fires)[0]
    if len(joint_bins) == 0:
        return None

    val = np.zeros(len(joint_bins))
    valid_per_bin = {}
    for i, b in enumerate(joint_bins):
        vs = [e for e in edges
              if check_v_fires(v_bin[b], e["spatial_pos"])
              and check_a_fires(a_bin[b], e["temporal_pos"])]
        val[i] = len(vs) / len(edges)
        valid_per_bin[b] = vs

    order = np.argsort(-val)
    top_bins = joint_bins[order[:4]]
    top_scores = val[order[:4]]

    grounding = float(np.mean(top_scores))

    fig = plt.figure(figsize=(20, 13))
    outer = fig.add_gridspec(3, 4, height_ratios=[1.2, 0.8, 0.9],
                             hspace=0.35, wspace=0.20,
                             left=0.03, right=0.99, top=0.90, bottom=0.05)

    per_frame_v = []
    per_frame_a = []

    for col, (b, sc) in enumerate(zip(top_bins, top_scores)):
        v_ok = valid_per_bin[b]

        # Row 0: frame + boxes
        ax_img = fig.add_subplot(outer[0, col])
        img_p = FRAMES / f"frame_{b+1:06d}.jpg"
        img = np.array(Image.open(img_p).convert("RGB")) if img_p.exists() else np.zeros((128,128,3),dtype=np.uint8)
        ax_img.imshow(img)

        yellow = 0
        for (r, c) in np.argwhere(v_bin[b]):
            rect = Rectangle((c*4, r*4), 4, 4, ec="#FFC300", lw=0.4, fc="none", alpha=0.65)
            ax_img.add_patch(rect); yellow += 1

        red = 0
        drawn = set()
        for e in v_ok:
            sp = e["spatial_pos"]
            if sp in drawn: continue
            drawn.add(sp)
            r0, c0, h, w = spatial_idx_to_pixel_box(sp)
            rect = Rectangle((c0, r0), w, h, ec="red", lw=2.5, fc="none")
            ax_img.add_patch(rect); red += 1

        per_frame_v.append(red / max(1, len(edges)))
        ax_img.set_title(
            f"frame #{b+1}  t={b/FPS:.1f}s\n"
            f"Glasso valid {sc:.0%}  ({red} red boxes)\n"
            f"yellow: all {yellow} code positions", fontsize=9)
        ax_img.axis("off")

        # Row 1: waveform
        ax_wav = fig.add_subplot(outer[1, col])
        s0 = int(b/FPS*SR)
        seg = wav[s0:s0+CLIP_LEN]
        if len(seg) < CLIP_LEN: seg = np.pad(seg, (0, CLIP_LEN-len(seg)))
        t_ax = np.arange(len(seg))/SR
        ax_wav.plot(t_ax, seg, lw=0.4, color="#333")

        ybands = 0
        for tau in np.where(a_bin[b])[0]:
            ax_wav.axvspan(tau*16/SR, (tau*16+16)/SR, color="#FFC300", alpha=0.18)
            ybands += 1

        rbands = 0
        drawn_t = set()
        for e in v_ok:
            tp = e["temporal_pos"]
            if tp in drawn_t: continue
            drawn_t.add(tp)
            lo, hi = tp*(CLIP_LEN//50), (tp+1)*(CLIP_LEN//50)
            ax_wav.axvspan(lo/SR, hi/SR, color="red", alpha=0.5)
            rbands += 1
        per_frame_a.append(rbands / max(1, len(edges)))
        ax_wav.set_xlim(0, 1); ax_wav.set_ylim(-1.05, 1.05); ax_wav.grid(alpha=0.2)
        ax_wav.set_title(f"waveform  red={rbands}  yellow={ybands}", fontsize=9)
        if col == 0:
            ax_wav.set_xlabel("s"); ax_wav.set_ylabel("amp")

        # Row 2: FFT
        ax_fft = fig.add_subplot(outer[2, col])
        for e in v_ok:
            tp = e["temporal_pos"]
            lo = tp*(CLIP_LEN//50); hi = lo + (CLIP_LEN//50)
            audio_seg = seg[lo:hi]
            if len(audio_seg) < 16: continue
            F = np.abs(np.fft.rfft(audio_seg))
            f_ax = np.fft.rfftfreq(len(audio_seg), 1/SR)
            ax_fft.semilogy(f_ax, F, lw=0.7, alpha=0.6)
        ax_fft.set_xlim(0, 8000); ax_fft.grid(alpha=0.2)
        ax_fft.set_title("FFT of Glasso-valid bands", fontsize=9)
        if col == 0:
            ax_fft.set_xlabel("Hz"); ax_fft.set_ylabel("|FFT|")

    fig.suptitle(
        f"v3 top-{MAX_EDGES_PER_PAIR} · Pair #{pair['pair_idx']+1}  ·  "
        f"V-L{pair['v_level']}-{pair['v_code']:03d}  ↔  A-L{pair['a_level']}-{pair['a_code']:03d}   ·   "
        f"CCA r = {pair['r_canonical']:+.3f}\n"
        f"filtered from {len(all_edges)} to {len(edges)} edges  ·  "
        f"mean validation {grounding:.0%}   "
        f"(V hit {np.mean(per_frame_v):.0%}, A hit {np.mean(per_frame_a):.0%})",
        fontsize=13, fontweight="bold")

    outp = OUT / f"pair_{pair['pair_idx']+1:02d}_glasso_v3.png"
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "pair_idx": pair["pair_idx"],
        "grounding": grounding,
        "n_edges_original": len(all_edges),
        "n_edges_kept": len(edges),
        "top_bins": [int(b) for b in top_bins],
        "png": str(outp.relative_to(BASE)),
    }


def main():
    log("Loading tokens + previous Glasso pipeline output ...")
    tokens = {"V1": np.load(TOK/"visual_tokens_L1.npy"),
              "V2": np.load(TOK/"visual_tokens_L2.npy"),
              "A1": np.load(TOK/"audio_tokens_L1.npy"),
              "A2": np.load(TOK/"audio_tokens_L2.npy")}
    T = min(v.shape[0] for v in tokens.values())
    for k in tokens: tokens[k] = tokens[k][:T]

    with open(BASE/"results.json") as f: R = json.load(f)
    ok = [r for r in R["results"] if r.get("status") == "ok"]
    log(f"  {len(ok)} pairs to process")

    wav, sr = sf.read(AUDIO)
    if wav.ndim > 1: wav = wav.mean(1)
    p = np.abs(wav).max()
    if p > 0: wav = wav/p * 0.95

    log(f"\n── v3 per-pair (top-{MAX_EDGES_PER_PAIR} edges only) ──")
    results = []
    for p_pair in ok:
        r = process_pair(p_pair, tokens, wav)
        if r is None:
            log(f"  pair #{p_pair['pair_idx']+1}: skipped"); continue
        results.append(r)
        log(f"  pair #{p_pair['pair_idx']+1:02d}: "
            f"{r['n_edges_kept']}/{r['n_edges_original']} edges kept, "
            f"validation {r['grounding']:.0%}")

    df = pd.DataFrame(results).sort_values("grounding", ascending=False)
    df.to_csv(BASE/"glasso_v3_summary.csv", index=False)

    log("")
    log("── v3 summary ──")
    log(df.to_string(index=False))
    log(f"\nMean validation v3: {df['grounding'].mean():.0%}")
    log(f"≥ 75%: {(df['grounding'] >= 0.75).sum()}/{len(df)}")
    log(f"≥ 50%: {(df['grounding'] >= 0.5).sum()}/{len(df)}")
    log(f"= 100%: {(df['grounding'] >= 0.99).sum()}/{len(df)}")

    # Build index page
    n = len(df); n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols*10, n_rows*4.2 + 1.5))
    gs = fig.add_gridspec(n_rows+1, n_cols, height_ratios=[0.4]+[1]*n_rows,
                          hspace=0.35, wspace=0.10, left=0.03, right=0.98,
                          top=0.97, bottom=0.02)

    ax_h = fig.add_subplot(gs[0, :]); ax_h.axis("off")
    ax_h.text(0.5, 0.75, f"v3: Top-{MAX_EDGES_PER_PAIR} Glasso Edges Only",
              ha="center", va="center", fontsize=20, fontweight="bold",
              color="#065A82", transform=ax_h.transAxes)
    ax_h.text(0.5, 0.20,
              f"filtered to strongest {MAX_EDGES_PER_PAIR} edges per pair · "
              f"red = validated, yellow = context",
              ha="center", va="center", fontsize=11, fontstyle="italic",
              color="#666", transform=ax_h.transAxes)

    for i, r in df.reset_index(drop=True).iterrows():
        row, col = i // n_cols + 1, i % n_cols
        ax = fig.add_subplot(gs[row, col])
        p_img = BASE / r["png"]
        if p_img.exists(): ax.imshow(np.array(Image.open(p_img)))
        ax.axis("off")
        ax.set_title(f"pair #{int(r['pair_idx'])+1}  "
                     f"validation = {r['grounding']:.0%}  "
                     f"({int(r['n_edges_kept'])}/{int(r['n_edges_original'])} edges)",
                     fontsize=11)

    for i in range(len(df), n_rows*n_cols):
        row, col = i // n_cols + 1, i % n_cols
        ax = fig.add_subplot(gs[row, col]); ax.axis("off")

    fig.savefig(BASE/"index_glasso_v3.png", dpi=100, bbox_inches="tight",
                facecolor="white")
    log(f"\n✅ v3 outputs in {OUT}/")


if __name__ == "__main__":
    main()
