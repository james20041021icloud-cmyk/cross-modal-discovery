"""
Interpret cross-modal edges by dumping evidence.

For each top-N cross-modal (visual code, audio code) pair from Glasso:
  - Find the frames where this visual code is most active
  - Show them side-by-side as a montage
  - Extract the audio snippets where this audio code is most active
  - Save as .wav

Also produces a summary contact-sheet PNG per pair.
"""
import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import soundfile as sf

TOK     = Path(os.environ.get("MMVQ_ROOT", ".")) / "tokens"
FRAMES  = Path(os.environ.get("MMVQ_ROOT", ".")) / "data/frames"
AUDIO   = Path(os.environ.get("MMVQ_ROOT", ".")) / "data/audio.wav"
EDGES   = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/glasso_v2/cross_modal_edges.csv"
OUT     = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/glasso_v2/pair_evidence"
OUT.mkdir(parents=True, exist_ok=True)

SR       = 16000
FPS      = 4.0


def make_pair_sheet(pair_idx, v_code, a_code, pc,
                    top_frame_indices, top_audio_start_times, tokens_dir):
    """Assemble a contact sheet: 6 top frames + audio waveform preview."""
    n_show = min(6, len(top_frame_indices))
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"Pair #{pair_idx+1}:  V{v_code:03d} ↔ A{a_code:03d}   "
                 f"partial-corr = {pc:+.3f}",
                 fontsize=13, fontweight="bold", y=0.98)

    # 1) Row of top frames for the visual code
    for i in range(n_show):
        ax = plt.subplot2grid((2, n_show), (0, i))
        idx = top_frame_indices[i]
        img_path = FRAMES / f"frame_{idx+1:06d}.jpg"
        if img_path.exists():
            ax.imshow(Image.open(img_path))
        ax.set_title(f"t={idx/FPS:.1f}s", fontsize=9)
        ax.axis("off")

    # 2) Waveform strip showing where the audio code is active
    wav, sr = sf.read(AUDIO)
    if wav.ndim > 1: wav = wav.mean(1)
    ax_wav = plt.subplot2grid((2, n_show), (1, 0), colspan=n_show)
    tvec = np.arange(len(wav)) / sr
    ax_wav.plot(tvec, wav, lw=0.4, color="#666")
    for st in top_audio_start_times:
        ax_wav.axvspan(st, st + 1.0, color="#E63946", alpha=0.35)
    ax_wav.set_xlim(0, tvec[-1])
    ax_wav.set_ylim(-1.05, 1.05)
    ax_wav.set_xlabel("time (s)")
    ax_wav.set_title(f"top windows where audio code A{a_code:03d} activates most (red bands)",
                     fontsize=10)

    plt.tight_layout()
    plt.savefig(tokens_dir / f"pair_{pair_idx+1:02d}_V{v_code:03d}_A{a_code:03d}.png",
                dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=8,
                    help="how many top pairs to explain")
    ap.add_argument("--n_examples", type=int, default=6,
                    help="how many top time-bin examples per code to dump")
    args = ap.parse_args()

    print(f"Loading edges from {EDGES}")
    edges = pd.read_csv(EDGES)
    edges["abs_pc"] = edges.partial_corr.abs()
    edges = edges.sort_values("abs_pc", ascending=False).reset_index(drop=True)
    print(f"  {len(edges)} total cross-modal edges; taking top {args.n_pairs}")

    v_hist = np.load(TOK/"visual_hist.npy")   # (T, K_v)
    a_hist = np.load(TOK/"audio_hist.npy")    # (T, K_a)
    T = v_hist.shape[0]

    wav, sr = sf.read(AUDIO)
    if wav.ndim > 1: wav = wav.mean(1)
    peak = np.abs(wav).max()
    if peak > 0: wav = wav / peak * 0.95

    summary_rows = []

    for k in range(min(args.n_pairs, len(edges))):
        row = edges.iloc[k]
        v_code = int(row.a[1:])   # strip 'V'
        a_code = int(row.b[1:])   # strip 'A'
        pc     = float(row.partial_corr)

        # rank time bins by activation strength
        v_act = v_hist[:, v_code]
        a_act = a_hist[:, a_code]
        v_top = np.argsort(-v_act)[:args.n_examples]
        a_top = np.argsort(-a_act)[:args.n_examples]

        # audio-window start times: window at index i covers [i/FPS, i/FPS + 1]
        a_top_starts = [i / FPS for i in a_top]

        # save audio snippets
        pair_dir = OUT / f"pair_{k+1:02d}_V{v_code:03d}_A{a_code:03d}"
        pair_dir.mkdir(exist_ok=True)
        for j, st in enumerate(a_top_starts):
            s0 = int(st * SR); s1 = s0 + SR
            if s1 <= len(wav):
                sf.write(pair_dir / f"audio_snippet_{j+1}.wav", wav[s0:s1], SR)

        # copy top visual frames
        for j, idx in enumerate(v_top):
            src = FRAMES / f"frame_{idx+1:06d}.jpg"
            if src.exists():
                shutil.copy(src, pair_dir / f"visual_top_{j+1}.jpg")

        # summary sheet
        make_pair_sheet(k, v_code, a_code, pc, v_top, a_top_starts, OUT)

        # temporal overlap
        v_bin = (v_act > 0).astype(int)
        a_bin = (a_act > 0).astype(int)
        joint = ((v_bin & a_bin).sum()) / max(1, (v_bin | a_bin).sum())
        v_freq = v_bin.mean()
        a_freq = a_bin.mean()

        summary_rows.append(dict(
            rank=k+1,
            visual_code=v_code, audio_code=a_code,
            partial_corr=round(pc, 4),
            v_freq_pct=round(100*v_freq, 1),
            a_freq_pct=round(100*a_freq, 1),
            jaccard_overlap=round(joint, 3),
        ))

        print(f"[#{k+1:>2}]  V{v_code:03d} ↔ A{a_code:03d}   "
              f"pc={pc:+.3f}   V fires in {100*v_freq:.1f}% of frames, "
              f"A in {100*a_freq:.1f}%, Jaccard {joint:.3f}")

    pd.DataFrame(summary_rows).to_csv(OUT/"pair_summary.csv", index=False)
    print(f"\n✅ Evidence dumped to {OUT}/")
    print("   Each pair_XX_VYYY_AZZZ/ contains:")
    print("     visual_top_*.jpg    top frames for the visual code")
    print("     audio_snippet_*.wav 1-sec audio clips for the audio code")
    print("   pair_XX_VYYY_AZZZ.png  contact-sheet visualisation")


if __name__ == "__main__":
    main()
