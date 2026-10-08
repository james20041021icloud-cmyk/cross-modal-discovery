"""
v4: Sparsity-filtered Glasso pipeline.

Analysis of v2/v3: low-scoring pairs have codes that fire in MANY positions
per frame (background patterns), so the localization signal is diluted.

v4 adds a code-sparsity filter:
  - visual: keep only V-codes where the mean fraction of firing positions
            per frame is < 25% (localized, not background)
  - audio:  keep only A-codes where the mean fraction of firing positions
            per window is < 25%
  - THEN pick per-canonical-component matched pairs from the FILTERED set

Also: renders using top-3 edges (from v3) + best-frame selection based on
validated edges AND low yellow-box count (clean localization).
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
from sklearn.cross_decomposition import CCA
from sklearn.covariance import GraphicalLasso

warnings.filterwarnings("ignore")

ROOT   = Path(os.environ.get("MMVQ_ROOT", "."))
EMB    = ROOT / "results/embedding_glasso"
TOK    = ROOT / "tokens_rvq"
FRAMES = ROOT / "data/frames"
AUDIO  = ROOT / "data/audio.wav"
BASE   = ROOT / "results/pipeline_final"
OUT    = BASE / "pairs_glasso_v4"
OUT.mkdir(parents=True, exist_ok=True)

FPS, SR, CLIP_LEN = 4.0, 16000, 16000
SPATIAL_DS, TEMPORAL_DS = 4, 20
FRAME_H, FRAME_W = 128, 128
N_PAIRS = 20
MAX_EDGES = 3

# NEW: sparsity thresholds
V_SPARSITY_MAX = 0.25   # code fires in < 25% of frame's 1024 positions on avg
A_SPARSITY_MAX = 0.25   # code fires in < 25% of window's 1000 positions on avg
V_MIN_ACTIVE   = 20     # code must fire in ≥ 20 frames total
A_MIN_ACTIVE   = 20


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


# ── Stage 1: CCA and matched-pair identification (with sparsity filter) ──
def fit_cca_and_identify_pairs(data):
    log("Fitting CCA (K=32) ...")
    V_emb = data["V_emb"]; A_emb = data["A_emb"]
    Vs = (V_emb - V_emb.mean(0)) / (V_emb.std(0) + 1e-9)
    As = (A_emb - A_emb.mean(0)) / (A_emb.std(0) + 1e-9)
    cca = CCA(n_components=32, max_iter=1500, tol=1e-5); cca.fit(Vs, As)
    U, S = cca.transform(Vs, As)

    T = U.shape[0]

    def per_bin_presence(tok, K):
        flat = tok.reshape(T, -1)
        P = np.zeros((T, K), dtype=np.float32)
        for k in range(K):
            P[:, k] = (flat == k).any(axis=1).astype(np.float32)
        return P

    def per_bin_density(tok, K):
        flat = tok.reshape(T, -1)
        L = flat.shape[1]
        D = np.zeros((T, K), dtype=np.float32)
        for k in range(K):
            D[:, k] = (flat == k).mean(axis=1)      # fraction of positions
        return D

    log("Computing per-code presence + density ...")
    V1_pres = per_bin_presence(data["V_tok_L1"], 256)
    V2_pres = per_bin_presence(data["V_tok_L2"], 256)
    A1_pres = per_bin_presence(data["A_tok_L1"], 128)
    A2_pres = per_bin_presence(data["A_tok_L2"], 128)
    V1_dens = per_bin_density (data["V_tok_L1"], 256)
    V2_dens = per_bin_density (data["V_tok_L2"], 256)
    A1_dens = per_bin_density (data["A_tok_L1"], 128)
    A2_dens = per_bin_density (data["A_tok_L2"], 128)

    def sparsity_ok(pres, dens, min_active, max_density):
        """Return boolean array over K codes indicating acceptance."""
        n_active = pres.sum(0)                       # frames where code fires
        mean_dens_active = dens.mean(0)              # mean fraction across all bins
        return (n_active >= min_active) & (mean_dens_active <= max_density)

    v1_ok = sparsity_ok(V1_pres, V1_dens, V_MIN_ACTIVE, V_SPARSITY_MAX)
    v2_ok = sparsity_ok(V2_pres, V2_dens, V_MIN_ACTIVE, V_SPARSITY_MAX)
    a1_ok = sparsity_ok(A1_pres, A1_dens, A_MIN_ACTIVE, A_SPARSITY_MAX)
    a2_ok = sparsity_ok(A2_pres, A2_dens, A_MIN_ACTIVE, A_SPARSITY_MAX)

    log(f"  V codes passing sparsity filter: L1={v1_ok.sum()}, L2={v2_ok.sum()}")
    log(f"  A codes passing sparsity filter: L1={a1_ok.sum()}, L2={a2_ok.sum()}")

    def top_code_filtered(pres_L1, pres_L2, ok_L1, ok_L2, canonical):
        best_corr, best_lvl, best_code = 0.0, 1, 0
        for lvl, (P, mask) in enumerate([(pres_L1, ok_L1),
                                          (pres_L2, ok_L2)], start=1):
            for k in range(P.shape[1]):
                if not mask[k] or P[:, k].std() == 0: continue
                c = np.corrcoef(P[:, k], canonical)[0, 1]
                if abs(c) > abs(best_corr):
                    best_corr, best_lvl, best_code = c, lvl, k
        return best_lvl, best_code, best_corr

    log("Identifying pairs on the filtered code set ...")
    pairs = []
    for cc in range(min(N_PAIRS, U.shape[1])):
        v_lvl, v_code, v_corr = top_code_filtered(V1_pres, V2_pres, v1_ok, v2_ok, U[:, cc])
        a_lvl, a_code, a_corr = top_code_filtered(A1_pres, A2_pres, a1_ok, a2_ok, S[:, cc])
        if v_corr == 0 or a_corr == 0:
            continue                                # no code passed the filter
        r_cc = float(np.corrcoef(U[:, cc], S[:, cc])[0, 1])
        pairs.append({
            "pair_idx":    len(pairs), "cc": cc,
            "v_level": v_lvl, "v_code": v_code, "v_corr": float(v_corr),
            "a_level": a_lvl, "a_code": a_code, "a_corr": float(a_corr),
            "r_canonical": r_cc,
        })
        log(f"  pair #{len(pairs):02d}  CC{cc+1}  r={r_cc:+.3f}  "
            f"V-L{v_lvl}-{v_code:03d} (r={v_corr:+.3f})  "
            f"A-L{a_lvl}-{a_code:03d} (r={a_corr:+.3f})")

    return pairs


# ── Stage 2: per-pair Glasso ────────────────────────────────────────────
def spatial_downsample(binary_T_H_W, factor):
    T, H, W = binary_T_H_W.shape
    H2, W2 = H // factor, W // factor
    return binary_T_H_W.reshape(T, H2, factor, W2, factor).mean(axis=(2, 4))


def temporal_downsample(binary_T_L, factor):
    T, L = binary_T_L.shape
    L2 = L // factor
    return binary_T_L[:, :L2*factor].reshape(T, L2, factor).mean(axis=2)


def fit_glasso_fallback(X, alphas=(0.10, 0.15, 0.20, 0.30, 0.40)):
    for a in alphas:
        try:
            m = GraphicalLasso(alpha=a, max_iter=200, tol=1e-3); m.fit(X)
            return m, a
        except Exception:
            continue
    return None, None


def stage2_edges(pair, tokens):
    v_tok = tokens["V1"] if pair["v_level"] == 1 else tokens["V2"]
    a_tok = tokens["A1"] if pair["a_level"] == 1 else tokens["A2"]
    v_bin = (v_tok == pair["v_code"]).astype(np.float32)
    a_bin = (a_tok == pair["a_code"]).astype(np.float32)

    v_ds = spatial_downsample(v_bin, SPATIAL_DS)
    v_flat = v_ds.reshape(v_ds.shape[0], -1)
    a_ds = temporal_downsample(a_bin, TEMPORAL_DS)
    X = np.concatenate([v_flat, a_ds], axis=1)

    keep = X.var(0) > 1e-8
    n_v_full = v_flat.shape[1]
    v_keep_idx = np.where(keep[:n_v_full])[0]
    a_keep_idx = np.where(keep[n_v_full:])[0]
    X = X[:, keep]
    if X.shape[1] < 8 or len(v_keep_idx) < 3 or len(a_keep_idx) < 3:
        return v_bin, a_bin, [], 0

    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    model, alpha = fit_glasso_fallback(X)
    if model is None:
        return v_bin, a_bin, [], 0

    prec = model.precision_
    D = np.sqrt(np.diag(prec))
    partial = -prec / np.outer(D, D)
    np.fill_diagonal(partial, 1.0)
    n_v_eff = len(v_keep_idx)
    cross = partial[:n_v_eff, n_v_eff:]

    edges = []
    for i in range(cross.shape[0]):
        for j in range(cross.shape[1]):
            pc = cross[i, j]
            if abs(pc) >= 0.05:
                edges.append({"spatial_pos":  int(v_keep_idx[i]),
                              "temporal_pos": int(a_keep_idx[j]),
                              "partial_corr": float(pc)})
    edges.sort(key=lambda e: -abs(e["partial_corr"]))
    return v_bin.astype(np.uint8), a_bin.astype(np.uint8), edges[:MAX_EDGES], alpha


def process_pair(pair, tokens, wav):
    v_bin, a_bin, edges, alpha = stage2_edges(pair, tokens)
    if not edges: return None
    T = v_bin.shape[0]

    v_fires = v_bin.sum(axis=(1, 2)) > 0
    a_fires = a_bin.sum(axis=1) > 0
    joint_bins = np.where(v_fires & a_fires)[0]
    if len(joint_bins) == 0: return None

    # Score bins: validated edges - 0.02 * yellow_count (prefer clean localization)
    val = np.zeros(len(joint_bins))
    v_pos_count = v_bin.sum(axis=(1, 2))
    for i, b in enumerate(joint_bins):
        vs = sum(1 for e in edges
                 if check_v_fires(v_bin[b], e["spatial_pos"])
                 and check_a_fires(a_bin[b], e["temporal_pos"]))
        val[i] = vs / len(edges) - 0.005 * v_pos_count[b] / 1024.0
    # Get true validation for reporting
    true_val = np.zeros(len(joint_bins))
    per_bin_valid = {}
    for i, b in enumerate(joint_bins):
        vs = [e for e in edges
              if check_v_fires(v_bin[b], e["spatial_pos"])
              and check_a_fires(a_bin[b], e["temporal_pos"])]
        true_val[i] = len(vs) / len(edges)
        per_bin_valid[b] = vs

    order = np.argsort(-val)                      # scoring incorporates clean-ness
    top_bins = joint_bins[order[:4]]
    grounding = float(np.mean(true_val[order[:4]]))

    fig = plt.figure(figsize=(20, 13))
    outer = fig.add_gridspec(3, 4, height_ratios=[1.2, 0.8, 0.9],
                             hspace=0.35, wspace=0.20,
                             left=0.03, right=0.99, top=0.90, bottom=0.05)
    v_hits_all = []; a_hits_all = []

    for col, b in enumerate(top_bins):
        v_ok = per_bin_valid[b]

        # frame + boxes
        ax_img = fig.add_subplot(outer[0, col])
        img_p = FRAMES / f"frame_{b+1:06d}.jpg"
        img = np.array(Image.open(img_p).convert("RGB")) if img_p.exists() \
              else np.zeros((128,128,3),dtype=np.uint8)
        ax_img.imshow(img)

        yellow = 0
        for (r, c) in np.argwhere(v_bin[b]):
            rect = Rectangle((c*4, r*4), 4, 4, ec="#FFC300", lw=0.4, fc="none", alpha=0.6)
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
        v_hits_all.append(red / max(1, len(edges)))
        ax_img.set_title(
            f"frame #{b+1}   t={b/FPS:.1f}s\n"
            f"Glasso valid {true_val[np.where(joint_bins==b)[0][0]]:.0%}  "
            f"({red} red, {yellow} yellow)",
            fontsize=9)
        ax_img.axis("off")

        # waveform
        ax_wav = fig.add_subplot(outer[1, col])
        s0 = int(b/FPS*SR)
        seg = wav[s0:s0+CLIP_LEN]
        if len(seg) < CLIP_LEN: seg = np.pad(seg, (0, CLIP_LEN-len(seg)))
        ax_wav.plot(np.arange(len(seg))/SR, seg, lw=0.4, color="#333")
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
        a_hits_all.append(rbands / max(1, len(edges)))
        ax_wav.set_xlim(0, 1); ax_wav.set_ylim(-1.05, 1.05); ax_wav.grid(alpha=0.2)
        ax_wav.set_title(f"waveform  red={rbands}  yellow={ybands}", fontsize=9)
        if col == 0:
            ax_wav.set_xlabel("s"); ax_wav.set_ylabel("amp")

        # FFT
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
        f"v4 sparse-filtered · Pair #{pair['pair_idx']+1}  ·  "
        f"V-L{pair['v_level']}-{pair['v_code']:03d}  ↔  A-L{pair['a_level']}-{pair['a_code']:03d}   ·   "
        f"CCA r = {pair['r_canonical']:+.3f}   ·   Glasso α = {alpha:.2f}\n"
        f"{len(edges)} top edges  ·  mean validation {grounding:.0%}   "
        f"(V hit {np.mean(v_hits_all):.0%}, A hit {np.mean(a_hits_all):.0%})",
        fontsize=13, fontweight="bold")

    outp = OUT / f"pair_{pair['pair_idx']+1:02d}_v4.png"
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "pair_idx": pair["pair_idx"],
        "grounding": grounding,
        "n_edges": len(edges),
        "top_bins": [int(b) for b in top_bins],
        "png": str(outp.relative_to(BASE)),
        "v_level": pair["v_level"], "v_code": pair["v_code"],
        "a_level": pair["a_level"], "a_code": pair["a_code"],
        "cca_r": pair["r_canonical"],
    }


def main():
    log("Loading embeddings + token maps ...")
    data = {
        "V_emb":    np.load(EMB / "V_embeddings.npy"),
        "A_emb":    np.load(EMB / "A_embeddings.npy"),
        "V_tok_L1": np.load(TOK / "visual_tokens_L1.npy"),
        "V_tok_L2": np.load(TOK / "visual_tokens_L2.npy"),
        "A_tok_L1": np.load(TOK / "audio_tokens_L1.npy"),
        "A_tok_L2": np.load(TOK / "audio_tokens_L2.npy"),
    }
    T = min(v.shape[0] for v in data.values())
    for k in data: data[k] = data[k][:T]

    pairs = fit_cca_and_identify_pairs(data)
    if not pairs:
        log("No pairs after sparsity filtering; consider loosening thresholds.")
        return

    tokens = {k[-2:]: data[k] for k in ("V_tok_L1", "V_tok_L2", "A_tok_L1", "A_tok_L2")}
    # keys: V1, V2, A1, A2  — but slicing above gives "L1" etc so fix:
    tokens = {"V1": data["V_tok_L1"], "V2": data["V_tok_L2"],
              "A1": data["A_tok_L1"], "A2": data["A_tok_L2"]}

    wav, sr = sf.read(AUDIO)
    if wav.ndim > 1: wav = wav.mean(1)
    p = np.abs(wav).max()
    if p > 0: wav = wav/p * 0.95

    log(f"\n── v4 per-pair (sparsity-filtered + top-{MAX_EDGES}) ──")
    results = []
    for pair in pairs:
        r = process_pair(pair, tokens, wav)
        if r is None:
            log(f"  pair #{pair['pair_idx']+1}: skipped")
            continue
        results.append(r)
        log(f"  pair #{pair['pair_idx']+1:02d}: "
            f"{r['n_edges']} edges, validation {r['grounding']:.0%}")

    df = pd.DataFrame(results).sort_values("grounding", ascending=False)
    df.to_csv(BASE / "glasso_v4_summary.csv", index=False)

    log("")
    log("── v4 summary ──")
    log(df.to_string(index=False))
    log(f"\nMean validation v4: {df['grounding'].mean():.0%}")
    log(f"≥ 75%: {(df['grounding'] >= 0.75).sum()}/{len(df)}")
    log(f"≥ 50%: {(df['grounding'] >= 0.5).sum()}/{len(df)}")

    # Index page
    n = len(df); n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols*10, n_rows*4.2 + 1.5))
    gs = fig.add_gridspec(n_rows+1, n_cols, height_ratios=[0.4]+[1]*n_rows,
                          hspace=0.35, wspace=0.10, left=0.03, right=0.98,
                          top=0.97, bottom=0.02)
    ax_h = fig.add_subplot(gs[0, :]); ax_h.axis("off")
    ax_h.text(0.5, 0.75, "v4: Sparsity-Filtered Glasso Pipeline",
              ha="center", va="center", fontsize=20, fontweight="bold",
              color="#065A82", transform=ax_h.transAxes)
    ax_h.text(0.5, 0.20,
              f"only pairs whose codes fire in <25% of positions (localized events, not background) · "
              f"top-{MAX_EDGES} edges",
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
                     f"({int(r['n_edges'])} edges)  CCA r={r['cca_r']:+.2f}",
                     fontsize=11)
    for i in range(len(df), n_rows*n_cols):
        row, col = i // n_cols + 1, i % n_cols
        ax = fig.add_subplot(gs[row, col]); ax.axis("off")
    fig.savefig(BASE/"index_glasso_v4.png", dpi=100, bbox_inches="tight",
                facecolor="white")
    log(f"\n✅ v4 outputs in {OUT}/")


if __name__ == "__main__":
    main()
