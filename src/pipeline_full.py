"""
End-to-end cross-modal fragment localization pipeline.

Stages:
  1  Fit CCA on the 64-D means (already done, refit here for K=32 components)
  2  For each canonical component, identify top V-code + top A-code → matched pair
  3  For each matched pair (parallelised across 20 pairs):
     a  reverse-lookup V-code in per-frame index maps  → (T, 32, 32) binary
        downsample 4×  →  (T, 64) spatial
     b  reverse-lookup A-code in per-window index seqs → (T, 1000) binary
        downsample 20× →  (T, 50) temporal
     c  concatenate → (T, 114); fit Graphical Lasso
     d  extract cross-block edges (spatial ↔ temporal)
     e  reverse edges back to pixel regions and audio-time bands + FFT
     f  render composite figure
  4  Compile summary index page ranking all pairs
"""
import os
# Suppress internal threading so joblib parallelism scales cleanly
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import soundfile as sf
from sklearn.cross_decomposition import CCA
from sklearn.covariance import GraphicalLasso
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────
ROOT   = Path(os.environ.get("MMVQ_ROOT", "."))
EMB    = ROOT / "results/embedding_glasso"
TOK    = ROOT / "tokens_rvq"
FRAMES = ROOT / "data/frames"
AUDIO  = ROOT / "data/audio.wav"
OUT    = ROOT / "results/pipeline_final"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "pairs").mkdir(exist_ok=True)

# ── Constants ───────────────────────────────────────────────────────────
FPS       = 4.0
SR        = 16000
CLIP_LEN  = 16000
FRAME_H, FRAME_W = 128, 128
LAT_H, LAT_W     = 32, 32
SPATIAL_DS       = 4                # 32×32 → 8×8
TEMPORAL_DS      = 20               # 1000 → 50
N_PAIRS          = 20
N_JOBS           = 8


# ── Helpers ─────────────────────────────────────────────────────────────
def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_data():
    log("Loading embeddings and index maps ...")
    V_emb = np.load(EMB / "V_embeddings.npy")
    A_emb = np.load(EMB / "A_embeddings.npy")
    V_tok_L1 = np.load(TOK / "visual_tokens_L1.npy")
    V_tok_L2 = np.load(TOK / "visual_tokens_L2.npy")
    A_tok_L1 = np.load(TOK / "audio_tokens_L1.npy")
    A_tok_L2 = np.load(TOK / "audio_tokens_L2.npy")

    # Align on the common T dimension
    T = min(V_emb.shape[0], A_emb.shape[0],
            V_tok_L1.shape[0], V_tok_L2.shape[0],
            A_tok_L1.shape[0], A_tok_L2.shape[0])
    log(f"  aligned T = {T}")
    return dict(
        V_emb=V_emb[:T], A_emb=A_emb[:T],
        V_tok_L1=V_tok_L1[:T], V_tok_L2=V_tok_L2[:T],
        A_tok_L1=A_tok_L1[:T], A_tok_L2=A_tok_L2[:T],
        T=T,
    )


def fit_cca(V_emb, A_emb, n_components=32):
    V = (V_emb - V_emb.mean(0)) / (V_emb.std(0) + 1e-9)
    A = (A_emb - A_emb.mean(0)) / (A_emb.std(0) + 1e-9)
    log(f"Fitting CCA with n_components={n_components} on standardised embeddings ...")
    cca = CCA(n_components=n_components, max_iter=1500, tol=1e-5)
    cca.fit(V, A)
    U, S = cca.transform(V, A)
    corrs = np.array([np.corrcoef(U[:, k], S[:, k])[0, 1]
                      for k in range(n_components)])
    log(f"  top canonical correlations: "
        f"r1={corrs[0]:+.3f}, r5={corrs[4]:+.3f}, r10={corrs[9]:+.3f}, "
        f"r20={corrs[min(19, n_components-1)]:+.3f}")
    return cca, U, S, corrs, V, A


def identify_matched_pairs(V_tok_L1, V_tok_L2, A_tok_L1, A_tok_L2,
                            U, S, K_V=256, K_A=128, n_pairs=N_PAIRS):
    """
    For each canonical component k:
      correlate every visual code's per-bin presence with U[:, k]
      correlate every audio code's per-bin presence with S[:, k]
    Return top-1 V-code and top-1 A-code per component as the matched pair.
    (Both levels of RVQ tried; keep whichever gives strongest correlation.)
    """
    log("Identifying matched pairs from canonical component loadings ...")
    T = U.shape[0]

    # Pre-compute per-bin presence for every (level, code)
    def presence(tok, K, T):
        # tok: (T, ...), returns (T, K) binary presence
        flat = tok.reshape(T, -1)
        P = np.zeros((T, K), dtype=np.float32)
        for k in range(K):
            P[:, k] = (flat == k).any(axis=1).astype(np.float32)
        return P

    V_pres_L1 = presence(V_tok_L1, K_V, T)
    V_pres_L2 = presence(V_tok_L2, K_V, T)
    A_pres_L1 = presence(A_tok_L1, K_A, T)
    A_pres_L2 = presence(A_tok_L2, K_A, T)

    def top_code(pres_L1, pres_L2, canonical):
        best_corr = 0; best_level = 1; best_code = 0
        for level, P in enumerate([pres_L1, pres_L2], start=1):
            for k in range(P.shape[1]):
                if P[:, k].std() == 0:
                    continue
                c = np.corrcoef(P[:, k], canonical)[0, 1]
                if abs(c) > abs(best_corr):
                    best_corr = c; best_level = level; best_code = k
        return best_level, best_code, best_corr

    pairs = []
    n_comp = U.shape[1]
    for cc in range(min(n_pairs, n_comp)):
        v_lvl, v_code, v_corr = top_code(V_pres_L1, V_pres_L2, U[:, cc])
        a_lvl, a_code, a_corr = top_code(A_pres_L1, A_pres_L2, S[:, cc])
        r_cc = np.corrcoef(U[:, cc], S[:, cc])[0, 1]
        pairs.append({
            "pair_idx":  cc,
            "cc":        cc,
            "v_level":   v_lvl, "v_code": v_code, "v_corr": float(v_corr),
            "a_level":   a_lvl, "a_code": a_code, "a_corr": float(a_corr),
            "r_canonical": float(r_cc),
            "score":     float(abs(r_cc) * abs(v_corr) * abs(a_corr)),
        })
        log(f"  pair #{cc+1:>2}  CC{cc+1}  r={r_cc:+.3f}  "
            f"→ V-L{v_lvl}-{v_code:03d} (r={v_corr:+.3f})  "
            f"A-L{a_lvl}-{a_code:03d} (r={a_corr:+.3f})")

    pairs_sorted = sorted(pairs, key=lambda p: -p["score"])
    return pairs_sorted


# ── Per-pair cycle ──────────────────────────────────────────────────────
def build_spatial_map(v_tok, v_code):
    """v_tok: (T, 32, 32).  Returns binary (T, 32, 32)."""
    return (v_tok == v_code).astype(np.float32)


def build_temporal_map(a_tok, a_code):
    """a_tok: (T, 1000). Returns binary (T, 1000)."""
    return (a_tok == a_code).astype(np.float32)


def spatial_downsample(binary_T_H_W, factor):
    T, H, W = binary_T_H_W.shape
    H2, W2 = H // factor, W // factor
    return binary_T_H_W.reshape(T, H2, factor, W2, factor).mean(axis=(2, 4))


def temporal_downsample(binary_T_L, factor):
    T, L = binary_T_L.shape
    L2 = L // factor
    return binary_T_L[:, :L2 * factor].reshape(T, L2, factor).mean(axis=2)


def fit_glasso_with_fallback(X, alphas=(0.1, 0.15, 0.2, 0.3, 0.4)):
    """Try increasingly large alpha until convergence."""
    for a in alphas:
        try:
            m = GraphicalLasso(alpha=a, max_iter=200, tol=1e-3)
            m.fit(X)
            return m, a
        except Exception:
            continue
    return None, None


def render_composite(pair, frame_idx, v_regions, a_regions, corrs,
                     img, wav_seg, out_path):
    """
    v_regions: list of (row_128, col_128, h, w) rectangles on the 128×128 frame
    a_regions: list of (start_sample, dur_samples) within the 1-sec waveform
    corrs:     list of partial-correlation values (one per surviving edge)
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1],
                          hspace=0.30, wspace=0.28,
                          left=0.05, right=0.97, top=0.94, bottom=0.06)

    # ── (0, 0-1) Frame with spatial boxes ─────────────────────────────
    ax_img = fig.add_subplot(gs[0, :2])
    ax_img.imshow(img)
    for (r, c, h, w) in v_regions:
        rect = Rectangle((c, r), w, h, edgecolor="red", linewidth=2.2,
                         facecolor="none")
        ax_img.add_patch(rect)
    ax_img.set_title(
        f"Frame at t = {frame_idx/FPS:.1f} s  ·  "
        f"spatial regions where V-L{pair['v_level']}-{pair['v_code']:03d} lives",
        fontsize=12)
    ax_img.axis("off")

    # ── (0, 2)  Info panel  ─────────────────────────────────────────
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis("off")
    info = (
        f"Matched pair #{pair['pair_idx']+1}\n"
        f"─────────────────────\n"
        f"CCA component: CC{pair['cc']+1}\n"
        f"canonical r  : {pair['r_canonical']:+.3f}\n\n"
        f"V-code : L{pair['v_level']}-{pair['v_code']:03d}\n"
        f"  drive r on U : {pair['v_corr']:+.3f}\n\n"
        f"A-code : L{pair['a_level']}-{pair['a_code']:03d}\n"
        f"  drive r on S : {pair['a_corr']:+.3f}\n\n"
        f"Best-example frame:\n  t = {frame_idx/FPS:.1f} s\n\n"
        f"Cross-modal edges:\n  {pair['n_edges']} surviving\n"
        f"  Glasso α = {pair['alpha']:.3f}\n\n"
        f"Top edges’ partial corr:\n"
        + ("\n".join(f"  {i+1}. {c:+.3f}" for i, c in enumerate(corrs[:5])))
    )
    ax_info.text(0.02, 0.98, info, fontsize=10, family="monospace",
                 va="top", transform=ax_info.transAxes)

    # ── (1, 0-1) Waveform with temporal bands ────────────────────────
    ax_wav = fig.add_subplot(gs[1, :2])
    t_axis = np.arange(len(wav_seg)) / SR
    ax_wav.plot(t_axis, wav_seg, lw=0.5, color="#333")
    for (s0, dur) in a_regions:
        ax_wav.axvspan(s0 / SR, (s0 + dur) / SR, color="red", alpha=0.35)
    ax_wav.set_xlim(0, 1.0)
    ax_wav.set_ylim(-1.05, 1.05)
    ax_wav.set_xlabel("time within 1-sec window (s)")
    ax_wav.set_title(
        f"Waveform of paired 1-sec window  ·  "
        f"red bands = positions where A-L{pair['a_level']}-{pair['a_code']:03d} activates",
        fontsize=12)
    ax_wav.grid(alpha=0.25)

    # ── (1, 2) FFT of highlighted audio ─────────────────────────────
    ax_fft = fig.add_subplot(gs[1, 2])
    for (s0, dur) in a_regions:
        seg = wav_seg[s0:s0 + dur]
        if len(seg) < 32:
            continue
        F = np.abs(np.fft.rfft(seg))
        f_axis = np.fft.rfftfreq(len(seg), 1 / SR)
        ax_fft.semilogy(f_axis, F, lw=0.8, alpha=0.7)
    ax_fft.set_xlim(0, 8000)
    ax_fft.set_xlabel("frequency (Hz)")
    ax_fft.set_ylabel("|FFT|")
    ax_fft.set_title("FFT of highlighted audio bands", fontsize=11)
    ax_fft.grid(alpha=0.25)

    fig.suptitle(
        f"Cross-modal fragment localization  ·  pair #{pair['pair_idx']+1}  "
        f"·  V-L{pair['v_level']}-{pair['v_code']:03d}  ↔  "
        f"A-L{pair['a_level']}-{pair['a_code']:03d}",
        fontsize=14, fontweight="bold")

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def analyze_pair(pair, tok_data, wav_full):
    """Full per-pair cycle. Runs in a worker process."""
    try:
        pair_idx = pair["pair_idx"]
        v_tok = tok_data["V_tok_L1"] if pair["v_level"] == 1 else tok_data["V_tok_L2"]
        a_tok = tok_data["A_tok_L1"] if pair["a_level"] == 1 else tok_data["A_tok_L2"]

        # (a) reverse-lookup occurrences
        v_binary_T_H_W = build_spatial_map(v_tok, pair["v_code"])   # (T, 32, 32)
        a_binary_T_L   = build_temporal_map(a_tok, pair["a_code"])  # (T, 1000)

        # If code fires in fewer than 20 bins, skip — not enough signal
        n_frames_with_v = (v_binary_T_H_W.sum(axis=(1, 2)) > 0).sum()
        n_bins_with_a   = (a_binary_T_L.sum(axis=1) > 0).sum()
        if n_frames_with_v < 20 or n_bins_with_a < 20:
            return {**pair, "status": "skipped (too rare)",
                    "n_frames_with_v": int(n_frames_with_v),
                    "n_bins_with_a":   int(n_bins_with_a)}

        # (b) downsample
        v_ds = spatial_downsample(v_binary_T_H_W, SPATIAL_DS)      # (T, 8, 8)
        v_flat = v_ds.reshape(v_ds.shape[0], -1)                    # (T, 64)
        a_ds   = temporal_downsample(a_binary_T_L, TEMPORAL_DS)     # (T, 50)

        # (c) concatenate and filter degenerate columns
        X = np.concatenate([v_flat, a_ds], axis=1)                  # (T, 114)
        col_var = X.var(0)
        keep = col_var > 1e-8
        n_v_full = v_flat.shape[1]
        v_keep_mask = keep[:n_v_full]
        a_keep_mask = keep[n_v_full:]
        v_keep_idx = np.where(v_keep_mask)[0]                       # spatial 0..63
        a_keep_idx = np.where(a_keep_mask)[0]                       # temporal 0..49
        X = X[:, keep]
        if X.shape[1] < 8 or len(v_keep_idx) < 3 or len(a_keep_idx) < 3:
            return {**pair, "status": "skipped (too few features)"}
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)

        # (d) fit Glasso
        model, alpha = fit_glasso_with_fallback(X)
        if model is None:
            return {**pair, "status": "glasso failed"}
        prec = model.precision_
        D = np.sqrt(np.diag(prec))
        partial = -prec / np.outer(D, D)
        np.fill_diagonal(partial, 1.0)

        n_v_eff = len(v_keep_idx)
        cross = partial[:n_v_eff, n_v_eff:]                         # (n_v, n_a)

        # (e) extract surviving cross-modal edges
        thresh = 0.05
        edges = []
        for i in range(cross.shape[0]):
            for j in range(cross.shape[1]):
                pc = cross[i, j]
                if abs(pc) >= thresh:
                    edges.append({
                        "spatial_pos": int(v_keep_idx[i]),   # 0..63
                        "temporal_pos": int(a_keep_idx[j]),  # 0..49
                        "partial_corr": float(pc),
                    })
        edges.sort(key=lambda e: -abs(e["partial_corr"]))

        if not edges:
            return {**pair, "status": "no edges above threshold",
                    "alpha": alpha, "n_edges": 0}

        # (f) reverse-map + render composite figure
        # Choose the frame where V-code fires most strongly among frames where
        # A-code also has some activity in the same bin
        frame_activity_v = v_binary_T_H_W.sum(axis=(1, 2))          # (T,)
        bin_activity_a   = a_binary_T_L.sum(axis=1)                 # (T,)
        joint_score = frame_activity_v * (bin_activity_a > 0)
        if joint_score.max() == 0:
            best_frame = int(np.argmax(frame_activity_v))
        else:
            best_frame = int(np.argmax(joint_score))

        # Spatial rectangles on 128×128
        # Each spatial index p ∈ [0, 64) → block position in 8×8 grid
        # Each 8×8 grid cell corresponds to a 4×4 patch of the 32×32 latent
        # which itself corresponds to a 16×16 pixel region on the 128×128 frame
        v_regions = []
        for e in edges[:8]:
            p = e["spatial_pos"]
            r_8, c_8 = divmod(p, 8)
            r0 = r_8 * 16; c0 = c_8 * 16
            v_regions.append((r0, c0, 16, 16))

        # Temporal bands within the 1-sec window
        # Each temporal index τ ∈ [0, 50) covers 20 latent positions = 320 samples = 20 ms
        a_regions = []
        for e in edges[:8]:
            t = e["temporal_pos"]
            s0 = t * (CLIP_LEN // (a_ds.shape[1]))     # start sample
            dur = CLIP_LEN // (a_ds.shape[1])
            a_regions.append((s0, dur))

        # Load frame image
        try:
            from PIL import Image
            img_path = FRAMES / f"frame_{best_frame+1:06d}.jpg"
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            img = np.zeros((128, 128, 3), dtype=np.uint8)

        # Extract 1-sec audio segment starting at best_frame's time
        s_start = int(best_frame / FPS * SR)
        wav_seg = wav_full[s_start:s_start + SR]
        if len(wav_seg) < SR:
            wav_seg = np.pad(wav_seg, (0, SR - len(wav_seg)))

        # Fill in metadata for the composite
        pair["n_edges"] = len(edges)
        pair["alpha"]   = alpha

        composite_path = OUT / "pairs" / f"pair_{pair_idx+1:02d}.png"
        render_composite(pair, best_frame, v_regions, a_regions,
                         [e["partial_corr"] for e in edges],
                         img, wav_seg, composite_path)

        # FFT peak analysis of the highlighted audio
        peak_freqs = []
        for (s0, dur) in a_regions:
            seg = wav_seg[s0:s0 + dur]
            if len(seg) < 32: continue
            F = np.abs(np.fft.rfft(seg))
            f_axis = np.fft.rfftfreq(len(seg), 1 / SR)
            peak = f_axis[np.argmax(F)]
            peak_freqs.append(float(peak))

        return {
            **pair,
            "status":         "ok",
            "alpha":          alpha,
            "n_edges":        len(edges),
            "top_edges":      edges[:10],
            "best_frame":     best_frame,
            "best_frame_time_s": float(best_frame / FPS),
            "fft_peak_freqs_hz": peak_freqs,
            "composite_png":  str(composite_path.relative_to(OUT)),
        }
    except Exception as e:
        import traceback
        return {**pair, "status": f"error: {e}",
                "trace": traceback.format_exc()[:500]}


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log("=" * 60)
    log("Cross-modal fragment localization pipeline")
    log("=" * 60)

    data = load_data()

    cca, U, S, corrs, V_std, A_std = fit_cca(data["V_emb"], data["A_emb"], 32)

    pairs = identify_matched_pairs(
        data["V_tok_L1"], data["V_tok_L2"],
        data["A_tok_L1"], data["A_tok_L2"],
        U, S,
    )

    # Save the CCA info and matched-pair list
    with open(OUT / "cca_info.json", "w") as f:
        json.dump({
            "canonical_correlations": corrs.tolist(),
            "n_pairs": len(pairs),
            "matched_pairs": pairs,
        }, f, indent=2)

    log(f"\n── Running per-pair cycles ({N_PAIRS} pairs, {N_JOBS} parallel jobs) ──")
    tok_data = {k: v for k, v in data.items()
                if k in ("V_tok_L1", "V_tok_L2", "A_tok_L1", "A_tok_L2")}

    # Load raw audio once (shared read; joblib will copy per worker)
    wav_full, sr = sf.read(AUDIO)
    if wav_full.ndim > 1: wav_full = wav_full.mean(1)
    peak = np.abs(wav_full).max()
    if peak > 0: wav_full = wav_full / peak * 0.95

    t_pipe = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(analyze_pair)(p, tok_data, wav_full)
        for p in pairs[:N_PAIRS]
    )
    log(f"  parallel section done in {time.time()-t_pipe:.0f} s")

    # Save results summary
    with open(OUT / "results.json", "w") as f:
        json.dump({
            "n_pairs_attempted": len(results),
            "results": results,
            "total_wall_time_s": time.time() - t0,
        }, f, indent=2, default=str)

    # Build a summary index page
    ok = [r for r in results if r.get("status") == "ok"]
    log(f"\n── Summary ──")
    log(f"  attempted : {len(results)}")
    log(f"  succeeded : {len(ok)}")
    log(f"  skipped   : {len(results) - len(ok)}")
    log(f"  wall time : {time.time()-t0:.0f} s")

    # Simple summary table
    rows = []
    for r in results:
        rows.append({
            "pair":  r["pair_idx"] + 1,
            "CC":    r["cc"] + 1,
            "r":     round(r["r_canonical"], 3),
            "V":     f"L{r['v_level']}-{r['v_code']:03d}",
            "A":     f"L{r['a_level']}-{r['a_code']:03d}",
            "status": r.get("status", "?"),
            "edges":  r.get("n_edges", 0),
            "png":    r.get("composite_png", ""),
        })
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)

    log(f"\n✅ Outputs in {OUT}/")
    log(f"   composite figures: {OUT/'pairs/'}")
    log(f"   summary.csv, results.json, cca_info.json")


if __name__ == "__main__":
    main()
