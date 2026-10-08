"""
v5: Synchronized-subset Graphical Lasso.

Change vs v2: instead of running Graphical Lasso on all T=2169 time bins
for each pair, restrict to the subset where BOTH V-code and A-code fire
(the co-occurrence subset). This removes noise from bins where the
cross-modal binding is absent by construction.

Rest of the pipeline (validation metric, rendering) is unchanged from v2.
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
OUT    = BASE / "pairs_glasso_v5"
OUT.mkdir(parents=True, exist_ok=True)

FPS, SR, CLIP_LEN = 4.0, 16000, 16000
SPATIAL_DS, TEMPORAL_DS = 4, 20
FRAME_H, FRAME_W = 128, 128
N_PAIRS = 20
MIN_JOINT_BINS = 30                          # ← need enough samples for Glasso


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


def spatial_downsample(binary_T_H_W, factor):
    T, H, W = binary_T_H_W.shape
    H2, W2 = H // factor, W // factor
    return binary_T_H_W.reshape(T, H2, factor, W2, factor).mean(axis=(2, 4))


def temporal_downsample(binary_T_L, factor):
    T, L = binary_T_L.shape
    L2 = L // factor
    return binary_T_L[:, :L2*factor].reshape(T, L2, factor).mean(axis=2)


def fit_glasso_fallback(X, alphas=(0.10, 0.15, 0.20, 0.30, 0.40, 0.5)):
    for a in alphas:
        try:
            m = GraphicalLasso(alpha=a, max_iter=200, tol=1e-3); m.fit(X)
            return m, a
        except Exception:
            continue
    return None, None


# ── Fit CCA and match pairs (identical to pipeline_full.py) ────────────
def fit_cca_and_identify_pairs(data):
    log("Fitting CCA (K=32) ...")
    V_emb, A_emb = data["V_emb"], data["A_emb"]
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

    V1p = per_bin_presence(data["V_tok_L1"], 256)
    V2p = per_bin_presence(data["V_tok_L2"], 256)
    A1p = per_bin_presence(data["A_tok_L1"], 128)
    A2p = per_bin_presence(data["A_tok_L2"], 128)

    def top_code(P1, P2, canonical):
        best_c, best_lvl, best_code = 0.0, 1, 0
        for lvl, P in enumerate([P1, P2], start=1):
            for k in range(P.shape[1]):
                if P[:, k].std() == 0: continue
                c = np.corrcoef(P[:, k], canonical)[0, 1]
                if abs(c) > abs(best_c):
                    best_c, best_lvl, best_code = c, lvl, k
        return best_lvl, best_code, best_c

    pairs = []
    for cc in range(min(N_PAIRS, U.shape[1])):
        v_lvl, v_code, v_c = top_code(V1p, V2p, U[:, cc])
        a_lvl, a_code, a_c = top_code(A1p, A2p, S[:, cc])
        r_cc = float(np.corrcoef(U[:, cc], S[:, cc])[0, 1])
        pairs.append({
            "pair_idx": cc, "cc": cc,
            "v_level": v_lvl, "v_code": v_code, "v_corr": float(v_c),
            "a_level": a_lvl, "a_code": a_code, "a_corr": float(a_c),
            "r_canonical": r_cc,
        })
    return pairs


# ── v5 core: synchronized-subset Graphical Lasso ─────────────────────────
def stage2_edges_sync(pair, tokens):
    """Restrict Graphical Lasso to bins where BOTH codes fire."""
    v_tok = tokens["V1"] if pair["v_level"] == 1 else tokens["V2"]
    a_tok = tokens["A1"] if pair["a_level"] == 1 else tokens["A2"]
    v_bin = (v_tok == pair["v_code"]).astype(np.float32)   # (T, 32, 32)
    a_bin = (a_tok == pair["a_code"]).astype(np.float32)   # (T, 1000)

    # Build synchronized subset
    v_fires = v_bin.sum(axis=(1, 2)) > 0
    a_fires = a_bin.sum(axis=1) > 0
    sync_mask = v_fires & a_fires
    sync_idx = np.where(sync_mask)[0]
    n_sync = len(sync_idx)

    if n_sync < MIN_JOINT_BINS:
        return v_bin.astype(np.uint8), a_bin.astype(np.uint8), [], 0, n_sync

    v_ds = spatial_downsample(v_bin[sync_idx], SPATIAL_DS)      # (n_sync, 8, 8)
    v_flat = v_ds.reshape(v_ds.shape[0], -1)                    # (n_sync, 64)
    a_ds = temporal_downsample(a_bin[sync_idx], TEMPORAL_DS)    # (n_sync, 50)
    X = np.concatenate([v_flat, a_ds], axis=1)                  # (n_sync, 114)

    keep = X.var(0) > 1e-8
    n_v_full = v_flat.shape[1]
    v_keep_idx = np.where(keep[:n_v_full])[0]
    a_keep_idx = np.where(keep[n_v_full:])[0]
    X = X[:, keep]
    if X.shape[1] < 8 or len(v_keep_idx) < 3 or len(a_keep_idx) < 3:
        return v_bin.astype(np.uint8), a_bin.astype(np.uint8), [], 0, n_sync

    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    model, alpha = fit_glasso_fallback(X)
    if model is None:
        return v_bin.astype(np.uint8), a_bin.astype(np.uint8), [], 0, n_sync

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
    return v_bin.astype(np.uint8), a_bin.astype(np.uint8), edges[:10], alpha, n_sync


# ── Same rendering + validation as v2 ────────────────────────────────────
def process_pair(pair, tokens, wav):
    v_bin, a_bin, edges, alpha, n_sync = stage2_edges_sync(pair, tokens)
    if not edges:
        return {"status": "no_edges", "pair_idx": pair["pair_idx"],
                "n_sync": n_sync}

    T = v_bin.shape[0]
    v_fires = v_bin.sum(axis=(1, 2)) > 0
    a_fires = a_bin.sum(axis=1) > 0
    joint_bins = np.where(v_fires & a_fires)[0]

    val = np.zeros(len(joint_bins))
    per_bin_valid = {}
    for i, b in enumerate(joint_bins):
        vs = [e for e in edges
              if check_v_fires(v_bin[b], e["spatial_pos"])
              and check_a_fires(a_bin[b], e["temporal_pos"])]
        val[i] = len(vs) / len(edges)
        per_bin_valid[b] = vs

    order = np.argsort(-val)
    top_bins = joint_bins[order[:4]]
    top_scores = val[order[:4]]
    grounding = float(np.mean(top_scores))

    fig = plt.figure(figsize=(20, 13))
    outer = fig.add_gridspec(3, 4, height_ratios=[1.2, 0.8, 0.9],
                             hspace=0.35, wspace=0.20,
                             left=0.03, right=0.99, top=0.90, bottom=0.05)

    v_hits_all, a_hits_all = [], []
    for col, (b, sc) in enumerate(zip(top_bins, top_scores)):
        v_ok = per_bin_valid[b]

        ax_img = fig.add_subplot(outer[0, col])
        img_p = FRAMES / f"frame_{b+1:06d}.jpg"
        img = np.array(Image.open(img_p).convert("RGB")) if img_p.exists() else np.zeros((128,128,3),dtype=np.uint8)
        ax_img.imshow(img)
        yellow = 0
        for (r, c) in np.argwhere(v_bin[b]):
            rect = Rectangle((c*4, r*4), 4, 4, ec="#FFC300", lw=0.4, fc="none", alpha=0.6)
            ax_img.add_patch(rect); yellow += 1
        red = 0; drawn = set()
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
            f"Glasso valid {sc:.0%}  ({red} red, {yellow} yellow)",
            fontsize=9)
        ax_img.axis("off")

        ax_wav = fig.add_subplot(outer[1, col])
        s0 = int(b/FPS*SR)
        seg = wav[s0:s0+CLIP_LEN]
        if len(seg) < CLIP_LEN: seg = np.pad(seg, (0, CLIP_LEN-len(seg)))
        ax_wav.plot(np.arange(len(seg))/SR, seg, lw=0.4, color="#333")
        ybands = 0
        for tau in np.where(a_bin[b])[0]:
            ax_wav.axvspan(tau*16/SR, (tau*16+16)/SR, color="#FFC300", alpha=0.18)
            ybands += 1
        rbands = 0; drawn_t = set()
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
        f"v5 sync-subset Glasso · Pair #{pair['pair_idx']+1}  ·  "
        f"V-L{pair['v_level']}-{pair['v_code']:03d}  ↔  A-L{pair['a_level']}-{pair['a_code']:03d}   ·   "
        f"CCA r = {pair['r_canonical']:+.3f}   ·   Glasso α = {alpha:.2f}\n"
        f"Glasso fit on {n_sync} sync bins   ·   {len(edges)} edges   ·   "
        f"validation {grounding:.0%}   (V hit {np.mean(v_hits_all):.0%}, A hit {np.mean(a_hits_all):.0%})",
        fontsize=13, fontweight="bold")

    outp = OUT / f"pair_{pair['pair_idx']+1:02d}_v5.png"
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "status":     "ok",
        "pair_idx":   pair["pair_idx"],
        "grounding":  grounding,
        "n_sync":     n_sync,
        "n_edges":    len(edges),
        "alpha":      alpha,
        "cca_r":      pair["r_canonical"],
        "v_level":    pair["v_level"], "v_code": pair["v_code"],
        "a_level":    pair["a_level"], "a_code": pair["a_code"],
        "top_bins":   [int(b) for b in top_bins],
        "png":        str(outp.relative_to(BASE)),
    }


def main():
    log("Loading data ...")
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

    tokens = {"V1": data["V_tok_L1"], "V2": data["V_tok_L2"],
              "A1": data["A_tok_L1"], "A2": data["A_tok_L2"]}

    wav, sr = sf.read(AUDIO)
    if wav.ndim > 1: wav = wav.mean(1)
    p = np.abs(wav).max()
    if p > 0: wav = wav/p * 0.95

    log(f"\n── v5: sync-subset Graphical Lasso ──")
    results = []
    for pair in pairs:
        r = process_pair(pair, tokens, wav)
        results.append(r)
        if r["status"] == "ok":
            log(f"  pair #{pair['pair_idx']+1:02d}: sync bins = {r['n_sync']:>4}, "
                f"edges = {r['n_edges']:>2}, validation = {r['grounding']:.0%}")
        else:
            log(f"  pair #{pair['pair_idx']+1:02d}: {r['status']} (sync bins = {r.get('n_sync', 0)})")

    ok = [r for r in results if r["status"] == "ok"]
    df = pd.DataFrame(ok).sort_values("grounding", ascending=False)
    df.to_csv(BASE / "glasso_v5_summary.csv", index=False)

    log("")
    log("── v5 summary ──")
    log(df[["pair_idx", "n_sync", "n_edges", "grounding", "cca_r"]].to_string(index=False))
    log(f"\nMean validation v5: {df['grounding'].mean():.0%}")
    log(f"≥ 75%: {(df['grounding'] >= 0.75).sum()}/{len(df)}")
    log(f"≥ 50%: {(df['grounding'] >= 0.5).sum()}/{len(df)}")

    # ── Comparison table (v5 vs v2) ─────────────────────────────────────
    try:
        v2_df = pd.read_csv(BASE / "glasso_v2_summary.csv")
        v2_map = dict(zip(v2_df["pair_idx"], v2_df["grounding"]))
        log("")
        log("── Δ vs v2 (per pair) ──")
        for _, r in df.iterrows():
            v2 = v2_map.get(r["pair_idx"], np.nan)
            delta = r["grounding"] - v2 if not np.isnan(v2) else float('nan')
            log(f"  pair #{int(r['pair_idx'])+1:02d}  "
                f"v2 {v2:.0%}  →  v5 {r['grounding']:.0%}   Δ = {delta:+.0%}")
    except Exception as e:
        log(f"  (comparison table skipped: {e})")

    # ── Index page ──────────────────────────────────────────────────────
    n = len(df); n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols*10, n_rows*4.2 + 1.5))
    gs = fig.add_gridspec(n_rows+1, n_cols, height_ratios=[0.4]+[1]*n_rows,
                          hspace=0.35, wspace=0.10, left=0.03, right=0.98,
                          top=0.97, bottom=0.02)
    ax_h = fig.add_subplot(gs[0, :]); ax_h.axis("off")
    ax_h.text(0.5, 0.75, "v5: Synchronized-Subset Graphical Lasso",
              ha="center", va="center", fontsize=20, fontweight="bold",
              color="#065A82", transform=ax_h.transAxes)
    ax_h.text(0.5, 0.20,
              f"Graphical Lasso restricted to bins where BOTH codes fire — pure signal subset · "
              f"min {MIN_JOINT_BINS} bins",
              ha="center", va="center", fontsize=11, fontstyle="italic",
              color="#666", transform=ax_h.transAxes)
    for i, r in df.reset_index(drop=True).iterrows():
        row, col = i // n_cols + 1, i % n_cols
        ax = fig.add_subplot(gs[row, col])
        p_img = BASE / r["png"]
        if p_img.exists(): ax.imshow(np.array(Image.open(p_img)))
        ax.axis("off")
        ax.set_title(
            f"pair #{int(r['pair_idx'])+1}  "
            f"validation = {r['grounding']:.0%}  "
            f"({int(r['n_edges'])} edges, {int(r['n_sync'])} sync bins)",
            fontsize=11)
    for i in range(len(df), n_rows*n_cols):
        row, col = i // n_cols + 1, i % n_cols
        ax = fig.add_subplot(gs[row, col]); ax.axis("off")
    fig.savefig(BASE/"index_glasso_v5.png", dpi=100, bbox_inches="tight",
                facecolor="white")
    log(f"\n✅ v5 outputs in {OUT}/")


if __name__ == "__main__":
    main()
