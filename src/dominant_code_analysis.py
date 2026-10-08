"""
Path A: One dominant code per time bin, then find cross-modal correspondences.

For each time bin t:
  visual_dominant[t] = most-frequent code in the 32×32 latent grid of frame t
  audio_dominant [t] = most-frequent code in the 1000 latent positions of the
                       1-sec audio window at t

We analyse two ways:
  1) Direct pointwise mutual information over the (V, A) contingency table
     — Answers: "which V code co-occurs with which A code more than expected
     by chance?"
  2) One-hot encode + block-bootstrap Glasso (universal pipeline)
     — Answers: same question, but under a sparse conditional-independence
     framework. Comparable to our earlier runs.

We run each level of RVQ separately (L1 = coarse, L2 = fine).
"""
import argparse
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

TOK  = Path(os.environ.get("MMVQ_ROOT", ".")) / "tokens_rvq"
OUT  = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/dominant_code"
OUT.mkdir(parents=True, exist_ok=True)


def dominant_per_bin(tokens):
    """(T, ...) integer array  →  (T,) integer array of MODE per bin."""
    T = tokens.shape[0]
    flat = tokens.reshape(T, -1)
    modes = np.empty(T, dtype=np.int64)
    for t in range(T):
        vals, counts = np.unique(flat[t], return_counts=True)
        modes[t] = vals[counts.argmax()]
    return modes


def pmi_analysis(v_dom, a_dom, K_v, K_a):
    """Build joint contingency table + PMI per pair."""
    T = len(v_dom)
    N = np.zeros((K_v, K_a), dtype=np.float64)
    for v, a in zip(v_dom, a_dom):
        N[v, a] += 1
    # Marginals
    N_v = N.sum(1)   # (K_v,)
    N_a = N.sum(0)   # (K_a,)
    T_total = N.sum()

    # PMI:  log( p(v,a) / (p(v) p(a)) )
    p_va = N / T_total
    p_v  = N_v / T_total
    p_a  = N_a / T_total
    exp  = np.outer(p_v, p_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(p_va / exp)
        pmi[np.isinf(pmi)] = 0
        pmi[np.isnan(pmi)] = 0
    # Positive-PMI (only over-representation matters here)
    ppmi = np.maximum(0, pmi)
    return N, pmi, ppmi, N_v, N_a


def top_pairs(N, pmi, N_v, N_a, min_joint=5, top=30):
    """List top edges by PMI subject to minimum joint count."""
    K_v, K_a = N.shape
    rows = []
    for v in range(K_v):
        for a in range(K_a):
            if N[v, a] >= min_joint and pmi[v, a] > 0:
                rows.append((v, a, int(N[v, a]), int(N_v[v]), int(N_a[a]),
                             float(pmi[v, a])))
    df = pd.DataFrame(rows, columns=["v_code","a_code","joint_count","v_count","a_count","pmi"])
    df = df.sort_values("pmi", ascending=False)
    return df


def onehot_glasso(v_dom, a_dom, K_v, K_a, out_dir, level_tag,
                  n_bootstrap=30, min_active=15, alpha=None):
    """One-hot encode dominant codes → Glasso with block bootstrap."""
    T = len(v_dom)
    V = np.zeros((T, K_v), dtype=np.float64)
    A = np.zeros((T, K_a), dtype=np.float64)
    V[np.arange(T), v_dom] = 1
    A[np.arange(T), a_dom] = 1
    # Drop codes that are dominant in < min_active bins
    v_keep = V.sum(0) >= min_active
    a_keep = A.sum(0) >= min_active
    V = V[:, v_keep]; A = A[:, a_keep]
    v_idx = np.where(v_keep)[0]; a_idx = np.where(a_keep)[0]
    v_labels = [f"V{i:04d}" for i in v_idx]
    a_labels = [f"A{i:04d}" for i in a_idx]
    all_labels = v_labels + a_labels
    K_v_eff, K_a_eff = V.shape[1], A.shape[1]

    X = np.concatenate([V, A], axis=1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    print(f"  [{level_tag}] one-hot: T={T}, K_v={K_v_eff}, K_a={K_a_eff}, T/K={T/(K_v_eff+K_a_eff):.1f}")

    if alpha is None:
        try:
            alphas = np.logspace(-2, -0.3, 6)
            m = GraphicalLassoCV(alphas=alphas, cv=3, max_iter=150, n_jobs=-1)
            m.fit(X); alpha = m.alpha_
        except Exception:
            alpha = 0.05
    print(f"  [{level_tag}] α = {alpha:.4f}")

    # Simple block bootstrap
    T_ = X.shape[0]
    block_len = 20
    n_blocks = int(np.ceil(T_ / block_len))
    n_pick = max(1, int(0.5 * n_blocks))
    K_total = X.shape[1]
    stab = np.zeros((K_total, K_total), dtype=np.float32)
    rng = np.random.default_rng(0)
    succ = 0
    for b in range(n_bootstrap):
        picks = rng.choice(n_blocks, size=n_pick, replace=True)
        rows = [np.arange(i*block_len, min((i+1)*block_len, T_)) for i in picks]
        Xb = X[np.concatenate(rows)]
        try:
            m = GraphicalLasso(alpha=alpha, max_iter=80, tol=1e-3); m.fit(Xb)
            nz = (np.abs(m.precision_) > 1e-6).astype(np.float32)
            np.fill_diagonal(nz, 0); stab += nz; succ += 1
        except Exception:
            pass
    stab /= max(succ, 1)
    print(f"  [{level_tag}] bootstraps succeeded: {succ}/{n_bootstrap}")

    # Extract cross-modal edges
    cross = stab[:K_v_eff, K_v_eff:]
    print(f"  [{level_tag}] cross-modal stability distribution:")
    for th in [0.30, 0.50, 0.70, 0.90]:
        n = int((cross >= th).sum())
        print(f"      ≥ {th:.2f}: {n} edges")
    print(f"  [{level_tag}] max cross stability: {cross.max():.3f}")

    # Save cross-modal heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cross, cmap="magma", vmin=0, vmax=1, ax=ax,
                xticklabels=False, yticklabels=False,
                cbar_kws={"label": "bootstrap stability"})
    ax.set_xlabel(f"Audio codes ({K_a_eff})", fontweight="bold")
    ax.set_ylabel(f"Visual codes ({K_v_eff})", fontweight="bold")
    plt.title(f"One-hot Glasso stability [{level_tag}] α={alpha:.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / f"onehot_glasso_stability_{level_tag}.png", dpi=140)
    plt.close()

    # Return the top edges
    rows = []
    for i in range(K_v_eff):
        for j in range(K_a_eff):
            if stab[i, K_v_eff+j] >= 0.30:
                rows.append((v_labels[i], a_labels[j], float(stab[i, K_v_eff+j])))
    df = pd.DataFrame(rows, columns=["v_code","a_code","stability"])
    df = df.sort_values("stability", ascending=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=[1, 2], type=int,
                    help="RVQ levels to analyse (1=coarse, 2=fine)")
    ap.add_argument("--min_joint", type=int, default=5,
                    help="minimum joint co-occurrence count for PMI table")
    ap.add_argument("--top_pmi", type=int, default=30)
    args = ap.parse_args()

    print("── Loading raw per-position tokens ─────────────────────────")
    for L in args.levels:
        v_tok = np.load(TOK / f"visual_tokens_L{L}.npy")   # (T, 32, 32)
        a_tok = np.load(TOK / f"audio_tokens_L{L}.npy")    # (T, 1000)
        T = min(v_tok.shape[0], a_tok.shape[0])
        v_tok, a_tok = v_tok[:T], a_tok[:T]
        K_v = int(v_tok.max()) + 1
        K_a = int(a_tok.max()) + 1
        print(f"\n══ Level {L} ═══════════════════════════════════════════")
        print(f"  T={T}, K_v={K_v}, K_a={K_a}")

        print("\n  Extracting dominant code per bin ...")
        v_dom = dominant_per_bin(v_tok)
        a_dom = dominant_per_bin(a_tok)
        print(f"    visual dominant range: [{v_dom.min()}, {v_dom.max()}]  "
              f"unique: {len(np.unique(v_dom))}")
        print(f"    audio  dominant range: [{a_dom.min()}, {a_dom.max()}]  "
              f"unique: {len(np.unique(a_dom))}")

        print("\n  ─ Method 1: PMI over the joint contingency ─")
        N, pmi, ppmi, N_v, N_a = pmi_analysis(v_dom, a_dom, K_v, K_a)
        df_pmi = top_pairs(N, pmi, N_v, N_a, min_joint=args.min_joint)
        print(f"    {len(df_pmi)} pairs with joint ≥ {args.min_joint} and PMI > 0")
        print(f"    Top {min(args.top_pmi, len(df_pmi))} by PMI:")
        print(df_pmi.head(args.top_pmi).to_string(index=False))
        df_pmi.to_csv(OUT / f"pmi_edges_L{L}.csv", index=False)

        # Save PMI heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(ppmi, cmap="magma", vmin=0, vmax=max(ppmi.max(), 0.5), ax=ax,
                    xticklabels=False, yticklabels=False,
                    cbar_kws={"label": "PPMI (bits)"})
        ax.set_xlabel(f"Audio codes ({K_a})", fontweight="bold")
        ax.set_ylabel(f"Visual codes ({K_v})", fontweight="bold")
        plt.title(f"L{L}: Positive PMI  (dominant-code co-occurrence)")
        plt.tight_layout()
        plt.savefig(OUT / f"pmi_heatmap_L{L}.png", dpi=140)
        plt.close()

        print("\n  ─ Method 2: One-hot Glasso with block bootstrap ─")
        df_glasso = onehot_glasso(v_dom, a_dom, K_v, K_a, OUT, level_tag=f"L{L}",
                                   n_bootstrap=30, min_active=15)
        print(f"    Top 20 edges by bootstrap stability (≥0.30):")
        print(df_glasso.head(20).to_string(index=False))
        df_glasso.to_csv(OUT / f"onehot_glasso_edges_L{L}.csv", index=False)

        # Also save the dominant-code time series for reference
        np.save(OUT / f"visual_dominant_L{L}.npy", v_dom)
        np.save(OUT / f"audio_dominant_L{L}.npy",  a_dom)

    print(f"\n✅ All outputs in {OUT}/")


if __name__ == "__main__":
    main()
