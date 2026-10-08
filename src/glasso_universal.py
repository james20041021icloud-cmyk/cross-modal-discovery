"""
Universal cross-modal Graphical Lasso:
   1) nonparanormal transform (rank → Gaussian) for distribution-agnostic input
   2) fixed-α chosen once by CV on the full data
   3) block-bootstrap stability selection (autocorrelation-aware)
   4) edges reported with a stability score in [0, 1]
"""
import argparse, warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import norm
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso

warnings.filterwarnings("ignore")

TOK  = Path(os.environ.get("MMVQ_ROOT", ".")) / "tokens_rvq"
OUT  = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/glasso_universal"
OUT.mkdir(parents=True, exist_ok=True)


# ── Universal transforms ─────────────────────────────────────────────────
def nonparanormal(X, eps=1e-3):
    """
    Map each column to a Gaussian via its empirical CDF.
       X[:, k] -> Φ^{-1}((rank_k - 0.5) / T)
    Clipped by eps to avoid ±∞ at the tails.
    Works for binary / count / continuous columns identically.
    """
    T, K = X.shape
    Xt = np.empty_like(X, dtype=np.float64)
    for k in range(K):
        col = X[:, k]
        # average-rank handles ties (which are frequent in binary/count data)
        ranks = pd.Series(col).rank(method="average").values
        p = (ranks - 0.5) / T
        p = np.clip(p, eps, 1 - eps)
        Xt[:, k] = norm.ppf(p)
    return Xt


def acf_block_length(x, max_lag=50, thresh=0.1):
    """Estimate block length from the first lag whose autocorrelation drops
    below `thresh`. Falls back to `max_lag` if never crosses."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = x.var()
    if var == 0:
        return 1
    b = max_lag
    for lag in range(1, max_lag + 1):
        cov = (x[:-lag] * x[lag:]).mean()
        rho = cov / var
        if abs(rho) < thresh:
            b = lag
            break
    return max(1, b)


def summarize_block_length(X, max_lag=50):
    """Median block length across the K features."""
    bs = []
    for k in range(min(50, X.shape[1])):        # sample first 50 columns
        bs.append(acf_block_length(X[:, k], max_lag=max_lag))
    return int(np.median(bs))


# ── Filtering ────────────────────────────────────────────────────────────
def filter_binary_ready(v_hist, a_hist, min_frac=0.02, max_frac=0.98):
    """Binary-ise then keep only columns firing in [min_frac, max_frac] of bins."""
    v_bin = (v_hist > 0).astype(np.float64)
    a_bin = (a_hist > 0).astype(np.float64)
    v_keep = (v_bin.mean(0) >= min_frac) & (v_bin.mean(0) <= max_frac)
    a_keep = (a_bin.mean(0) >= min_frac) & (a_bin.mean(0) <= max_frac)
    return v_bin[:, v_keep], a_bin[:, a_keep], v_keep, a_keep


# ── Block bootstrap Glasso ───────────────────────────────────────────────
def bootstrap_glasso(Xs, alpha, block_len, n_bootstrap=50, frac=0.5, seed=0):
    """
    Run Glasso `n_bootstrap` times on random block-subsamples of Xs.
    Return per-edge stability = fraction of runs where entry is non-zero.
    """
    T, K = Xs.shape
    n_blocks_total = int(np.ceil(T / block_len))
    n_pick = max(1, int(np.round(n_blocks_total * frac)))

    rng = np.random.default_rng(seed)
    stability = np.zeros((K, K), dtype=np.float32)

    for b in range(n_bootstrap):
        picks = rng.choice(n_blocks_total, size=n_pick, replace=True)
        rows = []
        for i in picks:
            s = i * block_len
            rows.append(np.arange(s, min(s + block_len, T)))
        idx = np.concatenate(rows)
        Xb = Xs[idx]

        try:
            m = GraphicalLasso(alpha=alpha, max_iter=100, tol=1e-3)
            m.fit(Xb)
            nz = (np.abs(m.precision_) > 1e-6).astype(np.float32)
            np.fill_diagonal(nz, 0)
            stability += nz
        except Exception:
            pass
        if (b + 1) % 10 == 0:
            print(f"    bootstrap {b+1}/{n_bootstrap}")

    stability /= n_bootstrap
    return stability


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_dir", default=str(TOK))
    ap.add_argument("--out_dir",    default=str(OUT))
    ap.add_argument("--n_bootstrap", type=int, default=50)
    ap.add_argument("--stability_thresh", type=float, default=0.60)
    ap.add_argument("--force_alpha", type=float, default=None,
                    help="skip CV, use this alpha")
    args = ap.parse_args()

    tokens_dir = Path(args.tokens_dir)
    out_dir    = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("── Loading token histograms ────────────────────────────────")
    v_hist = np.load(tokens_dir / "visual_hist.npy")
    a_hist = np.load(tokens_dir / "audio_hist.npy")
    print(f"  visual_hist: {v_hist.shape}")
    print(f"  audio_hist : {a_hist.shape}")

    print("\n── Binary-ising and filtering codes ────────────────────────")
    v_bin, a_bin, v_keep, a_keep = filter_binary_ready(v_hist, a_hist)
    K_v_eff, K_a_eff = v_bin.shape[1], a_bin.shape[1]
    v_labels = [f"V{i:04d}" for i in np.where(v_keep)[0]]
    a_labels = [f"A{i:04d}" for i in np.where(a_keep)[0]]
    all_labels = v_labels + a_labels
    print(f"  visual: kept {K_v_eff} / {len(v_keep)}")
    print(f"  audio : kept {K_a_eff} / {len(a_keep)}")

    X = np.concatenate([v_bin, a_bin], axis=1)
    T, K = X.shape
    print(f"\n  joint: T={T} × K={K}   (T/K = {T/K:.2f})")

    print("\n── UPGRADE 1: nonparanormal transform ─────────────────────")
    Xt = nonparanormal(X)
    print(f"  transformed shape: {Xt.shape}   "
          f"col-mean ≈ {Xt.mean():.3f}, col-std ≈ {Xt.std():.3f}")

    print("\n── UPGRADE 2: pick α by CV, once on full data ─────────────")
    if args.force_alpha is None:
        alphas = np.logspace(-2.5, -0.5, 8)
        try:
            model_full = GraphicalLassoCV(alphas=alphas, cv=3, max_iter=150, n_jobs=-1)
            model_full.fit(Xt)
            alpha = model_full.alpha_
        except Exception as e:
            print(f"  CV failed ({e}); falling back to α=0.1")
            alpha = 0.1
    else:
        alpha = args.force_alpha
    print(f"  α (locked) = {alpha:.4f}")

    print("\n── UPGRADE 3: block-bootstrap stability selection ─────────")
    block_len = summarize_block_length(Xt, max_lag=30)
    print(f"  auto block length (median across features): {block_len} bins")
    print(f"  running {args.n_bootstrap} bootstraps (half-sample blocks)...")
    stability = bootstrap_glasso(Xt, alpha=alpha, block_len=block_len,
                                 n_bootstrap=args.n_bootstrap, frac=0.5)

    np.save(out_dir / "stability_matrix.npy", stability)
    thresh = args.stability_thresh
    stable_edges = (stability >= thresh)
    np.fill_diagonal(stable_edges, False)

    # Extract edge list
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            if stable_edges[i, j]:
                src, dst = all_labels[i], all_labels[j]
                typ = "cross" if src[0] != dst[0] else ("visual" if src[0] == "V" else "audio")
                rows.append((src, dst, typ, float(stability[i, j])))
    edges = pd.DataFrame(rows, columns=["a", "b", "type", "stability"])
    edges = edges.sort_values("stability", ascending=False)
    edges.to_csv(out_dir / "stable_edges.csv", index=False)

    n_cross = (edges.type == "cross").sum()
    n_vis   = (edges.type == "visual").sum()
    n_aud   = (edges.type == "audio").sum()
    print(f"\n  edges surviving stability ≥ {thresh}:")
    print(f"    within-visual : {n_vis}")
    print(f"    within-audio  : {n_aud}")
    print(f"    CROSS-MODAL   : {n_cross}  ← headline")

    if n_cross:
        print("\n  top cross-modal (high-stability) edges:")
        print(edges[edges.type == "cross"].head(20).to_string(index=False))

    # ── Visualisations ─────────────────────────────────────────────────
    print("\n── Rendering plots ────────────────────────────────────────")

    # 1) Stability heatmap of the cross-modal block
    cross_block = stability[:K_v_eff, K_v_eff:]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cross_block, cmap="magma", vmin=0, vmax=1, ax=ax,
                xticklabels=False, yticklabels=False,
                cbar_kws={"label": "bootstrap stability"})
    ax.set_xlabel(f"Audio codes ({K_a_eff})", fontweight="bold")
    ax.set_ylabel(f"Visual codes ({K_v_eff})", fontweight="bold")
    plt.title(f"Cross-modal block: bootstrap stability "
              f"(α={alpha:.3f}, B={args.n_bootstrap})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "cross_block_stability.png", dpi=140)
    plt.close()

    # 2) Bipartite graph of stable cross edges
    cross = edges[edges.type == "cross"].copy()
    if len(cross):
        B = nx.Graph()
        v_nodes = sorted({n for n in set(cross.a)|set(cross.b) if n.startswith("V")})
        a_nodes = sorted({n for n in set(cross.a)|set(cross.b) if n.startswith("A")})
        for n in v_nodes: B.add_node(n, mod="V")
        for n in a_nodes: B.add_node(n, mod="A")
        for _, r in cross.iterrows():
            B.add_edge(r.a, r.b, stability=r.stability)

        pos = {n: (0, -i) for i, n in enumerate(v_nodes)}
        step = len(v_nodes) / max(1, len(a_nodes))
        for i, n in enumerate(a_nodes):
            pos[n] = (1, -i * step)

        fig, ax = plt.subplots(figsize=(11, max(6, min(24, 0.30*max(len(v_nodes),len(a_nodes))))))
        colors = ["#2E86AB" if B.nodes[n]["mod"] == "V" else "#E63946" for n in B.nodes()]
        nx.draw_networkx_nodes(B, pos, node_color=colors, node_size=380,
                               edgecolors="white", linewidths=1.5, ax=ax)
        nx.draw_networkx_labels(B, pos, font_size=6.5, font_color="white", ax=ax)
        # edge width = 3 * stability
        widths = [3 * B[u][v]["stability"] for u, v in B.edges()]
        nx.draw_networkx_edges(B, pos, width=widths, edge_color="#2A9D8F",
                               alpha=0.75, ax=ax)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#2E86AB", edgecolor="white", label="Visual code"),
            Patch(facecolor="#E63946", edgecolor="white", label="Audio code"),
        ], loc="upper right")
        ax.set_title(f"Stable cross-modal edges  (stability ≥ {thresh}, "
                     f"width ∝ stability, N={len(cross)})", fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "stable_bipartite.png", dpi=140, bbox_inches="tight")
        plt.close()

    print("\n" + "═" * 60)
    print("SUMMARY (universal Glasso pipeline)")
    print("═" * 60)
    print(f"  Effective codes  : {K_v_eff} visual + {K_a_eff} audio = {K}")
    print(f"  Time bins        : {T}")
    print(f"  α (CV)           : {alpha:.4f}")
    print(f"  Block length     : {block_len}")
    print(f"  Bootstraps       : {args.n_bootstrap}")
    print(f"  Stability thresh : {thresh}")
    print(f"  Stable edges     : total {len(edges)}   "
          f"(within-V {n_vis}, within-A {n_aud}, CROSS {n_cross})")
    print(f"\n  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
