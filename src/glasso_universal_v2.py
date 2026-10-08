"""
Universal Glasso — v2. Same 3 upgrades as universal v1, but with
FRACTION-OF-POSITIONS featurization instead of binary presence.

X[t, k] = fraction of latent positions in bin t that were assigned to code k.
Real-valued in [0, 1] — preserves "how dominant is this pattern" magnitude
that binary presence collapsed to 1.
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
OUT  = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/glasso_universal_v2"
OUT.mkdir(parents=True, exist_ok=True)


def nonparanormal(X, eps=1e-3):
    """Rank-normalise each column to Gaussian marginal."""
    T, K = X.shape
    Xt = np.empty_like(X, dtype=np.float64)
    for k in range(K):
        ranks = pd.Series(X[:, k]).rank(method="average").values
        p = np.clip((ranks - 0.5) / T, eps, 1 - eps)
        Xt[:, k] = norm.ppf(p)
    return Xt


def acf_block_length(x, max_lag=30, thresh=0.1):
    x = np.asarray(x, dtype=float) - float(x.mean())
    v = x.var()
    if v == 0: return 1
    for lag in range(1, max_lag + 1):
        rho = (x[:-lag] * x[lag:]).mean() / v
        if abs(rho) < thresh:
            return lag
    return max_lag


def summarize_block_length(X, max_lag=30):
    bs = [acf_block_length(X[:, k], max_lag=max_lag)
          for k in range(min(50, X.shape[1]))]
    return int(np.median(bs))


def to_fraction(hist, n_positions_per_bin):
    """Convert count histogram to fraction of positions in [0, 1]."""
    return hist.astype(np.float64) / n_positions_per_bin


def bootstrap_glasso(Xs, alpha, block_len, n_bootstrap=50, frac=0.5, seed=0):
    T, K = Xs.shape
    n_blocks_total = int(np.ceil(T / block_len))
    n_pick = max(1, int(np.round(n_blocks_total * frac)))
    rng = np.random.default_rng(seed)
    stability = np.zeros((K, K), dtype=np.float32)

    for b in range(n_bootstrap):
        picks = rng.choice(n_blocks_total, size=n_pick, replace=True)
        rows = [np.arange(i * block_len, min((i + 1) * block_len, T)) for i in picks]
        Xb = Xs[np.concatenate(rows)]
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
    return stability / n_bootstrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_dir", default=str(TOK))
    ap.add_argument("--out_dir",    default=str(OUT))
    # 32×32 = 1024 visual latent positions per frame, per level;
    # concatenated hist has 2 levels → 2048 total per bin.
    ap.add_argument("--visual_positions_per_bin", type=int, default=1024*2)
    # 1000 audio latent positions per 1-sec window, per level; 2 levels → 2000.
    ap.add_argument("--audio_positions_per_bin",  type=int, default=1000*2)
    ap.add_argument("--min_variance",     type=float, default=1e-8)
    ap.add_argument("--n_bootstrap",      type=int, default=50)
    ap.add_argument("--stability_thresh", type=float, default=0.60)
    args = ap.parse_args()

    tokens_dir = Path(args.tokens_dir)
    out_dir    = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("── Loading token histograms ────────────────────────────────")
    v_hist = np.load(tokens_dir / "visual_hist.npy")
    a_hist = np.load(tokens_dir / "audio_hist.npy")
    print(f"  visual_hist: {v_hist.shape}  row-sum sample = {v_hist[0].sum():.0f}")
    print(f"  audio_hist : {a_hist.shape}  row-sum sample = {a_hist[0].sum():.0f}")

    print("\n── Converting to fraction-of-positions ────────────────────")
    v_frac = to_fraction(v_hist, args.visual_positions_per_bin)
    a_frac = to_fraction(a_hist, args.audio_positions_per_bin)
    print(f"  visual_frac: range [{v_frac.min():.4f}, {v_frac.max():.4f}], "
          f"mean {v_frac.mean():.4f}")
    print(f"  audio_frac : range [{a_frac.min():.4f}, {a_frac.max():.4f}], "
          f"mean {a_frac.mean():.4f}")

    # Drop degenerate columns
    v_var = v_frac.var(0);  a_var = a_frac.var(0)
    v_keep = v_var > args.min_variance
    a_keep = a_var > args.min_variance
    v_frac = v_frac[:, v_keep];  a_frac = a_frac[:, a_keep]
    v_labels = [f"V{i:04d}" for i in np.where(v_keep)[0]]
    a_labels = [f"A{i:04d}" for i in np.where(a_keep)[0]]
    all_labels = v_labels + a_labels
    K_v_eff, K_a_eff = v_frac.shape[1], a_frac.shape[1]
    print(f"  visual: kept {K_v_eff} / {len(v_keep)}  (variance > {args.min_variance})")
    print(f"  audio : kept {K_a_eff} / {len(a_keep)}")

    X = np.concatenate([v_frac, a_frac], axis=1)
    T, K = X.shape
    print(f"\n  joint: T={T} × K={K}   (T/K = {T/K:.2f})")

    print("\n── Nonparanormal transform ────────────────────────────────")
    Xt = nonparanormal(X)
    print(f"  col-mean ≈ {Xt.mean():.3f}, col-std ≈ {Xt.std():.3f}")

    print("\n── Picking α by CV ───────────────────────────────────────")
    alphas = np.logspace(-2.5, -0.5, 8)
    try:
        m = GraphicalLassoCV(alphas=alphas, cv=3, max_iter=150, n_jobs=-1)
        m.fit(Xt)
        alpha = m.alpha_
    except Exception as e:
        print(f"  CV failed ({e}); using α=0.1"); alpha = 0.1
    print(f"  α (locked) = {alpha:.4f}")

    print("\n── Block-bootstrap stability selection ────────────────────")
    b = summarize_block_length(Xt)
    print(f"  auto block length: {b}")
    print(f"  running {args.n_bootstrap} bootstraps ...")
    stability = bootstrap_glasso(Xt, alpha=alpha, block_len=b,
                                 n_bootstrap=args.n_bootstrap, frac=0.5)
    np.save(out_dir / "stability_matrix.npy", stability)

    # Extract edges
    thresh = args.stability_thresh
    print(f"\n── Edges surviving stability ≥ {thresh} ─────────────────")
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            if stability[i, j] >= thresh:
                src, dst = all_labels[i], all_labels[j]
                typ = "cross" if src[0] != dst[0] else ("visual" if src[0] == "V" else "audio")
                rows.append((src, dst, typ, float(stability[i, j])))
    edges = pd.DataFrame(rows, columns=["a", "b", "type", "stability"])
    edges = edges.sort_values("stability", ascending=False)
    edges.to_csv(out_dir / "stable_edges.csv", index=False)
    n_v = (edges.type == "visual").sum()
    n_a = (edges.type == "audio").sum()
    n_c = (edges.type == "cross").sum()
    print(f"  within-visual : {n_v}")
    print(f"  within-audio  : {n_a}")
    print(f"  CROSS-MODAL   : {n_c}  ← headline")

    # Report cross-modal breakdown at multiple thresholds so we can see
    # if the story changed
    cross_block = stability[:K_v_eff, K_v_eff:]
    print("\n  Cross-modal stability distribution:")
    for th in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        n = int((cross_block >= th).sum())
        print(f"    ≥ {th:.2f}: {n} edges")
    print(f"    max cross-modal stability: {cross_block.max():.3f}")

    if n_c > 0:
        print("\n  Top 20 cross-modal:")
        print(edges[edges.type == "cross"].head(20).to_string(index=False))

    # ── Plots ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cross_block, cmap="magma", vmin=0, vmax=1, ax=ax,
                xticklabels=False, yticklabels=False,
                cbar_kws={"label": "bootstrap stability"})
    ax.set_xlabel(f"Audio codes ({K_a_eff})", fontweight="bold")
    ax.set_ylabel(f"Visual codes ({K_v_eff})", fontweight="bold")
    plt.title(f"Cross-modal stability, fraction-of-positions featurization "
              f"(α={alpha:.3f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "cross_block_stability.png", dpi=140)
    plt.close()

    cross = edges[edges.type == "cross"].copy()
    if len(cross):
        B = nx.Graph()
        v_nodes = sorted({n for n in set(cross.a) | set(cross.b) if n.startswith("V")})
        a_nodes = sorted({n for n in set(cross.a) | set(cross.b) if n.startswith("A")})
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
        widths = [3 * B[u][v]["stability"] for u, v in B.edges()]
        nx.draw_networkx_edges(B, pos, width=widths, edge_color="#2A9D8F",
                               alpha=0.75, ax=ax)
        ax.set_title(f"Stable cross-modal edges (fraction-of-positions)  "
                     f"stability ≥ {thresh}, N={len(cross)}", fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "stable_bipartite.png", dpi=140, bbox_inches="tight")
        plt.close()

    print(f"\n✅ Saved to {out_dir}/")


if __name__ == "__main__":
    main()
