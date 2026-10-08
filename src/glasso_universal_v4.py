"""
Universal Glasso — v4. log(1+count) + top-N variance filter to escape
multicollinearity.

Keeps the same 3 upgrades (informative featurization, α by CV, block-bootstrap
stability). Adds a "top-N" filter that keeps only the most-informative
(highest-variance) codes per modality — restores T/K ratio comfortably above
10 so Glasso is well-posed even under bootstrap resampling.
"""
import argparse, warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso

warnings.filterwarnings("ignore")

TOK  = Path(os.environ.get("MMVQ_ROOT", ".")) / "tokens_rvq"
OUT  = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/glasso_universal_v4"
OUT.mkdir(parents=True, exist_ok=True)


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


def bootstrap_glasso(Xs, alpha, block_len, n_bootstrap=50, frac=0.5, seed=0):
    T, K = Xs.shape
    n_blocks_total = int(np.ceil(T / block_len))
    n_pick = max(1, int(np.round(n_blocks_total * frac)))
    rng = np.random.default_rng(seed)
    stability = np.zeros((K, K), dtype=np.float32)
    succ = 0
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
            succ += 1
        except Exception:
            pass
        if (b + 1) % 10 == 0:
            print(f"    bootstrap {b+1}/{n_bootstrap}  (successful: {succ})")
    return stability / max(succ, 1), succ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_dir",  default=str(TOK))
    ap.add_argument("--out_dir",     default=str(OUT))
    ap.add_argument("--top_visual",  type=int, default=100)
    ap.add_argument("--top_audio",   type=int, default=60)
    ap.add_argument("--n_bootstrap", type=int, default=50)
    ap.add_argument("--stability_thresh", type=float, default=0.60)
    args = ap.parse_args()

    tokens_dir = Path(args.tokens_dir)
    out_dir    = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("── Loading token histograms ────────────────────────────────")
    v_hist = np.load(tokens_dir / "visual_hist.npy").astype(np.float64)
    a_hist = np.load(tokens_dir / "audio_hist.npy").astype(np.float64)

    print("\n── Featurization: log(1 + count) ──────────────────────────")
    v_feat = np.log1p(v_hist)
    a_feat = np.log1p(a_hist)

    # ── Top-N most variable per modality
    v_var = v_feat.var(0)
    a_var = a_feat.var(0)
    v_top = np.argsort(v_var)[::-1][:args.top_visual]
    a_top = np.argsort(a_var)[::-1][:args.top_audio]
    v_feat = v_feat[:, v_top]
    a_feat = a_feat[:, a_top]
    per_level_v = 256  # each visual level has 256 codes
    per_level_a = 128
    v_labels = [f"V-L{i//per_level_v+1}-{i%per_level_v:03d}" for i in v_top]
    a_labels = [f"A-L{i//per_level_a+1}-{i%per_level_a:03d}" for i in a_top]
    all_labels = v_labels + a_labels
    print(f"  kept top {args.top_visual} visual codes (variance range "
          f"{v_var[v_top].min():.3f}..{v_var[v_top].max():.3f})")
    print(f"  kept top {args.top_audio}  audio  codes (variance range "
          f"{a_var[a_top].min():.3f}..{a_var[a_top].max():.3f})")

    X = np.concatenate([v_feat, a_feat], axis=1)
    T, K = X.shape
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    print(f"\n  joint: T={T} × K={K}   T/K = {T/K:.1f}")

    print("\n── Picking α by CV ───────────────────────────────────────")
    alphas = np.logspace(-2.0, -0.3, 8)
    try:
        m = GraphicalLassoCV(alphas=alphas, cv=3, max_iter=200, n_jobs=-1)
        m.fit(X)
        alpha = m.alpha_
    except Exception as e:
        print(f"  CV failed ({e}); falling back to α=0.05"); alpha = 0.05
    print(f"  α (locked) = {alpha:.4f}")

    print("\n── Block-bootstrap stability selection ────────────────────")
    b = summarize_block_length(X)
    print(f"  auto block length: {b}")
    print(f"  running {args.n_bootstrap} bootstraps ...")
    stability, n_ok = bootstrap_glasso(X, alpha=alpha, block_len=b,
                                        n_bootstrap=args.n_bootstrap, frac=0.5)
    print(f"  {n_ok}/{args.n_bootstrap} bootstraps succeeded")
    np.save(out_dir / "stability_matrix.npy", stability)

    K_v_eff = len(v_labels); K_a_eff = len(a_labels)
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

    cross_block = stability[:K_v_eff, K_v_eff:]
    print("\n  Cross-modal stability distribution:")
    for th in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]:
        print(f"    ≥ {th:.2f}: {int((cross_block >= th).sum())} edges")
    print(f"    max cross-modal stability: {cross_block.max():.3f}")

    if n_c > 0:
        print("\n  Top 20 cross-modal:")
        print(edges[edges.type == "cross"].head(20).to_string(index=False))

    # ── Plots
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cross_block, cmap="magma", vmin=0, vmax=1, ax=ax,
                xticklabels=a_labels, yticklabels=v_labels,
                cbar_kws={"label": "bootstrap stability"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=5)
    ax.set_xlabel(f"Audio codes (top {K_a_eff})", fontweight="bold")
    ax.set_ylabel(f"Visual codes (top {K_v_eff})", fontweight="bold")
    plt.title(f"v4: log(1+count) + top-N filter, α={alpha:.3f}, "
              f"{n_ok} bootstraps", fontsize=12)
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
        nx.draw_networkx_nodes(B, pos, node_color=colors, node_size=650,
                               edgecolors="white", linewidths=1.5, ax=ax)
        nx.draw_networkx_labels(B, pos, font_size=7, font_color="white", ax=ax)
        widths = [3 * B[u][v]["stability"] for u, v in B.edges()]
        nx.draw_networkx_edges(B, pos, width=widths, edge_color="#2A9D8F",
                               alpha=0.75, ax=ax)
        ax.set_title(f"Stable cross-modal edges (v4)  "
                     f"stability ≥ {thresh}, N={len(cross)}", fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "stable_bipartite.png", dpi=140, bbox_inches="tight")
        plt.close()

    print(f"\n✅ Saved to {out_dir}/")


if __name__ == "__main__":
    main()
