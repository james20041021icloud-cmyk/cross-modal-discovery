"""
Second half: Graphical Lasso on the joint visual+audio token histograms
learnt by the two Soft-VQ-VAEs.

Input:
  tokens/visual_hist.npy   (T, K_v)   int/float counts per frame
  tokens/audio_hist.npy    (T, K_a)   int/float counts per 1-sec window
  tokens/timeline.csv      time index

Output (in results/glasso/):
  precision_matrix.npy         sparse precision estimate (K'×K')
  partial_correlation.npy      full partial-corr matrix
  cross_modal_edges.csv        significant visual↔audio pairs
  within_modal_edges.csv       within-modality pairs (sanity)
  block_heatmap.png            precision heatmap with V/A block dividers
  cross_bipartite.png          bipartite graph of cross-modal edges
"""
import argparse
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
TOK   = Path(os.environ.get("MMVQ_ROOT", ".")) / "tokens"
OUT   = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/glasso"
OUT.mkdir(parents=True, exist_ok=True)

MIN_ACTIVATIONS = 20      # drop codes that fire fewer than this many times total
MIN_VARIANCE    = 1e-6    # drop codes with essentially zero variance
CROSS_EDGE_TOL  = 0.02    # partial-correlation threshold for cross-modal edges


def preprocess(hist, min_act, min_var, name):
    """Drop dead / near-constant codes; return filtered hist + keep mask."""
    activations = (hist > 0).sum(0)                       # times this code fires
    variance    = hist.var(0)
    keep = (activations >= min_act) & (variance >= min_var)
    print(f"  {name}: kept {keep.sum()} / {len(keep)} codes "
          f"(dropped {(~keep).sum()} rare/constant)")
    return hist[:, keep], keep


def zscore(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample_every", type=int, default=1,
                    help="use every Nth time bin to reduce autocorrelation")
    args = ap.parse_args()

    print("── Loading token histograms ────────────────────────────────")
    v_hist = np.load(TOK / "visual_hist.npy")   # (T, K_v)
    a_hist = np.load(TOK / "audio_hist.npy")    # (T, K_a)
    print(f"  visual_hist: {v_hist.shape}")
    print(f"  audio_hist : {a_hist.shape}")

    if args.subsample_every > 1:
        v_hist = v_hist[::args.subsample_every]
        a_hist = a_hist[::args.subsample_every]
        print(f"  subsampled → visual {v_hist.shape}, audio {a_hist.shape}")

    print("\n── Filtering rare / constant codes ─────────────────────────")
    v_hist, v_keep = preprocess(v_hist, MIN_ACTIVATIONS, MIN_VARIANCE, "visual")
    a_hist, a_keep = preprocess(a_hist, MIN_ACTIVATIONS, MIN_VARIANCE, "audio")
    v_labels = [f"V{i:03d}" for i in np.where(v_keep)[0]]
    a_labels = [f"A{i:03d}" for i in np.where(a_keep)[0]]
    all_labels = v_labels + a_labels

    K_v_eff = len(v_labels)
    K_a_eff = len(a_labels)

    # Joint matrix
    X = np.concatenate([v_hist, a_hist], axis=1).astype(np.float64)
    T, K = X.shape
    print(f"\n  joint matrix: T={T} time bins × K={K} features "
          f"({K_v_eff} visual + {K_a_eff} audio)")
    print(f"  T/K ratio = {T/K:.2f}  (need ≳ 1 for stable Glasso)")

    Xs = zscore(X)

    print("\n── Fitting GraphicalLassoCV (5-fold CV picks α) ────────────")
    # For high-dim data GLassoCV can be slow — use small α grid, fewer folds
    alphas = np.logspace(-2.0, -0.3, 8)
    try:
        model = GraphicalLassoCV(alphas=alphas, cv=3, max_iter=200, n_jobs=-1)
        model.fit(Xs)
        alpha_chosen = model.alpha_
    except Exception as e:
        # Fall back to a fixed α if CV fails
        print(f"  CV failed ({e}); falling back to α=0.05")
        model = GraphicalLasso(alpha=0.05, max_iter=200)
        model.fit(Xs)
        alpha_chosen = 0.05

    prec = pd.DataFrame(model.precision_, index=all_labels, columns=all_labels)

    D = np.sqrt(np.diag(prec))
    partial = -model.precision_ / np.outer(D, D)
    np.fill_diagonal(partial, 1.0)
    partial = pd.DataFrame(partial, index=all_labels, columns=all_labels)

    n_nz = int((np.abs(model.precision_) > 1e-6).sum() - K)
    print(f"  chosen α = {alpha_chosen:.4f}")
    print(f"  non-zero off-diagonal precision entries: {n_nz} / {K*(K-1)}")

    # Save matrices
    np.save(OUT / "precision_matrix.npy", model.precision_)
    np.save(OUT / "partial_correlation.npy", partial.values)
    prec.to_csv(OUT / "precision_matrix.csv")
    partial.to_csv(OUT / "partial_correlation.csv")

    # ── Extract cross-modal & within-modal edges ──────────────────────
    print("\n── Extracting edges ────────────────────────────────────────")
    p = partial.values
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            if abs(p[i, j]) < CROSS_EDGE_TOL:
                continue
            src, dst = all_labels[i], all_labels[j]
            typ = "cross" if src[0] != dst[0] else ("visual" if src[0] == "V" else "audio")
            rows.append((src, dst, typ, p[i, j]))
    edges = pd.DataFrame(rows, columns=["a", "b", "type", "partial_corr"])
    edges["abs_pc"] = edges["partial_corr"].abs()
    edges = edges.sort_values("abs_pc", ascending=False)

    cross_edges = edges[edges.type == "cross"].copy()
    visual_edges = edges[edges.type == "visual"].copy()
    audio_edges  = edges[edges.type == "audio"].copy()

    print(f"  total significant edges (|pc| > {CROSS_EDGE_TOL}): {len(edges)}")
    print(f"    within-visual : {len(visual_edges)}")
    print(f"    within-audio  : {len(audio_edges)}")
    print(f"    CROSS-MODAL   : {len(cross_edges)}  ← headline")

    cross_edges.drop(columns="abs_pc").to_csv(OUT/"cross_modal_edges.csv", index=False)
    visual_edges.drop(columns="abs_pc").to_csv(OUT/"within_visual_edges.csv", index=False)
    audio_edges.drop(columns="abs_pc").to_csv(OUT/"within_audio_edges.csv", index=False)

    print("\n  top 15 cross-modal edges:")
    print(cross_edges.head(15).drop(columns="abs_pc").to_string(index=False))

    # ── Visualisations ─────────────────────────────────────────────────
    print("\n── Rendering visualisations ────────────────────────────────")

    # Block heatmap of partial correlations (masked diagonal)
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.eye(K, dtype=bool)
    vmax = np.abs(partial.values[~mask]).max()
    sns.heatmap(partial, mask=mask, cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, ax=ax,
                xticklabels=False, yticklabels=False,
                cbar_kws={"label": "partial correlation"})
    # Draw the V/A block boundary
    ax.axhline(K_v_eff, color="black", lw=1.5)
    ax.axvline(K_v_eff, color="black", lw=1.5)
    ax.text(K_v_eff / 2, -3, "Visual codes", ha="center", fontsize=12, fontweight="bold")
    ax.text(K_v_eff + K_a_eff / 2, -3, "Audio codes", ha="center", fontsize=12, fontweight="bold")
    ax.text(-8, K_v_eff / 2, "Visual", ha="center", va="center",
            fontsize=12, fontweight="bold", rotation=90)
    ax.text(-8, K_v_eff + K_a_eff / 2, "Audio", ha="center", va="center",
            fontsize=12, fontweight="bold", rotation=90)
    plt.title(f"Cross-modal Graphical Lasso partial correlations\n"
              f"(α = {alpha_chosen:.4f}, T = {T} time bins)")
    plt.tight_layout()
    plt.savefig(OUT / "block_heatmap.png", dpi=140)
    plt.close()

    # Bipartite graph of cross-modal edges only
    B = nx.Graph()
    v_nodes = set(cross_edges.a) | set(cross_edges.b)
    v_nodes = {n for n in v_nodes if n.startswith("V")}
    a_nodes = set(cross_edges.a) | set(cross_edges.b)
    a_nodes = {n for n in a_nodes if n.startswith("A")}
    for n in v_nodes: B.add_node(n, mod="V")
    for n in a_nodes: B.add_node(n, mod="A")
    for _, r in cross_edges.iterrows():
        B.add_edge(r.a, r.b, weight=r.partial_corr)

    # Layout: two vertical columns
    v_sorted = sorted(v_nodes)
    a_sorted = sorted(a_nodes)
    pos = {}
    for i, n in enumerate(v_sorted):
        pos[n] = (0, -i)
    for i, n in enumerate(a_sorted):
        pos[n] = (1, -i * (len(v_sorted)/max(1,len(a_sorted))))

    fig, ax = plt.subplots(figsize=(11, max(6, min(20, 0.28*max(len(v_sorted),len(a_sorted))))))
    node_colors = ["#2E86AB" if B.nodes[n]["mod"] == "V" else "#E63946" for n in B.nodes()]
    nx.draw_networkx_nodes(B, pos, node_color=node_colors, node_size=340,
                           edgecolors="white", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(B, pos, font_size=7, font_color="white", ax=ax)
    edges_pos = [(u,v) for u,v,d in B.edges(data=True) if d["weight"] > 0]
    edges_neg = [(u,v) for u,v,d in B.edges(data=True) if d["weight"] < 0]
    w_pos = [B[u][v]["weight"]*30 for u,v in edges_pos]
    w_neg = [-B[u][v]["weight"]*30 for u,v in edges_neg]
    nx.draw_networkx_edges(B, pos, edgelist=edges_pos, width=w_pos,
                           edge_color="#2A9D8F", alpha=0.7, ax=ax)
    nx.draw_networkx_edges(B, pos, edgelist=edges_neg, width=w_neg,
                           edge_color="#B5651D", alpha=0.7, ax=ax, style="dashed")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Patch(facecolor="#2E86AB", edgecolor="white", label="Visual code"),
        Patch(facecolor="#E63946", edgecolor="white", label="Audio code"),
        Line2D([0],[0], color="#2A9D8F", lw=2.5, label="positive"),
        Line2D([0],[0], color="#B5651D", lw=2.5, ls="--", label="negative"),
    ], loc="upper right", frameon=True)
    ax.set_title(f"Cross-modal edges from Graphical Lasso ({len(cross_edges)} pairs, |pc| > {CROSS_EDGE_TOL})",
                 fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "cross_bipartite.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ── Summary printout ───────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"  Effective codes analysed : {K_v_eff} visual + {K_a_eff} audio = {K}")
    print(f"  Time bins                : {T}")
    print(f"  Regulariser α (CV)       : {alpha_chosen:.4f}")
    print(f"  Total significant edges  : {len(edges)}")
    print(f"    within-visual  : {len(visual_edges)}")
    print(f"    within-audio   : {len(audio_edges)}")
    print(f"    cross-modal    : {len(cross_edges)}  ← ★ audio↔visual couplings")
    print(f"\n  All outputs saved to {OUT}/")


if __name__ == "__main__":
    main()
