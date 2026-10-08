"""
v2 — Graphical Lasso on BINARY code-presence indicators.

Instead of count histograms (which are dominated by within-frame spatial
redundancy), we use a binary indicator per (time, code):

    X[t, k] = 1  if code k appears anywhere in the frame/window at time t
              0  otherwise

This isolates the semantic co-occurrence between visual and audio codes.
"""
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

TOK = Path(os.environ.get("MMVQ_ROOT", ".")) / "tokens"
OUT = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/glasso_v2"
OUT.mkdir(parents=True, exist_ok=True)

# Filtering
MIN_ACT_FRAC = 0.02      # code must fire in ≥ 2% of time bins
MAX_ACT_FRAC = 0.98      # AND in ≤ 98% (drop always-on background codes)
EDGE_TOL     = 0.02


def preprocess(hist, name):
    """Binary-ise and filter."""
    T = hist.shape[0]
    binary = (hist > 0).astype(np.float64)          # (T, K)
    frac = binary.mean(0)
    keep = (frac >= MIN_ACT_FRAC) & (frac <= MAX_ACT_FRAC)
    print(f"  {name}: kept {keep.sum()} / {len(keep)} codes "
          f"({MIN_ACT_FRAC*100:.0f}%–{MAX_ACT_FRAC*100:.0f}% of frames)")
    return binary[:, keep], keep


def zscore(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def main():
    print("── Loading token histograms ────────────────────────────────")
    v_hist = np.load(TOK / "visual_hist.npy")
    a_hist = np.load(TOK / "audio_hist.npy")
    print(f"  visual_hist: {v_hist.shape}")
    print(f"  audio_hist : {a_hist.shape}")

    print("\n── Binary-ising and filtering codes ────────────────────────")
    v_bin, v_keep = preprocess(v_hist, "visual")
    a_bin, a_keep = preprocess(a_hist, "audio")
    v_labels = [f"V{i:03d}" for i in np.where(v_keep)[0]]
    a_labels = [f"A{i:03d}" for i in np.where(a_keep)[0]]
    all_labels = v_labels + a_labels

    K_v_eff, K_a_eff = len(v_labels), len(a_labels)
    X = np.concatenate([v_bin, a_bin], axis=1)
    T, K = X.shape
    print(f"\n  joint binary matrix: T={T} × K={K} "
          f"({K_v_eff} visual + {K_a_eff} audio)")
    print(f"  T/K ratio = {T/K:.2f}")

    Xs = zscore(X)

    print("\n── Fitting GraphicalLassoCV ────────────────────────────────")
    alphas = np.logspace(-2.5, -0.5, 10)
    try:
        model = GraphicalLassoCV(alphas=alphas, cv=3, max_iter=200, n_jobs=-1)
        model.fit(Xs)
        alpha_chosen = model.alpha_
    except Exception as e:
        print(f"  CV failed ({e}); falling back to α=0.05")
        model = GraphicalLasso(alpha=0.05, max_iter=200)
        model.fit(Xs)
        alpha_chosen = 0.05

    prec = model.precision_
    D = np.sqrt(np.diag(prec))
    partial = -prec / np.outer(D, D)
    np.fill_diagonal(partial, 1.0)
    partial_df = pd.DataFrame(partial, index=all_labels, columns=all_labels)

    n_nz = int((np.abs(prec) > 1e-6).sum() - K)
    print(f"  chosen α = {alpha_chosen:.4f}")
    print(f"  non-zero off-diagonal precision entries: {n_nz} / {K*(K-1)}")

    np.save(OUT / "precision_matrix.npy", prec)
    np.save(OUT / "partial_correlation.npy", partial)
    partial_df.to_csv(OUT / "partial_correlation.csv")

    # ── Edges ─────────────────────────────────────────────────────────
    print("\n── Extracting edges ────────────────────────────────────────")
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            if abs(partial[i, j]) < EDGE_TOL:
                continue
            src, dst = all_labels[i], all_labels[j]
            typ = "cross" if src[0] != dst[0] else ("visual" if src[0] == "V" else "audio")
            rows.append((src, dst, typ, partial[i, j]))
    edges = pd.DataFrame(rows, columns=["a", "b", "type", "partial_corr"])
    edges["abs_pc"] = edges["partial_corr"].abs()
    edges = edges.sort_values("abs_pc", ascending=False)

    cross_edges = edges[edges.type == "cross"].copy()
    visual_edges = edges[edges.type == "visual"].copy()
    audio_edges  = edges[edges.type == "audio"].copy()

    print(f"  total significant edges (|pc| > {EDGE_TOL}): {len(edges)}")
    print(f"    within-visual : {len(visual_edges)}")
    print(f"    within-audio  : {len(audio_edges)}")
    print(f"    CROSS-MODAL   : {len(cross_edges)}  ← headline")

    cross_edges.drop(columns="abs_pc").to_csv(OUT/"cross_modal_edges.csv", index=False)
    visual_edges.drop(columns="abs_pc").to_csv(OUT/"within_visual_edges.csv", index=False)
    audio_edges.drop(columns="abs_pc").to_csv(OUT/"within_audio_edges.csv", index=False)

    if len(cross_edges):
        n = min(20, len(cross_edges))
        print(f"\n  top {n} cross-modal edges:")
        print(cross_edges.head(n).drop(columns="abs_pc").to_string(index=False))

    # ── Visualisations ─────────────────────────────────────────────────
    print("\n── Rendering visualisations ────────────────────────────────")

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.eye(K, dtype=bool)
    vmax = max(0.05, np.abs(partial[~mask]).max())
    sns.heatmap(partial_df, mask=mask, cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, ax=ax,
                xticklabels=False, yticklabels=False,
                cbar_kws={"label": "partial correlation"})
    ax.axhline(K_v_eff, color="black", lw=1.5)
    ax.axvline(K_v_eff, color="black", lw=1.5)
    ax.text(K_v_eff/2, -3, "Visual codes", ha="center", fontsize=12, fontweight="bold")
    ax.text(K_v_eff + K_a_eff/2, -3, "Audio codes", ha="center", fontsize=12, fontweight="bold")
    ax.text(-8, K_v_eff/2, "Visual", va="center", ha="center",
            fontsize=12, fontweight="bold", rotation=90)
    ax.text(-8, K_v_eff + K_a_eff/2, "Audio", va="center", ha="center",
            fontsize=12, fontweight="bold", rotation=90)
    plt.title(f"Cross-modal Graphical Lasso (binary presence)\n"
              f"α = {alpha_chosen:.4f}, T = {T} bins")
    plt.tight_layout()
    plt.savefig(OUT / "block_heatmap.png", dpi=140)
    plt.close()

    # Zoom into just the cross-modal block
    if K_v_eff > 0 and K_a_eff > 0:
        cross_block = partial_df.iloc[:K_v_eff, K_v_eff:]
        fig, ax = plt.subplots(figsize=(11, 9))
        cbm = max(0.05, np.abs(cross_block.values).max())
        sns.heatmap(cross_block, cmap="RdBu_r", center=0,
                    vmin=-cbm, vmax=cbm, ax=ax,
                    xticklabels=False, yticklabels=False,
                    cbar_kws={"label": "partial correlation"})
        ax.set_xlabel(f"Audio codes ({K_a_eff})", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Visual codes ({K_v_eff})", fontsize=12, fontweight="bold")
        plt.title(f"Cross-modal block: {len(cross_edges)} edges above |pc| > {EDGE_TOL}",
                  fontsize=13)
        plt.tight_layout()
        plt.savefig(OUT / "cross_block_zoom.png", dpi=140)
        plt.close()

    # Bipartite graph of cross edges
    if len(cross_edges):
        B = nx.Graph()
        v_nodes = sorted({n for n in set(cross_edges.a)|set(cross_edges.b) if n.startswith("V")})
        a_nodes = sorted({n for n in set(cross_edges.a)|set(cross_edges.b) if n.startswith("A")})
        for n in v_nodes: B.add_node(n, mod="V")
        for n in a_nodes: B.add_node(n, mod="A")
        for _, r in cross_edges.iterrows():
            B.add_edge(r.a, r.b, weight=r.partial_corr)

        pos = {n: (0, -i) for i, n in enumerate(v_nodes)}
        step = len(v_nodes) / max(1, len(a_nodes))
        for i, n in enumerate(a_nodes):
            pos[n] = (1, -i * step)

        fig, ax = plt.subplots(figsize=(11, max(6, min(24, 0.30*max(len(v_nodes),len(a_nodes))))))
        colors = ["#2E86AB" if B.nodes[n]["mod"] == "V" else "#E63946" for n in B.nodes()]
        nx.draw_networkx_nodes(B, pos, node_color=colors, node_size=380,
                               edgecolors="white", linewidths=1.5, ax=ax)
        nx.draw_networkx_labels(B, pos, font_size=7, font_color="white", ax=ax)
        ep = [(u,v) for u,v,d in B.edges(data=True) if d["weight"] > 0]
        en = [(u,v) for u,v,d in B.edges(data=True) if d["weight"] < 0]
        wp = [ B[u][v]["weight"]*25 for u,v in ep]
        wn = [-B[u][v]["weight"]*25 for u,v in en]
        nx.draw_networkx_edges(B, pos, edgelist=ep, width=wp,
                               edge_color="#2A9D8F", alpha=0.7, ax=ax)
        nx.draw_networkx_edges(B, pos, edgelist=en, width=wn,
                               edge_color="#B5651D", alpha=0.7, ax=ax, style="dashed")
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Patch(facecolor="#2E86AB", edgecolor="white", label="Visual code"),
            Patch(facecolor="#E63946", edgecolor="white", label="Audio code"),
            Line2D([0],[0], color="#2A9D8F", lw=2.5, label="positive"),
            Line2D([0],[0], color="#B5651D", lw=2.5, ls="--", label="negative"),
        ], loc="upper right", frameon=True)
        ax.set_title(f"Cross-modal edges (binary presence): {len(cross_edges)} pairs",
                     fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUT / "cross_bipartite.png", dpi=140, bbox_inches="tight")
        plt.close()

    print("\n" + "═" * 60)
    print("SUMMARY (binary presence indicators)")
    print("═" * 60)
    print(f"  Effective codes: {K_v_eff} visual + {K_a_eff} audio = {K}")
    print(f"  Time bins      : {T}")
    print(f"  α (CV)         : {alpha_chosen:.4f}")
    print(f"  Total edges    : {len(edges)}  "
          f"(within-V {len(visual_edges)}, within-A {len(audio_edges)}, "
          f"cross {len(cross_edges)})")
    print(f"\n  Saved to {OUT}/")


if __name__ == "__main__":
    main()
