"""
Final Glasso stage of the new pipeline:
   video → RVQ VQ-VAE codes → 64-D mean-quantised embeddings per bin → Glasso

Produces:
  - full 128×128 partial-correlation heatmap  (V/A blocks marked)
  - stability heatmap (from block bootstrap)
  - network graph with within-modal + cross-modal edges
  - CSV of top edges (sorted by |partial correlation|)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx
import os
from pathlib import Path
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
import warnings
warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────
EMB_DIR = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/embedding_glasso"
OUT     = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/glasso_final_graph"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load already-computed 64-D embeddings ───────────────────────────────
V_emb = np.load(EMB_DIR / "V_embeddings.npy")   # (T, 64)
A_emb = np.load(EMB_DIR / "A_embeddings.npy")   # (T, 64)
T, D = V_emb.shape
print(f"Loaded V_emb {V_emb.shape}, A_emb {A_emb.shape}")

# Standardise each dim
V = (V_emb - V_emb.mean(0)) / (V_emb.std(0) + 1e-9)
A = (A_emb - A_emb.mean(0)) / (A_emb.std(0) + 1e-9)
X = np.concatenate([V, A], axis=1)
K = X.shape[1]
K_v = V.shape[1]
labels = [f"V{i:02d}" for i in range(K_v)] + [f"A{i:02d}" for i in range(D)]

# ── Fit Glasso stepping down α until we hit smallest α that still converges
print("\nFinding smallest stable α (data is highly collinear)...")
alpha_lo = None
for a in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06]:
    try:
        m = GraphicalLasso(alpha=a, max_iter=300)
        m.fit(X)
        alpha_lo = a
        m_lo = m
        print(f"  α = {a:.3f} → CONVERGED  ({int((np.abs(m.precision_)>1e-6).sum()-K)} nonzero off-diag)")
    except Exception as e:
        print(f"  α = {a:.3f} → FAILED (ill-conditioned)")
        continue

if alpha_lo is None:
    raise SystemExit("Even α=0.5 failed — matrix too ill-conditioned.")

print(f"\n  Using α = {alpha_lo:.3f} for graph visualisation")

# For reference, also fit CV (may fail — that's OK)
alpha_cv = alpha_lo
try:
    m_cv = GraphicalLassoCV(alphas=[alpha_lo*0.8, alpha_lo, alpha_lo*1.3, alpha_lo*1.6],
                            cv=3, max_iter=200, n_jobs=-1)
    m_cv.fit(X)
    alpha_cv = m_cv.alpha_
    print(f"  α (CV)  = {alpha_cv:.4f}")
except Exception:
    pass

# Convert precision → partial correlations
def partial_from_precision(prec):
    D = np.sqrt(np.diag(prec))
    P = -prec / np.outer(D, D)
    np.fill_diagonal(P, 1.0)
    return P

partial_cv = partial_from_precision(m_cv.precision_)
partial_lo = partial_from_precision(m_lo.precision_)

# ── Block bootstrap stability at the looser α
print(f"\nRunning block-bootstrap stability at α={alpha_lo:.4f} ...")
b_len = 20
n_boot = 50
n_blocks = int(np.ceil(T / b_len))
n_pick = max(1, int(0.5 * n_blocks))
stab = np.zeros((K, K), dtype=np.float32)
rng = np.random.default_rng(0)
succ = 0
for b in range(n_boot):
    picks = rng.choice(n_blocks, size=n_pick, replace=True)
    rows = [np.arange(i*b_len, min((i+1)*b_len, T)) for i in picks]
    Xb = X[np.concatenate(rows)]
    try:
        mm = GraphicalLasso(alpha=alpha_lo, max_iter=120, tol=1e-3); mm.fit(Xb)
        nz = (np.abs(mm.precision_) > 1e-6).astype(np.float32)
        np.fill_diagonal(nz, 0); stab += nz; succ += 1
    except Exception:
        pass
stab /= max(succ, 1)
print(f"  bootstraps succeeded: {succ}/{n_boot}")

# ── Extract edges ────────────────────────────────────────────────────────
rows = []
for i in range(K):
    for j in range(i+1, K):
        p  = partial_lo[i, j]
        s  = stab[i, j]
        typ = "cross" if labels[i][0] != labels[j][0] else \
              ("visual" if labels[i][0] == "V" else "audio")
        rows.append((labels[i], labels[j], typ, float(p), float(s)))
edges = pd.DataFrame(rows, columns=["a","b","type","partial_corr","stability"])
edges["abs_pc"] = edges.partial_corr.abs()
edges = edges.sort_values("abs_pc", ascending=False)
edges.drop(columns="abs_pc").to_csv(OUT / "all_edges.csv", index=False)

print("\n── Top 20 within-VISUAL edges ────────────────────────")
print(edges[edges.type=="visual"].head(20).drop(columns="abs_pc").to_string(index=False))
print("\n── Top 20 within-AUDIO edges ─────────────────────────")
print(edges[edges.type=="audio"].head(20).drop(columns="abs_pc").to_string(index=False))
print("\n── Top 20 CROSS-modal edges (by |pc|) ────────────────")
print(edges[edges.type=="cross"].head(20).drop(columns="abs_pc").to_string(index=False))

# ── Heatmaps ─────────────────────────────────────────────────────────────
def block_heatmap(mat, title, cmap, path, vmin=None, vmax=None, center=None,
                  cbar_label="value"):
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.eye(K, dtype=bool)
    if vmax is None:
        vmax = float(np.abs(mat[~mask]).max()) or 0.1
        vmin = -vmax if center == 0 else 0
    sns.heatmap(mat, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                ax=ax, xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": cbar_label})
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, rotation=0,  fontsize=5)
    ax.axhline(K_v, color="black", lw=1.5)
    ax.axvline(K_v, color="black", lw=1.5)
    ax.text(K_v/2, -2, "Visual (64)",   ha="center", fontsize=12, fontweight="bold")
    ax.text(K_v + D/2, -2, "Audio (64)", ha="center", fontsize=12, fontweight="bold")
    ax.text(-3, K_v/2, "V", ha="center", va="center", fontsize=12, fontweight="bold", rotation=90)
    ax.text(-3, K_v + D/2, "A", ha="center", va="center", fontsize=12, fontweight="bold", rotation=90)
    plt.title(title, fontsize=12)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()

block_heatmap(partial_lo, f"Partial correlation graph  (α = {alpha_lo:.3f})",
              "RdBu_r", OUT / "partial_correlation_heatmap.png", center=0,
              cbar_label="partial correlation")
block_heatmap(stab, f"Bootstrap stability  (α = {alpha_lo:.3f}, {succ}/{n_boot} boots)",
              "magma", OUT / "stability_heatmap.png", vmin=0, vmax=1,
              cbar_label="stability")

# ── Network graph  ───────────────────────────────────────────────────────
# Draw thick edges for strong within-modal + highlight any cross-modal
G = nx.Graph()
for lbl in labels:
    mod = "V" if lbl[0] == "V" else "A"
    G.add_node(lbl, mod=mod)

PC_THRESH = 0.05  # keep edges with |pc| >= threshold for display
for _, r in edges.iterrows():
    if abs(r.partial_corr) >= PC_THRESH:
        G.add_edge(r.a, r.b, pc=r.partial_corr, stab=r.stability,
                   typ=r.type)

# Circular layout: visual on left semicircle, audio on right
pos = {}
v_nodes = sorted([n for n in G.nodes if n[0] == "V"])
a_nodes = sorted([n for n in G.nodes if n[0] == "A"])
import math
for i, n in enumerate(v_nodes):
    ang = math.pi + math.pi * (i / max(1, len(v_nodes) - 1))
    pos[n] = (math.cos(ang), math.sin(ang))
for i, n in enumerate(a_nodes):
    ang = -math.pi/2 + math.pi * (i / max(1, len(a_nodes) - 1))
    pos[n] = (math.cos(ang), math.sin(ang))

fig, ax = plt.subplots(figsize=(13, 11))
colors = ["#2E86AB" if G.nodes[n]["mod"] == "V" else "#E63946" for n in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=280,
                       edgecolors="white", linewidths=1.4, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=5.5, font_color="white", ax=ax)

# Draw within-modal edges (thin)
within_pos = [(u,v) for u,v,d in G.edges(data=True) if d["typ"] != "cross" and d["pc"] > 0]
within_neg = [(u,v) for u,v,d in G.edges(data=True) if d["typ"] != "cross" and d["pc"] < 0]
if within_pos:
    w = [3*abs(G[u][v]["pc"]) for u,v in within_pos]
    nx.draw_networkx_edges(G, pos, edgelist=within_pos, width=w,
                           edge_color="#2A9D8F", alpha=0.35, ax=ax)
if within_neg:
    w = [3*abs(G[u][v]["pc"]) for u,v in within_neg]
    nx.draw_networkx_edges(G, pos, edgelist=within_neg, width=w,
                           edge_color="#B5651D", alpha=0.35, ax=ax, style="dashed")

# Draw cross-modal edges (thick, on top)
cross_pos = [(u,v) for u,v,d in G.edges(data=True) if d["typ"] == "cross" and d["pc"] > 0]
cross_neg = [(u,v) for u,v,d in G.edges(data=True) if d["typ"] == "cross" and d["pc"] < 0]
if cross_pos:
    w = [6*abs(G[u][v]["pc"]) for u,v in cross_pos]
    nx.draw_networkx_edges(G, pos, edgelist=cross_pos, width=w,
                           edge_color="#F72585", alpha=0.9, ax=ax)
if cross_neg:
    w = [6*abs(G[u][v]["pc"]) for u,v in cross_neg]
    nx.draw_networkx_edges(G, pos, edgelist=cross_neg, width=w,
                           edge_color="#7209B7", alpha=0.9, ax=ax, style="dashed")

n_cross_edges = len(cross_pos) + len(cross_neg)
n_within_edges = len(within_pos) + len(within_neg)
ax.legend(handles=[
    Patch(facecolor="#2E86AB", edgecolor="white", label=f"Visual dim (64)"),
    Patch(facecolor="#E63946", edgecolor="white", label=f"Audio dim (64)"),
    Line2D([0],[0], color="#2A9D8F", lw=2, alpha=0.6, label="within-modal (+)"),
    Line2D([0],[0], color="#B5651D", lw=2, alpha=0.6, ls="--", label="within-modal (-)"),
    Line2D([0],[0], color="#F72585", lw=3, label="CROSS-modal (+)"),
    Line2D([0],[0], color="#7209B7", lw=3, ls="--", label="CROSS-modal (-)"),
], loc="upper right", frameon=True, facecolor="white", edgecolor="lightgray")
ax.set_title(f"Cross-modal Graphical Lasso on 64-D quantised embeddings  "
             f"|  α = {alpha_lo:.3f}, T = {T} bins\n"
             f"Displayed: |partial-corr| ≥ {PC_THRESH}  "
             f"(within-modal = {n_within_edges},  cross-modal = {n_cross_edges})",
             fontsize=11)
ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "network_graph.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Summary ──────────────────────────────────────────────────────────────
n_edges_by_type = edges[edges.abs_pc >= PC_THRESH].type.value_counts()
print("\n" + "═" * 60)
print("SUMMARY (Glasso on 64-D quantised embeddings)")
print("═" * 60)
print(f"  T = {T} bins  ×  K = {K} features (64 V + 64 A)")
print(f"  α (CV)               : {alpha_cv:.4f}")
print(f"  α (used for graph)   : {alpha_lo:.4f}")
print(f"  bootstraps           : {succ}/{n_boot}")
print(f"  Edges shown (|pc| ≥ {PC_THRESH}):")
for typ in ["visual","audio","cross"]:
    n = int(n_edges_by_type.get(typ, 0))
    print(f"    {typ:>6}: {n}")
print(f"\n  All outputs saved to {OUT}/")
