"""
Final relationship graph via CCA on the 64-D quantised embeddings.
Because Glasso's partial-correlation framework can't capture the multi-way
cross-modal structure in this data, we build the relationship graph from
canonical correlation analysis instead:

  CC_c has visual loading u_c ∈ R^64, audio loading v_c ∈ R^64, correlation r_c.
  Effective cross-modal correlation between V_i and A_j is
        M[i, j] = Σ_c  u_c[i] · v_c[j] · r_c
  reflecting the total signal shared through the top c components.
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
from sklearn.cross_decomposition import CCA
import warnings, math
warnings.filterwarnings("ignore")

EMB = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/embedding_glasso"
OUT = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/cca_final_graph"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load 64-D embeddings ─────────────────────────────────────────────────
V_emb = np.load(EMB / "V_embeddings.npy")
A_emb = np.load(EMB / "A_embeddings.npy")
T, D = V_emb.shape

V = (V_emb - V_emb.mean(0)) / (V_emb.std(0) + 1e-9)
A = (A_emb - A_emb.mean(0)) / (A_emb.std(0) + 1e-9)
print(f"V {V.shape}, A {A.shape}")

# ── Full CCA (all 64 components) ─────────────────────────────────────────
N_COMP = 64
print(f"Fitting CCA with {N_COMP} components ...")
cca = CCA(n_components=N_COMP, max_iter=1500, tol=1e-5)
cca.fit(V, A)
U, S = cca.transform(V, A)                       # (T, N_COMP) each
corrs = np.array([np.corrcoef(U[:, i], S[:, i])[0, 1] for i in range(N_COMP)])
print(f"Canonical correlations: [{corrs.min():+.3f}, {corrs.max():+.3f}], "
      f"mean {corrs.mean():+.3f}")

x_loads = cca.x_weights_     # (64, N_COMP) — visual weights per component
y_loads = cca.y_weights_     # (64, N_COMP) — audio  weights per component

# ── Bar chart of canonical correlations ──────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
bars = ax.bar(range(1, N_COMP + 1), corrs, color="#065A82")
# highlight the top 12
for i in range(12):
    bars[i].set_color("#06D6A0")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("canonical component index")
ax.set_ylabel("canonical correlation r")
ax.set_title(f"CCA canonical correlations between visual and audio "
             f"64-D quantised embeddings (T={T})")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "cca_correlations.png", dpi=140); plt.close()

# ── Effective cross-modal correlation matrix M[i, j] ─────────────────────
# Only use top-k components (bigger r = more signal). Try top 12.
TOP_K = 12
M = np.zeros((D, D))
for c in range(TOP_K):
    M += corrs[c] * np.outer(x_loads[:, c], y_loads[:, c])
print(f"Effective correlation matrix M: shape {M.shape}, "
      f"range [{M.min():+.3f}, {M.max():+.3f}]")

# ── Heatmap of effective correlation ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
vmax = float(np.abs(M).max())
sns.heatmap(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, center=0, ax=ax,
            xticklabels=[f"A{i:02d}" for i in range(D)],
            yticklabels=[f"V{i:02d}" for i in range(D)],
            cbar_kws={"label": "effective cross-modal correlation"})
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=5)
ax.set_xlabel("Audio dims (64)",  fontweight="bold")
ax.set_ylabel("Visual dims (64)", fontweight="bold")
plt.title(f"Effective cross-modal correlation (sum of top {TOP_K} CCA components)\n"
          f"max |corr| = {vmax:.3f}", fontsize=12)
plt.tight_layout(); plt.savefig(OUT / "effective_cross_correlation.png", dpi=140)
plt.close()

# ── Top edges by |M[i,j]| ───────────────────────────────────────────────
rows = []
for i in range(D):
    for j in range(D):
        rows.append((f"V{i:02d}", f"A{j:02d}", float(M[i, j])))
edges = pd.DataFrame(rows, columns=["v_dim","a_dim","effective_corr"])
edges["abs"] = edges.effective_corr.abs()
edges = edges.sort_values("abs", ascending=False)
edges.drop(columns="abs").to_csv(OUT / "cross_modal_edges.csv", index=False)
print("\nTop 20 cross-modal edges by |effective corr|:")
print(edges.drop(columns="abs").head(20).to_string(index=False))

# ── Bipartite network graph (edges = strongest |effective corr|) ────────
THRESH_FRAC = 0.15    # keep edges > 15% of max
threshold = THRESH_FRAC * vmax
kept = edges[edges["abs"] >= threshold].copy()
print(f"\n{len(kept)} edges pass |corr| >= {threshold:.3f}")

G = nx.Graph()
v_nodes = [f"V{i:02d}" for i in range(D)]
a_nodes = [f"A{i:02d}" for i in range(D)]
for n in v_nodes: G.add_node(n, mod="V")
for n in a_nodes: G.add_node(n, mod="A")
for _, r in kept.iterrows():
    G.add_edge(r.v_dim, r.a_dim, weight=r.effective_corr)

# Semicircle layout
pos = {}
for i, n in enumerate(v_nodes):
    ang = math.pi + math.pi * (i / max(1, len(v_nodes) - 1))
    pos[n] = (math.cos(ang) * 3, math.sin(ang) * 3)
for i, n in enumerate(a_nodes):
    ang = -math.pi/2 + math.pi * (i / max(1, len(a_nodes) - 1))
    pos[n] = (math.cos(ang) * 3, math.sin(ang) * 3)

fig, ax = plt.subplots(figsize=(14, 12))
colors = ["#2E86AB" if G.nodes[n]["mod"] == "V" else "#E63946" for n in G.nodes]
sizes  = [400 if G.degree(n) > 0 else 60 for n in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes,
                       edgecolors="white", linewidths=1.4, ax=ax)
label_nodes = {n: n for n in G.nodes if G.degree(n) > 0}
nx.draw_networkx_labels(G, pos, labels=label_nodes,
                        font_size=7, font_color="white", ax=ax)
edges_pos = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] > 0]
edges_neg = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 0]
if edges_pos:
    w = [8 * abs(G[u][v]["weight"]) / vmax for u, v in edges_pos]
    nx.draw_networkx_edges(G, pos, edgelist=edges_pos, width=w,
                           edge_color="#F72585", alpha=0.75, ax=ax)
if edges_neg:
    w = [8 * abs(G[u][v]["weight"]) / vmax for u, v in edges_neg]
    nx.draw_networkx_edges(G, pos, edgelist=edges_neg, width=w,
                           edge_color="#3A0CA3", alpha=0.75, ax=ax, style="dashed")

ax.legend(handles=[
    Patch(facecolor="#2E86AB", edgecolor="white", label=f"Visual dim (64)"),
    Patch(facecolor="#E63946", edgecolor="white", label=f"Audio  dim (64)"),
    Line2D([0], [0], color="#F72585", lw=3, label="cross-modal (+)"),
    Line2D([0], [0], color="#3A0CA3", lw=3, ls="--", label="cross-modal (−)"),
], loc="upper left", frameon=True)
ax.set_title(
    f"Cross-modal relationship graph — top {TOP_K} CCA components\n"
    f"Edges = effective correlation ≥ {THRESH_FRAC*100:.0f}% of max ({threshold:.3f})  "
    f"| max |corr| = {vmax:.3f}  | CCA r₁..₁₂ = {corrs[:12].mean():+.2f} avg",
    fontsize=13)
ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "cross_modal_graph.png", dpi=160,
                                 bbox_inches="tight")
plt.close()

# ── Top 4 CC scatter plots ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for k, ax in enumerate(axes.flat):
    ax.scatter(U[:, k], S[:, k], s=4, alpha=0.4, color="#065A82")
    ax.set_xlabel(f"Visual CC{k+1}")
    ax.set_ylabel(f"Audio  CC{k+1}")
    ax.set_title(f"CC{k+1}:  r = {corrs[k]:+.3f}")
    ax.grid(alpha=0.3)
plt.suptitle("Top 4 canonical components (each dot = one 0.25-s bin)",
             fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "cca_top4_scatter.png", dpi=140)
plt.close()

# ── Loadings heatmap ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
im0 = axes[0].imshow(x_loads[:, :12], cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                     aspect="auto")
axes[0].set_xlabel("Canonical component"); axes[0].set_ylabel("Visual dim")
axes[0].set_xticks(range(12)); axes[0].set_xticklabels([f"CC{i+1}" for i in range(12)],
                                                        rotation=90)
axes[0].set_title("Visual loadings on top 12 CCA components")
plt.colorbar(im0, ax=axes[0], label="loading")

im1 = axes[1].imshow(y_loads[:, :12], cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                     aspect="auto")
axes[1].set_xlabel("Canonical component"); axes[1].set_ylabel("Audio dim")
axes[1].set_xticks(range(12)); axes[1].set_xticklabels([f"CC{i+1}" for i in range(12)],
                                                        rotation=90)
axes[1].set_title("Audio loadings on top 12 CCA components")
plt.colorbar(im1, ax=axes[1], label="loading")
plt.tight_layout(); plt.savefig(OUT / "cca_loadings.png", dpi=140)
plt.close()

# ── Summary printout ────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("SUMMARY — CCA-based cross-modal relationship")
print("═" * 60)
print(f"  T = {T},  D = {D} per modality")
print(f"  Top 12 canonical correlations: "
      f"[{corrs[0]:+.3f}, {corrs[11]:+.3f}], mean {corrs[:12].mean():+.3f}")
print(f"  Effective cross-corr matrix max |corr| = {vmax:.3f}")
print(f"  Edges kept in graph: {len(kept)}  (|corr| ≥ {threshold:.3f})")
print(f"\n  All outputs in {OUT}/")
