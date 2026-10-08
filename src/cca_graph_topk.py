"""Clean top-K version of the CCA cross-modal graph."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import networkx as nx
import os
from pathlib import Path
from sklearn.cross_decomposition import CCA
import math

EMB = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/embedding_glasso"
OUT = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/cca_final_graph"

V_emb = np.load(EMB/"V_embeddings.npy")
A_emb = np.load(EMB/"A_embeddings.npy")
T, D = V_emb.shape
V = (V_emb - V_emb.mean(0))/(V_emb.std(0)+1e-9)
A = (A_emb - A_emb.mean(0))/(A_emb.std(0)+1e-9)

TOP_K_COMP = 12
cca = CCA(n_components=TOP_K_COMP, max_iter=1500, tol=1e-5)
cca.fit(V, A)
U, S = cca.transform(V, A)
corrs = np.array([np.corrcoef(U[:,i], S[:,i])[0,1] for i in range(TOP_K_COMP)])

# Effective cross-modal correlation matrix
M = np.zeros((D, D))
for c in range(TOP_K_COMP):
    M += corrs[c] * np.outer(cca.x_weights_[:,c], cca.y_weights_[:,c])

# ── Keep only top-N edges by |M| for a readable graph ─────────────────
TOP_N = 40
rows = []
for i in range(D):
    for j in range(D):
        rows.append((i, j, float(M[i, j])))
df = pd.DataFrame(rows, columns=["v","a","corr"])
df["abs"] = df["corr"].abs()
df = df.sort_values("abs", ascending=False).head(TOP_N).reset_index(drop=True)
df.rename(columns={"corr": "eff_corr"}, inplace=True)

used_v = sorted(df.v.unique())
used_a = sorted(df.a.unique())
print(f"Top {TOP_N} edges  |  visual dims used: {len(used_v)}  |  audio dims used: {len(used_a)}")
print(df[["v","a","eff_corr"]].to_string(index=False))

G = nx.Graph()
for i in used_v: G.add_node(f"V{i:02d}", mod="V")
for j in used_a: G.add_node(f"A{j:02d}", mod="A")
for _, r in df.iterrows():
    G.add_edge(f"V{int(r.v):02d}", f"A{int(r.a):02d}", w=r.eff_corr)

# Bipartite layout — two vertical columns, sorted by degree
v_deg = sorted(used_v, key=lambda i: -G.degree(f"V{i:02d}"))
a_deg = sorted(used_a, key=lambda i: -G.degree(f"A{i:02d}"))
pos = {}
n_v, n_a = len(v_deg), len(a_deg)
for i, v in enumerate(v_deg):
    pos[f"V{v:02d}"] = (0, -i * (max(n_v, n_a)/max(n_v,1)))
for i, a in enumerate(a_deg):
    pos[f"A{a:02d}"] = (2, -i * (max(n_v, n_a)/max(n_a,1)))

fig, ax = plt.subplots(figsize=(12, max(6, 0.35*max(n_v,n_a))))
colors = ["#2E86AB" if G.nodes[n]["mod"]=="V" else "#E63946" for n in G.nodes]
sizes  = [500 + 200*G.degree(n) for n in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes,
                       edgecolors="white", linewidths=1.5, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", ax=ax)

# Edges colored by sign, width by magnitude
vmax = df["abs"].max()
edges_pos = [(u,v) for u,v,d in G.edges(data=True) if d["w"]>0]
edges_neg = [(u,v) for u,v,d in G.edges(data=True) if d["w"]<0]
w_pos = [10*abs(G[u][v]["w"])/vmax for u,v in edges_pos]
w_neg = [10*abs(G[u][v]["w"])/vmax for u,v in edges_neg]
nx.draw_networkx_edges(G, pos, edgelist=edges_pos, width=w_pos,
                       edge_color="#F72585", alpha=0.75, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=edges_neg, width=w_neg,
                       edge_color="#3A0CA3", alpha=0.75, ax=ax, style="dashed")

# Annotate strongest edges
top5 = df.head(5)
for _, r in top5.iterrows():
    u = f"V{int(r.v):02d}"; v = f"A{int(r.a):02d}"
    mx = (pos[u][0]+pos[v][0])/2 + 0.02
    my = (pos[u][1]+pos[v][1])/2
    ax.text(mx, my, f"{r.eff_corr:+.3f}", fontsize=8, alpha=0.7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

ax.legend(handles=[
    Patch(facecolor="#2E86AB", edgecolor="white", label=f"Visual dim  ({n_v} used)"),
    Patch(facecolor="#E63946", edgecolor="white", label=f"Audio dim   ({n_a} used)"),
    Line2D([0],[0], color="#F72585", lw=3, label="positive effective corr"),
    Line2D([0],[0], color="#3A0CA3", lw=3, ls="--", label="negative effective corr"),
], loc="upper right", frameon=True)
ax.set_title(
    f"Cross-modal relationship — top {TOP_N} edges from CCA\n"
    f"(64-D quantised embeddings, top {TOP_K_COMP} components, mean r = {corrs.mean():+.2f})",
    fontsize=13)
ax.axis("off")
plt.tight_layout(); plt.savefig(OUT/"cross_modal_graph_top40.png", dpi=160,
                                 bbox_inches="tight")
plt.close()
print(f"\n✅ Saved: {OUT/'cross_modal_graph_top40.png'}")
