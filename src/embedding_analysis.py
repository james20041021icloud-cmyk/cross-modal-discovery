"""
Path B: use the 64-D codebook EMBEDDINGS, not the discrete indices.

For each time bin t:
  visual_emb[t] = mean over spatial positions of the SUM (L1+L2) codebook
                  vectors selected for that position   →  (64,)
  audio_emb [t] = same, over the 1000 audio latent positions              →  (64,)

We then have a joint (T, 128) matrix of CONTINUOUS features. On this we run:
  1) canonical correlation analysis (CCA) — the natural cross-modal method
  2) Graphical Lasso on the 128-D features — sparse conditional dependence graph
     between the 64 visual and 64 audio dimensions

Because the features are continuous and low-dimensional, both estimators become
well-posed and interpretable.
"""
import argparse, warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.cross_decomposition import CCA
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso

from vqvae_rvq_models import ImageRVQVAE, AudioRVQVAE

warnings.filterwarnings("ignore")

TOK  = Path(os.environ.get("MMVQ_ROOT", ".")) / "tokens_rvq"
OUT  = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/embedding_glasso"
OUT.mkdir(parents=True, exist_ok=True)

V_CKPT = Path(os.environ.get("MMVQ_ROOT", ".")) / "models_rvq/visual/image_rvq.pt"
A_CKPT = Path(os.environ.get("MMVQ_ROOT", ".")) / "models_rvq/audio/audio_rvq.pt"


def load_codebooks(ckpt_path, kind):
    """Return list of (K, D) codebook tensors, one per RVQ level."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    if kind == "image":
        m = ImageRVQVAE(3, a["hidden"], a["latent"],
                        a["num_levels"], a["codes_per_level"])
    else:
        m = AudioRVQVAE(1, a["hidden"], a["latent"],
                        a["num_levels"], a["codes_per_level"])
    m.load_state_dict(ckpt["model"])
    codebooks = [level.embedding.detach().cpu().numpy()
                 for level in m.vq.levels]                # list of (K, D)
    return codebooks


def tokens_to_embedding(tokens_list, codebooks):
    """
    tokens_list: list per RVQ level, each (T, *spatial_shape) integer indices.
    codebooks  : list per RVQ level, each (K, D)  codebook embeddings.

    Returns a (T, D) array — the mean-over-positions of the SUM of per-level
    codebook vectors at each spatial position.
    """
    T = tokens_list[0].shape[0]
    D = codebooks[0].shape[1]
    # For each level, look up (T, *spatial) → (T, *spatial, D)
    per_level_z = []
    for tok, cb in zip(tokens_list, codebooks):
        # Vectorised gather
        looked = cb[tok]                              # (T, *spatial, D)
        per_level_z.append(looked)
    # RVQ: sum the levels
    z = np.sum(per_level_z, axis=0)                    # (T, *spatial, D)
    # Mean over all spatial dims
    z_mean = z.reshape(T, -1, D).mean(axis=1)          # (T, D)
    return z_mean


def cca_analysis(V, A, n_components=8):
    """Fit CCA on (T, D_v) visual embeddings vs (T, D_a) audio embeddings."""
    cca = CCA(n_components=n_components, max_iter=500)
    cca.fit(V, A)
    U, S = cca.transform(V, A)
    # Canonical correlations
    corrs = [np.corrcoef(U[:, i], S[:, i])[0, 1] for i in range(n_components)]
    return cca, U, S, np.array(corrs)


def glasso_analysis(X, out_dir, tag=""):
    """Glasso on standardised joint (T, K) matrix.  Includes block bootstrap."""
    T, K = X.shape
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    print(f"  Glasso: T={T}, K={K}, T/K = {T/K:.1f}")
    alphas = np.logspace(-2.5, -0.3, 8)
    try:
        m = GraphicalLassoCV(alphas=alphas, cv=3, max_iter=200, n_jobs=-1)
        m.fit(X)
        alpha = m.alpha_
    except Exception as e:
        print(f"    CV failed: {e}"); alpha = 0.05
    print(f"  α = {alpha:.4f}")

    # Bootstrap
    b_len = 20
    n_bootstrap = 50
    n_blocks = int(np.ceil(T / b_len))
    n_pick = max(1, int(0.5 * n_blocks))
    stab = np.zeros((K, K), dtype=np.float32)
    rng = np.random.default_rng(0)
    succ = 0
    for b in range(n_bootstrap):
        picks = rng.choice(n_blocks, size=n_pick, replace=True)
        rows = [np.arange(i*b_len, min((i+1)*b_len, T)) for i in picks]
        Xb = X[np.concatenate(rows)]
        try:
            m = GraphicalLasso(alpha=alpha, max_iter=100, tol=1e-3); m.fit(Xb)
            nz = (np.abs(m.precision_) > 1e-6).astype(np.float32)
            np.fill_diagonal(nz, 0); stab += nz; succ += 1
        except Exception:
            pass
    stab /= max(succ, 1)
    print(f"  bootstraps succeeded: {succ}/{n_bootstrap}")

    # Fit final Glasso
    m_final = GraphicalLasso(alpha=alpha, max_iter=200); m_final.fit(X)
    prec = m_final.precision_
    D = np.sqrt(np.diag(prec))
    partial = -prec / np.outer(D, D)
    np.fill_diagonal(partial, 1.0)
    return partial, stab, alpha


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    print("── Loading tokens + codebooks ──────────────────────────────")
    v_tokens = [np.load(TOK / f"visual_tokens_L{L}.npy") for L in [1, 2]]
    a_tokens = [np.load(TOK / f"audio_tokens_L{L}.npy") for L in [1, 2]]
    T = min(v_tokens[0].shape[0], a_tokens[0].shape[0])
    v_tokens = [x[:T] for x in v_tokens]
    a_tokens = [x[:T] for x in a_tokens]
    print(f"  T = {T}")
    print(f"  visual tokens: L1 {v_tokens[0].shape}, L2 {v_tokens[1].shape}")
    print(f"  audio  tokens: L1 {a_tokens[0].shape}, L2 {a_tokens[1].shape}")

    v_codebooks = load_codebooks(V_CKPT, "image")
    a_codebooks = load_codebooks(A_CKPT, "audio")
    print(f"  visual codebooks: L1 {v_codebooks[0].shape}, L2 {v_codebooks[1].shape}")
    print(f"  audio  codebooks: L1 {a_codebooks[0].shape}, L2 {a_codebooks[1].shape}")

    print("\n── Computing mean-quantized embeddings per time bin ────────")
    V_emb = tokens_to_embedding(v_tokens, v_codebooks)    # (T, 64)
    A_emb = tokens_to_embedding(a_tokens, a_codebooks)    # (T, 64)
    print(f"  V_emb: {V_emb.shape},  range [{V_emb.min():+.3f}, {V_emb.max():+.3f}]")
    print(f"  A_emb: {A_emb.shape},  range [{A_emb.min():+.3f}, {A_emb.max():+.3f}]")

    np.save(OUT / "V_embeddings.npy", V_emb)
    np.save(OUT / "A_embeddings.npy", A_emb)

    # Standardise
    V = (V_emb - V_emb.mean(0)) / (V_emb.std(0) + 1e-9)
    A = (A_emb - A_emb.mean(0)) / (A_emb.std(0) + 1e-9)

    # ── 1) CCA — the natural cross-modal method ─────────────────────
    print("\n══ Method 1: Canonical Correlation Analysis (CCA) ══════════")
    n_comp = 12
    cca, U, S, corrs = cca_analysis(V, A, n_components=n_comp)
    print(f"  Top {n_comp} canonical correlations:")
    for i, c in enumerate(corrs):
        bar = "█" * int(40 * abs(c))
        print(f"    CC{i+1:2d}: r = {c:+.4f}  {bar}")

    # Plot canonical correlations
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, n_comp + 1), corrs, color="#065A82")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("canonical component"); ax.set_ylabel("correlation")
    ax.set_title(f"Canonical correlations between visual and audio embeddings\n"
                 f"(mean-quantised 64-D features per time bin)")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "cca_correlations.png", dpi=140)
    plt.close()

    # First-component scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(U[:, 0], S[:, 0], s=6, alpha=0.4, color="#065A82")
    ax.set_xlabel("Visual canonical component 1")
    ax.set_ylabel("Audio  canonical component 1")
    ax.set_title(f"CC1 scatter: r = {corrs[0]:+.3f}  (T={U.shape[0]} bins)")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "cca_cc1_scatter.png", dpi=140)
    plt.close()

    # Report which V/A dims dominate CC1
    print(f"\n  CC1 loadings (top 8 by |weight|):")
    v_load = cca.x_weights_[:, 0]
    a_load = cca.y_weights_[:, 0]
    top_v = np.argsort(-np.abs(v_load))[:8]
    top_a = np.argsort(-np.abs(a_load))[:8]
    print(f"    Visual  dims: {['V%02d(%+.2f)'%(i, v_load[i]) for i in top_v]}")
    print(f"    Audio   dims: {['A%02d(%+.2f)'%(i, a_load[i]) for i in top_a]}")

    # ── 2) Glasso on the joint 128-D matrix ────────────────────────
    print("\n══ Method 2: Glasso on joint 128-D matrix ══════════════════")
    X = np.concatenate([V, A], axis=1)
    partial, stab, alpha = glasso_analysis(X, OUT, tag="emb")

    labels = [f"V{i:02d}" for i in range(V.shape[1])] + \
             [f"A{i:02d}" for i in range(A.shape[1])]
    K_v = V.shape[1]

    # Cross-modal block (V×A)
    cross_partial = partial[:K_v, K_v:]
    cross_stab    = stab   [:K_v, K_v:]

    print(f"\n  Cross-modal partial correlations:")
    print(f"    max |partial corr| = {np.abs(cross_partial).max():.3f}")
    print(f"    max stability      = {cross_stab.max():.3f}")

    # Ranked cross edges
    rows = []
    for i in range(K_v):
        for j in range(V.shape[1] * 0, cross_partial.shape[1]):
            rows.append((f"V{i:02d}", f"A{j:02d}", float(cross_partial[i, j]),
                         float(cross_stab[i, j])))
    df = pd.DataFrame(rows, columns=["v_dim","a_dim","partial_corr","stability"])
    df["abs_pc"] = df.partial_corr.abs()
    df = df.sort_values("abs_pc", ascending=False)
    print(f"\n  Top 20 cross-modal edges by |partial correlation|:")
    print(df.drop(columns="abs_pc").head(20).to_string(index=False))
    df.drop(columns="abs_pc").to_csv(OUT / "cross_edges.csv", index=False)

    print(f"\n  Cross edges surviving bootstrap stability ≥ 0.5: "
          f"{int((cross_stab >= 0.5).sum())}")
    print(f"  Cross edges surviving bootstrap stability ≥ 0.7: "
          f"{int((cross_stab >= 0.7).sum())}")
    print(f"  Cross edges surviving bootstrap stability ≥ 0.9: "
          f"{int((cross_stab >= 0.9).sum())}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax = np.abs(cross_partial).max()
    sns.heatmap(cross_partial, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                ax=axes[0], annot=False, cbar_kws={"label": "partial correlation"})
    axes[0].set_title("Cross-modal partial correlation (64V × 64A)")
    axes[0].set_xlabel("Audio dims"); axes[0].set_ylabel("Visual dims")
    sns.heatmap(cross_stab, cmap="magma", vmin=0, vmax=1, ax=axes[1],
                cbar_kws={"label": "bootstrap stability"})
    axes[1].set_title("Cross-modal bootstrap stability")
    axes[1].set_xlabel("Audio dims"); axes[1].set_ylabel("Visual dims")
    plt.tight_layout(); plt.savefig(OUT / "cross_matrices.png", dpi=140)
    plt.close()

    print(f"\n✅ Everything in {OUT}/")


if __name__ == "__main__":
    main()
