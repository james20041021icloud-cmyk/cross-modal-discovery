"""Cross-modal retrieval metrics: correctly done in the CCA canonical space."""
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.cross_decomposition import CCA
import json

EMB = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/embedding_glasso"
OUT = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/cca_metrics"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load and standardize embeddings ─────────────────────────────
V = np.load(EMB / "V_embeddings.npy")
A = np.load(EMB / "A_embeddings.npy")
T, D = V.shape
V_std = (V - V.mean(0)) / (V.std(0) + 1e-9)
A_std = (A - A.mean(0)) / (A.std(0) + 1e-9)

# ── 80/20 random split ─────────────────────────────────────────
np.random.seed(0)
idx = np.random.permutation(T)
n_train = int(0.8 * T)
tr_idx, te_idx = idx[:n_train], idx[n_train:]
V_tr, V_te = V_std[tr_idx], V_std[te_idx]
A_tr, A_te = A_std[tr_idx], A_std[te_idx]
n_te = len(te_idx)
print(f"Train: {n_train}, Test: {n_te}, feature dim per modality: {D}")

# ── Fit CCA on training set ─────────────────────────────────────
def compute_metrics(K):
    """Fit CCA with K components, compute retrieval metrics on test set."""
    cca = CCA(n_components=K, max_iter=1500, tol=1e-5)
    cca.fit(V_tr, A_tr)

    U_tr, S_tr = cca.transform(V_tr, A_tr)
    U_te, S_te = cca.transform(V_te, A_te)
    train_r = np.array([np.corrcoef(U_tr[:, k], S_tr[:, k])[0, 1] for k in range(K)])
    test_r  = np.array([np.corrcoef(U_te[:, k], S_te[:, k])[0, 1] for k in range(K)])

    # For retrieval, work in the shared canonical space, weighted by r_k
    # (the more-correlated axes should count more)
    w = np.abs(train_r)                             # (K,)
    Uw = U_te * w[None, :]                          # (n_te, K)
    Sw = S_te * w[None, :]                          # (n_te, K)

    def ranks(query, database):
        n = query.shape[0]
        r = np.zeros(n, dtype=np.int64)
        # normalise to unit vectors so cosine ≡ dot product
        q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
        d_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-12)
        sims = q_norm @ d_norm.T                    # (n, n) cosine similarity
        for i in range(n):
            sorted_j = np.argsort(-sims[i])         # descending
            r[i] = int(np.where(sorted_j == i)[0][0]) + 1
        return r

    ranks_v2a = ranks(Uw, Sw)      # V-query, retrieve A
    ranks_a2v = ranks(Sw, Uw)      # A-query, retrieve V

    def m(rk):
        return {
            "R@1":  float(np.mean(rk <= 1)),
            "R@5":  float(np.mean(rk <= 5)),
            "R@10": float(np.mean(rk <= 10)),
            "R@50": float(np.mean(rk <= 50)),
            "MedR": float(np.median(rk)),
            "MRR":  float(np.mean(1.0 / rk)),
        }
    return train_r, test_r, m(ranks_v2a), m(ranks_a2v)

# Random baseline for reference
def random_baseline(n):
    return {
        "R@1":  1/n,
        "R@5":  5/n,
        "R@10": 10/n,
        "R@50": 50/n,
        "MedR": (n+1)/2,
        "MRR":  float(np.mean([1.0/(i+1) for i in range(n)])),
    }

# Sweep K to see how retrieval quality depends on number of components used
print("\n══ Cross-modal retrieval metrics ══════════════════════════════")
print(f"  Test set size: N = {n_te}   Random baseline for reference:")
rb = random_baseline(n_te)
print(f"    R@1={rb['R@1']:.4f}  R@5={rb['R@5']:.4f}  R@10={rb['R@10']:.4f}  "
      f"R@50={rb['R@50']:.4f}  MedR={rb['MedR']:.0f}  MRR={rb['MRR']:.4f}")

results = {"n_train": n_train, "n_test": n_te, "D": D,
           "random_baseline": rb, "by_K": {}}
for K in [1, 3, 6, 12, 24]:
    print(f"\n── K = {K} canonical components ──")
    train_r, test_r, m_v2a, m_a2v = compute_metrics(K)
    print(f"  train r₁..r_K mean = {train_r.mean():+.4f}   "
          f"test r₁..r_K mean = {test_r.mean():+.4f}   "
          f"gen. gap = {(train_r.mean() - test_r.mean()):+.4f}")
    print(f"  {'':>10}  {'V→A':>10}  {'A→V':>10}  {'random':>10}   V→A / random")
    for metric in ["R@1", "R@5", "R@10", "R@50", "MedR", "MRR"]:
        v = m_v2a[metric]; a = m_a2v[metric]; r = rb[metric]
        if metric in ["MedR"]:
            speed = f"{r/max(v, 1):.1f}× lower"
        else:
            speed = f"{v/max(r, 1e-6):.1f}× above"
        val_v = f"{v:.4f}" if metric != "MedR" else f"{v:.1f}"
        val_a = f"{a:.4f}" if metric != "MedR" else f"{a:.1f}"
        val_r = f"{r:.4f}" if metric != "MedR" else f"{r:.1f}"
        print(f"  {metric:>10}  {val_v:>10}  {val_a:>10}  {val_r:>10}   {speed}")

    results["by_K"][K] = {
        "train_r_mean": float(train_r.mean()),
        "test_r_mean": float(test_r.mean()),
        "V_to_A": m_v2a,
        "A_to_V": m_a2v,
        "train_r": train_r.tolist(),
        "test_r": test_r.tolist(),
    }

with open(OUT / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# Also save a summary CSV for the final report table
rows = []
for K, r in results["by_K"].items():
    row = {"K": K,
           "train_r_mean": r["train_r_mean"],
           "test_r_mean": r["test_r_mean"]}
    for m in ["R@1", "R@5", "R@10", "R@50", "MedR", "MRR"]:
        row[f"V2A_{m}"] = r["V_to_A"][m]
        row[f"A2V_{m}"] = r["A_to_V"][m]
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
print(f"\n✅ Saved to {OUT}/")
