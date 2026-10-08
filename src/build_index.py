"""Build a single-page summary index of all successful pairs."""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

ROOT = Path(os.environ.get("MMVQ_ROOT", ".")) / "results/pipeline_final"

with open(ROOT / "results.json") as f:
    R = json.load(f)

ok = [r for r in R["results"] if r.get("status") == "ok"]
ok.sort(key=lambda p: -abs(p["r_canonical"]))
print(f"{len(ok)} successful pairs to display")

# 4 columns, ceil(n/4) rows of thumbnails, plus a header
n_cols = 4
n_rows = (len(ok) + n_cols - 1) // n_cols

fig = plt.figure(figsize=(n_cols * 4.5, n_rows * 3.2 + 1.5))
gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[0.5] + [1] * n_rows,
                      hspace=0.35, wspace=0.15,
                      left=0.03, right=0.98, top=0.97, bottom=0.02)

# ── Header
ax_h = fig.add_subplot(gs[0, :]); ax_h.axis("off")
ax_h.text(0.5, 0.7,
          "Cross-Modal Fragment Localization  ·  Pipeline Summary Index",
          ha="center", va="center", fontsize=18, fontweight="bold",
          color="#065A82", transform=ax_h.transAxes)
ax_h.text(0.5, 0.15,
          f"{len(ok)} successful matched pairs   ·   ranked by CCA canonical correlation   ·   "
          f"click into  pairs/pair_XX.png  for full composite",
          ha="center", va="center", fontsize=11, fontstyle="italic",
          color="#666", transform=ax_h.transAxes)

# ── Thumbnails
for idx, r in enumerate(ok):
    row = idx // n_cols + 1
    col = idx % n_cols
    ax = fig.add_subplot(gs[row, col])

    img_path = ROOT / r["composite_png"]
    if img_path.exists():
        img = np.array(Image.open(img_path))
        ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"pair #{r['pair_idx']+1}  ·  r = {r['r_canonical']:+.3f}\n"
        f"V-L{r['v_level']}-{r['v_code']:03d}   ↔   "
        f"A-L{r['a_level']}-{r['a_code']:03d}   ·   {r['n_edges']} edges",
        fontsize=9)

# Fill remaining cells with blanks
for idx in range(len(ok), n_rows * n_cols):
    row = idx // n_cols + 1
    col = idx % n_cols
    ax = fig.add_subplot(gs[row, col]); ax.axis("off")

fig.savefig(ROOT / "index.png", dpi=110, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"Saved: {ROOT / 'index.png'}")
