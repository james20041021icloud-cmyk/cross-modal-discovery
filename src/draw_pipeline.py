"""Render the corrected pipeline diagram as a clean PNG."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("docs/pipeline_diagram.png")

fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 14); ax.set_ylim(0, 22); ax.axis("off")

C_VIS   = "#2E86AB"
C_AUD   = "#E63946"
C_SHARE = "#065A82"
C_STAGE = "#F72585"
C_OUT   = "#06D6A0"

def box(x, y, w, h, text, fc="white", ec="#333", fontsize=9,
        weight="normal", text_color="black"):
    """Draw a rounded box with text inside. Text rendered ONCE."""
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.05,rounding_size=0.15",
                       linewidth=1.5, facecolor=fc, edgecolor=ec)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color, wrap=True)

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=15, color="#333", linewidth=1.5)
    ax.add_patch(a)

def stage_label(x, y, text):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_STAGE,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFE7F0",
                      edgecolor=C_STAGE, linewidth=1.5))

# ── Title
ax.text(0.4, 21.7, "Cross-modal discovery pipeline",
        fontsize=16, fontweight="bold", color=C_SHARE)
ax.text(0.4, 21.2, "Coarse  →  fine  →  concrete localization",
        fontsize=11, fontstyle="italic", color="#666")

# ── Input
box(7, 20.5, 5, 0.7, "INPUT   paired video",
    fc="#EEE", fontsize=12, weight="bold")

# ── Split
box(3.5, 19.3, 3.8, 0.75, "Visual stream\n(T frames)",
    fc="#DCE9F2", fontsize=10, weight="bold")
box(10.5, 19.3, 3.8, 0.75, "Audio stream\n(T 1-sec windows)",
    fc="#F9DDDF", fontsize=10, weight="bold")
arrow(6.3, 20.15, 4.5, 19.75)
arrow(7.7, 20.15, 9.5, 19.75)

# ── VQ-VAEs
box(3.5, 18.0, 4.0, 0.8, "Visual Soft-VQ-VAE\n(independent training)",
    fc=C_VIS, ec=C_VIS, fontsize=10, weight="bold", text_color="white")
box(10.5, 18.0, 4.0, 0.8, "Audio Soft-VQ-VAE\n(independent training)",
    fc=C_AUD, ec=C_AUD, fontsize=10, weight="bold", text_color="white")
arrow(3.5, 18.9, 3.5, 18.45)
arrow(10.5, 18.9, 10.5, 18.45)

# ── Codebooks
box(3.5, 16.7, 4.0, 0.8, "Visual codebook\n(256 × 64) × 2 RVQ",
    fc="#EDF5FA", fontsize=10)
box(10.5, 16.7, 4.0, 0.8, "Audio codebook\n(128 × 64) × 2 RVQ",
    fc="#FCE9EA", fontsize=10)
arrow(3.5, 17.6, 3.5, 17.15)
arrow(10.5, 17.6, 10.5, 17.15)

# ── Index maps
box(3.5, 15.4, 4.0, 0.8, "Per-frame index maps\n(T, 32, 32)  integers",
    fc="#EDF5FA", fontsize=10)
box(10.5, 15.4, 4.0, 0.8, "Per-window index seqs\n(T, 1000)  integers",
    fc="#FCE9EA", fontsize=10)
arrow(3.5, 16.3, 3.5, 15.85)
arrow(10.5, 16.3, 10.5, 15.85)

# ── 64-D embeddings
box(3.5, 14.1, 4.0, 0.8, "Visual 64-D embedding\n(T, 64)   mean-pool spatial",
    fc="#DCE9F2", fontsize=10)
box(10.5, 14.1, 4.0, 0.8, "Audio 64-D embedding\n(T, 64)   mean-pool temporal",
    fc="#F9DDDF", fontsize=10)
arrow(3.5, 15.0, 3.5, 14.55)
arrow(10.5, 15.0, 10.5, 14.55)

# ── Stage 1: CCA
stage_label(7, 13.05, "STAGE 1  •  Coarse alignment")
box(7, 12.15, 6.0, 0.8,
    "CCA on time-aligned  (T, 128)\nr₁ = 0.68,   12 shared canonical axes",
    fc=C_SHARE, ec=C_SHARE, fontsize=10, weight="bold", text_color="white")
arrow(3.9, 13.7, 5.3, 12.55)
arrow(10.1, 13.7, 8.7, 12.55)

# ── Matched pairs
box(7, 10.95, 6.5, 0.8,
    "MATCHED code pairs   (V-code #i , A-code #j)\n"
    "top-K pairs ranked by CCA drive  (K = 5–10)",
    fc="#F0F9FC", fontsize=10)
arrow(7, 11.75, 7, 11.35)

# ── Stage 2 wrapper
stage_label(7, 9.85, "STAGE 2  •  For each matched pair  (loop)")

big = FancyBboxPatch((0.5, 4.9), 13.0, 4.3,
                     boxstyle="round,pad=0.15,rounding_size=0.15",
                     linewidth=2, edgecolor=C_STAGE, facecolor="#FFF8FB",
                     linestyle="--")
ax.add_patch(big)

box(3.5, 8.4, 5.0, 1.0,
    "Reverse-lookup V-code #i in index maps\n"
    "spatial occurrence  (T, 32×32)\n"
    "downsample 4× →  (T, 64)",
    fc="#EDF5FA", fontsize=9)
box(10.5, 8.4, 5.0, 1.0,
    "Reverse-lookup A-code #j in index seqs\n"
    "temporal occurrence  (T, 1000)\n"
    "downsample 20× →  (T, 50)",
    fc="#FCE9EA", fontsize=9)

box(7, 7.1, 6.5, 0.8,
    "Concatenate  →  time-aligned  (T, 114)  binary matrix",
    fc="white", fontsize=10)
arrow(3.5, 7.9, 5.3, 7.5)
arrow(10.5, 7.9, 8.7, 7.5)

box(7, 5.8, 8.0, 1.0,
    "Graphical Lasso  +  block-bootstrap stability\n"
    "cross-block  (64 spatial × 50 temporal)\n"
    "= most-matched fragment pairs",
    fc=C_SHARE, ec=C_SHARE, fontsize=10, weight="bold", text_color="white")
arrow(7, 6.7, 7, 6.3)

# ── Stage 3
arrow(7, 4.9, 7, 4.3)
stage_label(7, 3.95, "STAGE 3  •  Reverse to physical location")
box(7, 3.0, 9.0, 1.0,
    "spatial index p  →  16 × 16 pixel region on the 128 × 128 frame\n"
    "temporal index τ  →  20 ms segment  →  FFT  →  frequency band",
    fc="#EAF9F3", fontsize=10)
arrow(7, 3.5, 7, 3.5)

# ── Stage 4
arrow(7, 2.5, 7, 1.9)
stage_label(7, 1.55, "STAGE 4  •  Composite visualization per matched pair")
box(7, 0.55, 10.5, 1.2,
    "frame + red spatial boxes   │   waveform + red time bands   │   FFT panel\n\n"
    "→ answers:  \"this pixel region  ↔  this frequency band\"",
    fc=C_OUT, ec=C_OUT, fontsize=10, weight="bold", text_color="white")

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
