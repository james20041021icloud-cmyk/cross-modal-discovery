#!/bin/bash
#SBATCH --account=<ALLOCATION>
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=mm-vqvae
#SBATCH --output=/projects/<ALLOCATION>/<NETID>/multimodal_vqvae/logs/slurm-%j.out
#SBATCH --error=/projects/<ALLOCATION>/<NETID>/multimodal_vqvae/logs/slurm-%j.err

set -euo pipefail

# ─── Configurable ─────────────────────────────────────────────────────────
VIDEO="${VIDEO:-/projects/<ALLOCATION>/<NETID>/multimodal_vqvae/data/input.mp4}"
FPS="${FPS:-4}"
IMG_SIZE="${IMG_SIZE:-128}"
K_VISUAL="${K_VISUAL:-512}"
K_AUDIO="${K_AUDIO:-256}"
STEPS_VISUAL="${STEPS_VISUAL:-3000}"
STEPS_AUDIO="${STEPS_AUDIO:-3000}"
BATCH_VISUAL="${BATCH_VISUAL:-64}"
BATCH_AUDIO="${BATCH_AUDIO:-64}"
# ──────────────────────────────────────────────────────────────────────────

ROOT=/projects/<ALLOCATION>/<NETID>/multimodal_vqvae
DATA=$ROOT/data
MODELS=$ROOT/models
TOKENS=$ROOT/tokens

cd $ROOT/code

module purge
module load python-miniconda3/4.12.0
module load ffmpeg/6.1-gcc-11.2.0
source activate /projects/<ALLOCATION>/<NETID>/envs/vqvae
which ffmpeg ffprobe

echo "=== Node:  $(hostname)"
echo "=== Date:  $(date)"
echo "=== GPU:  "; nvidia-smi -L
echo "=== Video: $VIDEO"
echo "=== FPS:   $FPS,  img_size=$IMG_SIZE"
echo "=== Codebooks: visual=$K_VISUAL, audio=$K_AUDIO"
echo "=== Steps:    visual=$STEPS_VISUAL, audio=$STEPS_AUDIO"
echo ""

if [ ! -f "$VIDEO" ]; then
  echo "ERROR: video not found at $VIDEO"
  exit 1
fi

# ─── Step 1: extract frames + audio ───────────────────────────────────────
echo ""; echo "▶ Step 1/4 — extract frames + audio"
python -u extract_media.py "$VIDEO" "$DATA" --fps "$FPS" --size "$IMG_SIZE"

# ─── Step 2a: train visual VQ-VAE ────────────────────────────────────────
echo ""; echo "▶ Step 2/4 — train visual Soft-VQ-VAE"
python -u train_visual_vqvae.py \
    --frames_dir "$DATA/frames" \
    --out_dir    "$MODELS/visual" \
    --size       "$IMG_SIZE" \
    --batch      "$BATCH_VISUAL" \
    --steps      "$STEPS_VISUAL" \
    --codebook   "$K_VISUAL" \
    --hidden     128 \
    --latent     64 \
    --lr         3e-4 \
    --log_every  100 \
    --save_every 500

# ─── Step 2b: train audio VQ-VAE ────────────────────────────────────────
echo ""; echo "▶ Step 3/4 — train audio Soft-VQ-VAE"
python -u train_audio_vqvae.py \
    --audio_path "$DATA/audio.wav" \
    --out_dir    "$MODELS/audio" \
    --batch      "$BATCH_AUDIO" \
    --steps      "$STEPS_AUDIO" \
    --codebook   "$K_AUDIO" \
    --hidden     128 \
    --latent     64 \
    --lr         3e-4 \
    --log_every  100 \
    --save_every 500 \
    --n_virtual  4000

# ─── Step 3: extract time-aligned tokens ───────────────────────────────
echo ""; echo "▶ Step 4/4 — extract time-aligned token histograms"
python -u extract_tokens.py \
    --frames_dir  "$DATA/frames" \
    --audio_path  "$DATA/audio.wav" \
    --visual_ckpt "$MODELS/visual/visual_vqvae.pt" \
    --audio_ckpt  "$MODELS/audio/audio_vqvae.pt" \
    --out_dir     "$TOKENS" \
    --fps         "$FPS"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All done!"
echo "  Frames / audio : $DATA"
echo "  Trained models : $MODELS"
echo "  Token streams  : $TOKENS"
echo "════════════════════════════════════════════════════════════════"
