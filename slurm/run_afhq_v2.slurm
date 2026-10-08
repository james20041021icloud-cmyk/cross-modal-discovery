#!/bin/bash
#SBATCH --account=<ALLOCATION>
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=vqvae-afhq-v2
#SBATCH --output=/projects/<ALLOCATION>/<NETID>/vqvae/runs/afhq_v2/slurm-%j.out
#SBATCH --error=/projects/<ALLOCATION>/<NETID>/vqvae/runs/afhq_v2/slurm-%j.err

set -euo pipefail
mkdir -p /projects/<ALLOCATION>/<NETID>/vqvae/runs/afhq_v2
cd /projects/<ALLOCATION>/<NETID>/vqvae

module purge
module load python-miniconda3/4.12.0
source activate /projects/<ALLOCATION>/<NETID>/envs/vqvae

echo "=== Node:    $(hostname)"
echo "=== Date:    $(date)"
echo "=== GPU:    "; nvidia-smi -L
echo ""

python -u train_afhq.py \
    --data /projects/<ALLOCATION>/<NETID>/data/afhq/train \
    --out  /projects/<ALLOCATION>/<NETID>/vqvae/runs/afhq_v2 \
    --image_size 384 \
    --batch 24 \
    --steps 15000 \
    --codebook 2048 \
    --hidden 192 \
    --latent 64 \
    --lr 2e-4 \
    --workers 8 \
    --log_every 100 \
    --save_every 1500
