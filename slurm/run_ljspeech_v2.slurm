#!/bin/bash
#SBATCH --account=<ALLOCATION>
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=vqvae-audio-v2
#SBATCH --output=/projects/<ALLOCATION>/<NETID>/vqvae/runs/ljspeech_v2/slurm-%j.out
#SBATCH --error=/projects/<ALLOCATION>/<NETID>/vqvae/runs/ljspeech_v2/slurm-%j.err

set -euo pipefail
mkdir -p /projects/<ALLOCATION>/<NETID>/vqvae/runs/ljspeech_v2
cd /projects/<ALLOCATION>/<NETID>/vqvae

module purge
module load python-miniconda3/4.12.0
source activate /projects/<ALLOCATION>/<NETID>/envs/vqvae

echo "=== Node:    $(hostname)"
echo "=== Date:    $(date)"
echo "=== GPU:    "; nvidia-smi -L
echo ""

python -u train_ljspeech.py \
    --data /projects/<ALLOCATION>/<NETID>/data/LJSpeech-1.1 \
    --out  /projects/<ALLOCATION>/<NETID>/vqvae/runs/ljspeech_v2 \
    --batch 64 \
    --steps 15000 \
    --codebook 1024 \
    --hidden 192 \
    --latent 64 \
    --lr 2e-4 \
    --workers 8 \
    --log_every 100 \
    --save_every 1500
