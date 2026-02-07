#!/bin/bash
#SBATCH --job-name=ns_future_pred
#SBATCH --partition=slowlane
#SBATCH --qos=student_high_res

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=1-12:00:00

#SBATCH --output=outputs/aisa/%x_%j.out
#SBATCH --error=outputs/aisa/%x_%j.err

# ============================================================
# 0. Safety & environment
# ============================================================

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export HDF5_USE_FILE_LOCKING=FALSE

echo "=================================================="
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "GPUs:     $SLURM_GPUS"
echo "=================================================="

# ============================================================
# 1. Conda environment
# ============================================================

module purge
module load Miniconda3

eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"
conda activate /mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs/.conda

echo "Using Python:"
which python

python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# ============================================================
# 2. Data staging (NFS → local SSD)
# ============================================================

NFS_DATA_PATH="/mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs/datasets"
LOCAL_DATA_PATH="$TMPDIR/datasets"

echo "---"
echo "Staging dataset"
echo "From: $NFS_DATA_PATH"
echo "To:   $LOCAL_DATA_PATH"
echo "---"

rm -rf "$LOCAL_DATA_PATH"
mkdir -p "$LOCAL_DATA_PATH"
cp -r "$NFS_DATA_PATH"/* "$LOCAL_DATA_PATH"/

echo "Data staging complete."

# ============================================================
# 3. Run future-state training + evaluation
# ============================================================

cd /mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs

echo "---"
echo "Starting EISSL Navier–Stokes future-state prediction"
echo "---"

echo "Started at $(date)"

python -u Next_Step.py \
    --data-root "$LOCAL_DATA_PATH" \
    --encoder-ckpt runs/equiv_ns/checkpoints/encoder_ssl.pt \
    --epochs 20 \
    --batch-size 8

echo "Finished at $(date)"
