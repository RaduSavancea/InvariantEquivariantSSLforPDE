#!/bin/bash
#SBATCH --job-name=comb_loss_essl
#SBATCH --partition=slowlane
#SBATCH --qos=student_high_res

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
# 3. Run hybrid Essl Losses Combined
# ============================================================

cd /mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs

echo "---"
echo "Starting Navier–Stokes future-state prediction"
echo "---"

echo "Started at $(date)"


echo "---"
echo "Starting Combined losses Train"
echo "---"
echo "Started at $(date)"

python -u pde_inv_equiv.py \
    --data-root "$LOCAL_DATA_PATH"

echo "Finished at $(date)"
