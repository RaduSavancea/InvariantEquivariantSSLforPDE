#!/bin/bash
#SBATCH --job-name=equiv_ns_ssl
#SBATCH --partition=slowlane
#SBATCH --qos=student_high_res

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

#SBATCH --output=outputs/aisa/%x_%j.out
#SBATCH --error=outputs/aisa/%x_%j.err



#=======================================================================
# 0. Environment & Safety

set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO

# Required for HDF5 on NFS
export HDF5_USE_FILE_LOCKING=FALSE

echo "=================================================="
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "GPUs:     $SLURM_GPUS"
echo "=================================================="


# 1. Modules & Conda

#module purge
#module load Anaconda3/2024.02-1

# Activate your existing environment
#source activate /mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs/.conda

module purge
module load Miniconda3

# This is the recommended way to get conda activate as a shell function:
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"
conda activate /mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs/.conda




echo "Using Python:"
which python

python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# 2. Data Staging (NFS → Local SSD)

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


# 3. Run Training

cd /mnt/nfs/home/st195720/SSLPDEs/SSLForPDEs
mkdir -p runs/equiv_ns

echo "---"
echo "Starting baseline Navier–Stokes SSL training"
echo "---"

echo "Reached training at $(date)"

python -u baseline_train.py \
    --data-root "$LOCAL_DATA_PATH" \
    --batch-size 8 \
    --epochs 10

echo "Training finished successfully."
