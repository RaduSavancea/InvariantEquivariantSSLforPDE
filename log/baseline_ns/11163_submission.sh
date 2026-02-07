#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=10
#SBATCH --error=/mnt/nfs/home/ac148019/SSLForPDEs/log/baseline_ns/%j_0_log.err
#SBATCH --gpus-per-node=4
#SBATCH --job-name=baseline_ns
#SBATCH --mem=160GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --open-mode=append
#SBATCH --output=/mnt/nfs/home/ac148019/SSLForPDEs/log/baseline_ns/%j_0_log.out
#SBATCH --partition=slowlane
#SBATCH --signal=USR2@120
#SBATCH --time=2880
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /mnt/nfs/home/ac148019/SSLForPDEs/log/baseline_ns/%j_%t_log.out --error /mnt/nfs/home/ac148019/SSLForPDEs/log/baseline_ns/%j_%t_log.err /mnt/nfs/home/ac148019/.conda/envs/ssl/bin/python -u -m submitit.core._submit /mnt/nfs/home/ac148019/SSLForPDEs/log/baseline_ns
