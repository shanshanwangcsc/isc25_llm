#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-1
#SBATCH --gpus-per-node=8

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5


export HF_TOKEN="you token"
export HF_HOME="/scratch/project_462000131/${USER}/hf_cache"
mkdir -p $HF_HOME
export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

TORCHRUN_ARGS="--nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT"

# Speed benchmark
#torchrun --nproc_per_node=4 main.py --benchmark speed --device-type cuda
#srun torchrun $TORCHRUN_ARGS main.py --benchmark speed --device-type cuda

# Speed benchmark and save stdout/loggings
srun torchrun $TORCHRUN_ARGS main.py --benchmark speed --device-type cuda > loggings/speed_1node_lumi.log 2<&1


# Speed and accuracy benchmark
#torchrun --nproc_per_node=1 main.py --benchmark accuracy --device-type cuda

# Accuracy benchmark
#torchrun --nproc_per_node=1 main.py --benchmark accuracy --device-type cuda --checkpoint checkpoints

# Accuracy benchmark and save stdout/loggings
#torchrun --nproc_per_node=1 main.py --benchmark accuracy --device-type cuda --checkpoint checkpoints > loggings/accuracy.log 2<&1
