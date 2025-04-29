#!/bin/bash
#SBATCH --account=project_2012524
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:a100:4
#SBATCH --reservation=scc-gpu

module purge
module load pytorch/2.5


#export HF_TOKEN="your token"
export HF_HOME="/scratch/project_2012524/${USER}/hf-cache"
mkdir -p $HF_HOME


# Check if HF_TOKEN is set
if [ -z "${HF_TOKEN}" ]; then
	echo "Error: HF_TOKEN environment variable is not set."
	exit 1
fi

# Speed benchmark
torchrun --nproc_per_node=4 main.py --benchmark speed --device-type cuda

# Speed benchmark and save stdout/loggings
#srun torchrun --nproc_per_node=4 main.py --benchmark speed --device-type cuda > loggings/speed.log 2<&1


# Speed and accuracy benchmark
#srun torchrun --nproc_per_node=4 main.py --benchmark accuracy --device-type cuda

# Accuracy benchmark
#srun torchrun --nproc_per_node=4 main.py --benchmark accuracy --device-type cuda --checkpoint checkpoints

# Accuracy benchmark and save stdout/loggings
#srun torchrun --nproc_per_node=4 main.py --benchmark accuracy --device-type cuda --checkpoint checkpoints > loggings/accuracy.log 2<&1
