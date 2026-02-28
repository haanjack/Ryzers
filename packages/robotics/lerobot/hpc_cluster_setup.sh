#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# This script sets up the environment for training a policy using the LeRobot framework on the AMD AI & HPC cluster.
# It creates a Python virtual environment, installs necessary packages, and submits an example job to the scheduler.
# AMD AI & HPC cluster: https://www.amd.com/en/corporate/university-program/ai-hpc-cluster.html

cd $WORK
python3.12 -m venv robot_env
source robot_env/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

git clone https://github.com/huggingface/lerobot
cd lerobot
git checkout edfebd5
sed -i '67d' pyproject.toml
python -m pip install -e .

python -m pip install transformers pytest

# =========================================
# Single-GPU Job Template
# =========================================
cat << 'EOF' > job.train_pusht_example
#!/bin/bash
#SBATCH -J train_policy           # Job name
#SBATCH -o train_pusht_example.log         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -t 08:00:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p defq               # Desired partition
module purge
module load hpcfund
source $WORK/robot_env/bin/activate
export HF_HOME="${HOME}/hf_models/" && mkdir -p "$HF_HOME" # mkdir if it doesn't exist
cd $WORK/lerobot
python lerobot/scripts/train.py \
--policy.type=act \
--dataset.repo_id=lerobot/pusht \
--dataset.video_backend=pyav \
--policy.device=cuda
EOF

# =========================================
# Multi-GPU Single-Node Distributed Training (4 GPUs)
# =========================================
cat << 'EOF' > job.train_pusht_distributed_4gpu
#!/bin/bash
#SBATCH -J train_dist_4gpu       # Job name
#SBATCH -o train_pusht_distributed_4gpu.log  # Name of stdout output file
#SBATCH -N 1                     # Total number of nodes requested
#SBATCH -n 4                     # Number of tasks (GPUs)
#SBATCH -t 08:00:00              # Run time (hh:mm:ss)
#SBATCH -p mi2104x               # Desired partition (4x MI210 GPUs per node)

module purge
module load hpcfund
source $WORK/robot_env/bin/activate
export HF_HOME="${HOME}/hf_models/" && mkdir -p "$HF_HOME"
cd $WORK/lerobot

# Distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export RCCL_ENABLE_INTERRUPT=1

# Run distributed training with torchrun
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/pusht \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --training.batch_size=64
EOF

# =========================================
# Multi-Node Distributed Training (2 nodes, 8 GPUs total)
# =========================================
cat << 'EOF' > job.train_pusht_multinode
#!/bin/bash
#SBATCH -J train_multinode        # Job name
#SBATCH -o train_pusht_multinode.log  # Name of stdout output file
#SBATCH -N 2                      # Total number of nodes requested
#SBATCH -n 8                      # Total number of tasks (4 GPUs per node x 2 nodes)
#SBATCH -t 08:00:00               # Run time (hh:mm:ss)
#SBATCH -p mi2104x                # Desired partition

module purge
module load hpcfund
source $WORK/robot_env/bin/activate
export HF_HOME="${HOME}/hf_models/" && mkdir -p "$HF_HOME"
cd $WORK/lerobot

# Distributed training environment variables
# Get master address from SLURM
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=8
export RCCL_ENABLE_INTERRUPT=1
export NODE_RANK=$SLURM_NODEID

# Number of processes per node
NPROC_PER_NODE=4

echo "Multi-node distributed training:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NODE_RANK: $NODE_RANK"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"

# Run distributed training with srun
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/pusht \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --training.batch_size=128
EOF

# =========================================
# GR00T Fine-tuning Job (Single GPU)
# =========================================
cat << 'EOF' > job.train_groot_example
#!/bin/bash
#SBATCH -J train_groot            # Job name
#SBATCH -o train_groot_example.log  # Name of stdout output file
#SBATCH -N 1                      # Total number of nodes requested
#SBATCH -t 24:00:00               # Run time (hh:mm:ss) - 24 hours for GR00T
#SBATCH -p mi2104x                # Desired partition

module purge
module load hpcfund
source $WORK/robot_env/bin/activate
export HF_HOME="${HOME}/hf_models/" && mkdir -p "$HF_HOME"
export HF_TOKEN=""  # Set your Hugging Face token here
cd $WORK/lerobot

# Login to Hugging Face
hf auth login --token "$HF_TOKEN" --add-to-git-credential

# Run GR00T fine-tuning
python lerobot/scripts/train.py \
    --policy.type=groot \
    --dataset.repo_id=lerobot/pusht \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --training.save_every=1000 \
    --training.eval_every=1000
EOF

# =========================================
# GR00T Multi-GPU Fine-tuning (4 GPUs)
# =========================================
cat << 'EOF' > job.train_groot_distributed_4gpu
#!/bin/bash
#SBATCH -J train_groot_dist       # Job name
#SBATCH -o train_groot_distributed_4gpu.log  # Name of stdout output file
#SBATCH -N 1                      # Total number of nodes requested
#SBATCH -n 4                      # Number of tasks (GPUs)
#SBATCH -t 24:00:00               # Run time (hh:mm:ss) - 24 hours for GR00T
#SBATCH -p mi2104x                # Desired partition

module purge
module load hpcfund
source $WORK/robot_env/bin/activate
export HF_HOME="${HOME}/hf_models/" && mkdir -p "$HF_HOME"
export HF_TOKEN=""  # Set your Hugging Face token here
cd $WORK/lerobot

# Login to Hugging Face
hf auth login --token "$HF_TOKEN" --add-to-git-credential

# Distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export RCCL_ENABLE_INTERRUPT=1

# Run GR00T distributed fine-tuning
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    lerobot/scripts/train.py \
    --policy.type=groot \
    --dataset.repo_id=lerobot/pusht \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --training.batch_size=64 \
    --training.save_every=1000 \
    --training.eval_every=1000
EOF

echo "========================================"
echo "Created SLURM job scripts:"
echo "  - job.train_pusht_example         (single-GPU)"
echo "  - job.train_pusht_distributed_4gpu (4 GPUs, 1 node)"
echo "  - job.train_pusht_multinode        (8 GPUs, 2 nodes)"
echo "  - job.train_groot_example          (single-GPU GR00T)"
echo "  - job.train_groot_distributed_4gpu (4 GPUs GR00T)"
echo "========================================"
echo ""
echo "To submit a job:"
echo "  sbatch job.train_pusht_example"
echo "  sbatch job.train_pusht_distributed_4gpu"
echo "  sbatch job.train_pusht_multinode"
echo ""
echo "Don't forget to set HF_TOKEN in the job scripts for GR00T training!"
echo ""

# Submit the default single-GPU job
sbatch job.train_pusht_example
tail -f $WORK/lerobot/train_pusht_example.log
