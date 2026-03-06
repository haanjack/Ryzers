#!/usr/bin/env bash
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

# Defaults (override via CLI args below)
POLICY_TYPE="act"
TRAINING_STEPS="2"
BATCH_SIZE="4"
NPROC_PER_NODE="1"
NNODES="1"
NODE_RANK="0"
MASTER_ADDR="localhost"
MASTER_PORT="29500"
DRY_RUN=false
TRAIN_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --policy)
            POLICY_TYPE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --training-steps)
            TRAINING_STEPS="$2"
            shift 2
            ;;
        --nproc-per-node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --use-compile)
            TRAIN_ARGS+=(\"--use-compile\")
            shift
            ;;
        --use-amp)
            TRAIN_ARGS+=(\"--use-amp\")
            shift
            ;;

        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --)
            shift
            TRAIN_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS] [-- TRAIN_ARGS]"
            echo ""
            echo "Options:"
            echo "  --policy act|groot       Policy type (default: act)"
            echo "  --batch-size N           Batch size (default: 4)"
            echo "  --training-steps N       Training steps (default: 2)"
            echo "  --nproc-per-node N       GPUs per node (default: 1)"
            echo "  --nnodes N               Number of nodes (default: 1)"
            echo "  --node-rank N            Node rank (default: 0)"
            echo "  --master-addr ADDR       Master address (default: localhost)"
            echo "  --master-port PORT       Master port (default: 29500)"
            echo "  --use-compile            Enable torch.compile optimization"
            echo "  --use-amp                Enable automatic mixed precision"
            echo "  --dry-run                Print command without executing"
            echo ""
            echo "Example:"
            echo "  $0 --nproc-per-node 1 --batch-size 1 --training-steps 1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "LeRobot Training Test"
echo "========================================"

# Test 1: Basic lerobot installation check
echo ""
echo "Test 1: Checking lerobot installation..."
python -c "import lerobot; print(f'LeRobot version: {lerobot.__version__}')"

# Test 2: Check policy availability
echo ""
echo "Test 2: Checking policy availability..."
if [ "$POLICY_TYPE" = "groot" ]; then
    python -c "from lerobot.policies.groot.modeling_groot import GrootPolicy; print('GR00T/GrootPolicy imported successfully')"
else
    python -c "from lerobot.policies.act.modeling_act import ACTPolicy; print('ACTPolicy imported successfully')"
fi

# Detect available GPUs (for info only)
echo ""
echo "Test 3: Detecting available GPUs..."
DETECTED_GPUS=$(python3 /ryzers/detect_gpus.py --count)
echo "Detected $DETECTED_GPUS total GPU(s)"

# Use NPROC_PER_NODE to decide how many GPUs to use
GPU_COUNT="$NPROC_PER_NODE"
echo "Using $GPU_COUNT GPU(s) for training"

# Patch test training script to run quickly
echo ""
echo "Patching training script for quick test..."
sed -i "s/training_steps = 5000/training_steps = ${TRAINING_STEPS}/" /ryzers/lerobot/examples/training/train_policy.py
sed -i "s/batch_size=64/batch_size=${BATCH_SIZE}/" /ryzers/lerobot/examples/training/train_policy.py

# Pre-download dataset to avoid concurrent download races during torchrun
echo ""
echo "Pre-downloading dataset metadata..."
python - << 'PY'
import os
from huggingface_hub import snapshot_download

hf_home = os.environ.get("HF_HOME", "/root/.cache/huggingface")
local_dir = os.path.join(hf_home, "lerobot", "lerobot", "pusht")
snapshot_download(
    repo_id="lerobot/pusht",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"Dataset cached under: {local_dir}")
PY

# Run test with torchrun (works for both single and multi-GPU)
echo ""
echo "========================================"
echo "Running training test with $GPU_COUNT GPU(s)..."
echo "Policy: $POLICY_TYPE"
echo "========================================"

WORLD_SIZE=$((NNODES * GPU_COUNT))

# Set environment variables for distributed training
export WORLD_SIZE
export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT
export RCCL_ENABLE_INTERRUPT="${RCCL_ENABLE_INTERRUPT:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker0,lo}"

TORCHRUN_CMD="torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPU_COUNT \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT"

# Training arguments
DISTRIBUTED_ARGS=(
    --training-steps=$TRAINING_STEPS
    --batch-size=$BATCH_SIZE
    --policy=$POLICY_TYPE
)

if [ ${#TRAIN_ARGS[@]} -gt 0 ]; then
    FULL_CMD="$TORCHRUN_CMD /ryzers/train_policy_distributed.py ${TRAIN_ARGS[*]}"
else
    FULL_CMD="$TORCHRUN_CMD /ryzers/train_policy_distributed.py ${DISTRIBUTED_ARGS[*]}"
fi

if [ "$DRY_RUN" = true ]; then
    echo "Dry run - command that would be executed:"
    echo "$FULL_CMD"
    exit 0
fi

exec $FULL_CMD

echo ""
echo "========================================"
echo "All tests passed!"
echo "========================================"
