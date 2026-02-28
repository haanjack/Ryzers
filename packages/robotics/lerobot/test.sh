#!/usr/bin/env bash
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "========================================"
echo "LeRobot GR00T Fine-tuning Test"
echo "========================================"

# Test 1: Basic lerobot installation check
echo ""
echo "Test 1: Checking lerobot installation..."
python -c "import lerobot; print(f'LeRobot version: {lerobot.__version__}')"

# Test 2: Check GR00T policy is available
echo ""
echo "Test 2: Checking GR00T policy availability..."
python -c "from lerobot.policies.groot.modeling_groot import GrootPolicy; print('GR00T/GrootPolicy imported successfully')"

# Detect available GPUs
echo ""
echo "Test 3: Detecting available GPUs..."
GPU_COUNT=$(python3 /ryzers/detect_gpus.py --count)
echo "Detected $GPU_COUNT GPU(s)"

# Check if distributed training is explicitly disabled
if [ "${DISTRIBUTED:-1}" = "0" ]; then
    echo "Distributed training disabled by DISTRIBUTED=0"
    GPU_COUNT=1
fi

# Use NPROC_PER_NODE if explicitly set
if [ -n "$NPROC_PER_NODE" ]; then
    echo "Using NPROC_PER_NODE=$NPROC_PER_NODE"
    GPU_COUNT=$NPROC_PER_NODE
fi

# Patch test training script to run quickly
echo ""
echo "Patching training script for quick test..."
sed -i 's/training_steps = 5000/training_steps = 2/' /ryzers/lerobot/examples/training/train_policy.py
sed -i 's/batch_size=64/batch_size=4/' /ryzers/lerobot/examples/training/train_policy.py

# Run test based on GPU count
if [ "$GPU_COUNT" -gt 1 ]; then
    echo ""
    echo "========================================"
    echo "Running distributed training test with $GPU_COUNT GPUs..."
    echo "========================================"

    # Set environment variables for distributed training
    export WORLD_SIZE=$GPU_COUNT
    export MASTER_ADDR="${MASTER_ADDR:-localhost}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export RCCL_ENABLE_INTERRUPT="${RCCL_ENABLE_INTERRUPT:-1}"

    # Run distributed training test
    exec /ryzers/distributed_train.sh --nproc-per-node "$GPU_COUNT" -- \
        --dataset.repo_id=lerobot/pusht \
        --policy.type=act \
        --training.online_steps=2 \
        --policy.device=cuda
else
    echo ""
    echo "========================================"
    echo "Running single-GPU training test..."
    echo "========================================"

    # Run Test
    python /ryzers/lerobot/examples/training/train_policy.py
fi

echo ""
echo "========================================"
echo "All tests passed!"
echo "========================================"
