#!/bin/bash
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Distributed Training Launcher for LeRobot on ROCm
# This script wraps torchrun with proper ROCm/RCCL configuration

set -e

# Default values
NPROC_PER_NODE=""
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
DRY_RUN=false
TRAIN_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "Usage: $0 [OPTIONS] -- [TRAIN_ARGS]"
            echo ""
            echo "Options:"
            echo "  --nproc-per-node N  Number of GPUs per node (default: auto-detect)"
            echo "  --nnodes N          Number of nodes (default: \$NNODES or 1)"
            echo "  --node-rank N       Rank of this node (default: \$NODE_RANK or 0)"
            echo "  --master-addr ADDR  Master node address (default: \$MASTER_ADDR or localhost)"
            echo "  --master-port PORT  Master node port (default: \$MASTER_PORT or 29500)"
            echo "  --dry-run           Print command without executing"
            echo ""
            echo "Example:"
            echo "  $0 --nproc-per-node 4 -- --dataset.repo_id=lerobot/pusht --policy.type=act"
            exit 1
            ;;
    esac
done

# Auto-detect GPU count if not specified
if [ -z "$NPROC_PER_NODE" ] || [ "$NPROC_PER_NODE" = "auto" ]; then
    NPROC_PER_NODE=$(python3 /ryzers/detect_gpus.py --count)
    echo "Auto-detected $NPROC_PER_NODE GPU(s)"
fi

# Calculate world size
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# Set ROCm/RCCL environment variables
export RCCL_ENABLE_INTERRUPT="${RCCL_ENABLE_INTERRUPT:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker0,lo}"

# Export distributed training variables
export WORLD_SIZE
export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT

echo "========================================"
echo "Distributed Training Configuration"
echo "========================================"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo "  NNODES:         $NNODES"
echo "  NODE_RANK:      $NODE_RANK"
echo "  WORLD_SIZE:     $WORLD_SIZE"
echo "  MASTER_ADDR:    $MASTER_ADDR"
echo "  MASTER_PORT:    $MASTER_PORT"
echo "  RCCL_ENABLE_INTERRUPT: $RCCL_ENABLE_INTERRUPT"
echo "========================================"

# Build torchrun command
TORCHRUN_CMD="torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT"

# If training args provided, run the LeRobot training script with them
if [ ${#TRAIN_ARGS[@]} -gt 0 ]; then
    FULL_CMD="$TORCHRUN_CMD /ryzers/lerobot/examples/training/train_policy.py ${TRAIN_ARGS[*]}"
else
    # Default: just run torchrun (user should provide the script)
    FULL_CMD="$TORCHRUN_CMD"
fi

if [ "$DRY_RUN" = true ]; then
    echo "Dry run - command that would be executed:"
    echo "$FULL_CMD"
    exit 0
fi

echo "Running: $FULL_CMD"
echo ""

# Execute the command
exec $FULL_CMD
