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

# Test 3: Run a quick training test with ACT (baseline test)
echo ""
echo "Test 3: Running quick ACT training test..."
# Patch test training script to run quickly
sed -i 's/training_steps = 5000/training_steps = 2/' /ryzers/lerobot/examples/training/train_policy.py
sed -i 's/batch_size=64/batch_size=4/' /ryzers/lerobot/examples/training/train_policy.py

# Run Test
python /ryzers/lerobot/examples/training/train_policy.py

echo ""
echo "========================================"
echo "All tests passed!"
echo "========================================"