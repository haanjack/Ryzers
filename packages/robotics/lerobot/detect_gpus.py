#!/usr/bin/env python3
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
GPU Detection Utility for AMD ROCm GPUs.

This script detects available AMD GPUs using rocm-smi and provides
information for distributed training setup.

Usage:
    python3 detect_gpus.py --count      # Print just the GPU count
    python3 detect_gpus.py --info       # Print detailed GPU info
    python3 detect_gpus.py --setup-env  # Print environment setup commands
"""

import argparse
import os
import subprocess
import sys


def run_command(cmd):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip(), result.returncode
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "", 1


def detect_gpus_rocm_smi():
    """Detect GPUs using rocm-smi."""
    output, returncode = run_command("rocm-smi --showid")
    if returncode != 0 or not output:
        return 0, []

    # Parse GPU IDs from rocm-smi output
    gpu_ids = []
    for line in output.split("\n"):
        if line.startswith("GPU") and ":" in line:
            # Extract GPU index from lines like "GPU[0]		: GPU 0"
            try:
                idx = line.split("[")[1].split("]")[0]
                gpu_ids.append(int(idx))
            except (IndexError, ValueError):
                continue

    return len(gpu_ids), gpu_ids


def detect_gpus_hip_visible():
    """Detect GPUs from HIP_VISIBLE_DEVICES environment variable."""
    hip_devices = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if hip_devices:
        try:
            # HIP_VISIBLE_DEVICES can be comma-separated list
            device_ids = [int(x.strip()) for x in hip_devices.split(",") if x.strip()]
            return len(device_ids), device_ids
        except ValueError:
            pass
    return 0, []


def detect_gpus_render_devices():
    """Detect GPUs by counting /dev/dri/renderD* devices."""
    render_devices = []
    if os.path.exists("/dev/dri"):
        for entry in os.listdir("/dev/dri"):
            if entry.startswith("renderD"):
                try:
                    # renderD128 is GPU 0, renderD129 is GPU 1, etc.
                    idx = int(entry.replace("renderD", "")) - 128
                    if idx >= 0:
                        render_devices.append(idx)
                except ValueError:
                    continue
    return len(render_devices), sorted(render_devices)


def get_gpu_info():
    """Get detailed GPU information using rocm-smi."""
    info = []
    output, returncode = run_command("rocm-smi --showallinfo")
    if returncode == 0 and output:
        info.append(output)
    else:
        # Fallback to basic info
        output, returncode = run_command("rocm-smi")
        if returncode == 0 and output:
            info.append(output)
    return "\n".join(info) if info else "No GPU info available"


def detect_gpus():
    """
    Detect available GPUs using multiple methods.

    Returns:
        tuple: (gpu_count, gpu_ids)
    """
    # Method 1: rocm-smi (most reliable)
    count, ids = detect_gpus_rocm_smi()
    if count > 0:
        return count, ids

    # Method 2: HIP_VISIBLE_DEVICES
    count, ids = detect_gpus_hip_visible()
    if count > 0:
        return count, ids

    # Method 3: /dev/dri/renderD* devices
    count, ids = detect_gpus_render_devices()
    if count > 0:
        return count, ids

    # Default: assume 1 GPU if detection fails
    return 1, [0]


def main():
    parser = argparse.ArgumentParser(
        description="Detect AMD GPUs for distributed training setup"
    )
    parser.add_argument(
        "--count", action="store_true", help="Print only the GPU count"
    )
    parser.add_argument(
        "--info", action="store_true", help="Print detailed GPU information"
    )
    parser.add_argument(
        "--setup-env", action="store_true",
        help="Print environment variable setup commands"
    )
    parser.add_argument(
        "--default", type=int, default=1,
        help="Default GPU count to return if detection fails (default: 1)"
    )

    args = parser.parse_args()

    # If no specific output mode is requested, default to count
    if not (args.count or args.info or args.setup_env):
        args.count = True

    count, ids = detect_gpus()

    # If detection failed, use default
    if count == 0:
        count = args.default
        ids = list(range(count))

    if args.count:
        print(count)
        return 0

    if args.info:
        print(f"Detected {count} GPU(s): {ids}")
        print()
        print(get_gpu_info())
        return 0

    if args.setup_env:
        print(f"# Detected {count} GPU(s): {ids}")
        print(f"export WORLD_SIZE={count}")
        if count > 1:
            print(f"export HIP_VISIBLE_DEVICES={','.join(map(str, ids))}")
        print("export MASTER_ADDR=localhost")
        print("export MASTER_PORT=29500")
        if count > 1:
            print(f"# Run distributed training with {count} GPUs:")
            print(f"# torchrun --nproc_per_node={count} your_script.py")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
