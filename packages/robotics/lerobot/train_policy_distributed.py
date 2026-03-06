#!/usr/bin/env python3
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
DDP-enabled training script for LeRobot policies.
Based on lerobot/examples/training/train_policy.py with distributed training support.
"""

import argparse
import os
from pathlib import Path
import warnings

# Suppress torchvision video deprecation warnings (not critical for training)
warnings.filterwarnings("ignore", message="The video decoding and encoding capabilities")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.factory import make_pre_post_processors


def setup_distributed():
    """Initialize distributed training if running under torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return True, rank, world_size, local_rank

    return False, 0, 1, 0


def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DDP-enabled LeRobot training")
    parser.add_argument("--use-compile", action="store_true", help="Enable torch.compile optimization")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--training-steps", type=int, default=5000, help="Total training steps (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--policy", type=str, default="act", choices=["act", "diffusion", "groot"], help="Policy type (default: act)")
    parser.add_argument("--dataset", type=str, default="lerobot/pusht", help="Dataset repo ID (default: lerobot/pusht)")

    # Filter out torchrun/DDP arguments before parsing
    if args is None:
        # Remove any arguments that aren't recognized
        import sys
        known_args, unknown = parser.parse_known_args()
    else:
        known_args = parser.parse_args(args)

    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"  Distributed: {is_distributed}")
        print(f"  World Size:  {world_size}")
        print(f"  Rank:        {rank}")
        print(f"  Local Rank:  {local_rank}")
        print(f"{'='*60}\n")

    # Create output directory
    output_directory = Path(f"outputs/train/{known_args.policy}_{known_args.dataset.split('/')[-1]}")
    if rank == 0:
        output_directory.mkdir(parents=True, exist_ok=True)

    if is_distributed:
        dist.barrier()

    # Setup device
    device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")

    # Training hyperparameters (from arguments or defaults)
    training_steps = known_args.training_steps
    batch_size = known_args.batch_size
    learning_rate = known_args.learning_rate
    log_freq = 100
    num_workers = 4

    # Optimization flags (from arguments or defaults)
    use_compile = known_args.use_compile
    use_amp = known_args.use_amp
    amp_dtype = torch.bfloat16  # Use bfloat16 instead of float16

    if rank == 0:
        print(f"  Optimizations:")
        print(f"    torch.compile: {use_compile}")
        print(f"    Mixed precision: {use_amp} (dtype: {amp_dtype})")
        print(f"  Policy: {known_args.policy}")
        print(f"  Dataset: {known_args.dataset}")

    # Load dataset metadata and configure policy
    dataset_metadata = LeRobotDatasetMetadata(known_args.dataset)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Select policy config and class based on policy type
    policy_type = known_args.policy.lower()
    if policy_type == "act":
        cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            device=device
        )
        policy = ACTPolicy(cfg)
    elif policy_type == "diffusion":
        cfg = DiffusionConfig(
            input_features=input_features,
            output_features=output_features,
            device=device
        )
        policy = DiffusionPolicy(cfg)
    elif policy_type == "groot":
        cfg = GrootConfig(
            input_features=input_features,
            output_features=output_features,
            device=device
        )
        policy = GrootPolicy(cfg)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    # Setup policy for training
    policy.train()
    policy.to(device)
    for module in policy.modules():
        for buf_name, buf in module.named_buffers(recurse=False):
            if buf.device != device:
                module.register_buffer(buf_name, buf.to(device))

    # Apply torch.compile if enabled (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        if rank == 0:
            print("Applying torch.compile optimization...")
        policy = torch.compile(policy, mode="reduce-overhead")

    # Wrap with DDP if distributed
    if is_distributed:
        policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # Configure delta timestamps (if supported by policy config)
    delta_timestamps = {}
    if hasattr(cfg, 'observation_delta_indices') and hasattr(cfg, 'action_delta_indices'):
        delta_timestamps = {
            "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
            "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
            "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
        }
    else:
        # Default for policies without delta indices (e.g., some versions of GRoOT)
        if rank == 0:
            print("  Note: Policy config doesn't have delta_indices, using default timestamps")
        delta_timestamps = None

    # Create dataset
    dataset = LeRobotDataset(known_args.dataset, delta_timestamps=delta_timestamps)

    # Create distributed sampler if needed
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None

    # Create optimizer and dataloader
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # Setup GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Training loop
    step = 0
    done = False

    if rank == 0:
        print("Starting training...\n")
        print(f"Optimizations:")
        print(f"  torch.compile: {use_compile and hasattr(torch, 'compile')}")
        print(f"  Mixed precision: {use_amp} (dtype: {amp_dtype if use_amp else 'N/A'})\n")

    while not done:
        if sampler is not None:
            sampler.set_epoch(step)

        for batch in dataloader:
            batch = preprocessor(batch)

            # Forward pass with automatic mixed precision (bfloat16)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                loss, _ = policy.forward(batch)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if rank == 0 and step % log_freq == 0:
                print(f"step: {step:5d} loss: {loss.item():.3f}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # Save checkpoint (rank 0 only)
    if rank == 0:
        print(f"\nSaving checkpoint to {output_directory}...")
        policy_to_save = policy.module if is_distributed else policy
        policy_to_save.save_pretrained(output_directory)
        preprocessor.save_pretrained(output_directory)
        postprocessor.save_pretrained(output_directory)
        print("✓ Training complete!")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    main()
