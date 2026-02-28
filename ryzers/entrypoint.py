# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import sys
import argparse
from .ryzer import RyzerManager
from .runner import DockerRunner

def build(base_path, name, packages, init_image):
    """
    Builds the Docker images using the specified packages path and selected packages.

    Args:
        base_path (str): The base directory to scan for Dockerfiles.
        name (str): The name of the Docker image to build.
        packages (list): List of package names to manage.
        init_image (str, optional): The initial base image to start with.
    """
    mgr = RyzerManager(base_path, name, packages, init_image)
    mgr.build()

def run(name, docker_cmd, distributed=None, no_distributed=False, nproc_per_node=None,
        nnodes=1, node_rank=0, master_addr="localhost", master_port=29500):
    """
    Runs the Docker container with the specified name.

    Args:
        name (str): The name of the Docker image to run.
        docker_cmd (str): The command to run with the Docker run call
        distributed (bool, optional): Enable distributed training (auto-detected by default)
        no_distributed (bool): Disable distributed training (force single-GPU)
        nproc_per_node (int, optional): Number of processes per node (default: auto-detect all GPUs)
        nnodes (int): Number of nodes for multi-node training
        node_rank (int): Rank of this node (0 for master, 1+ for workers)
        master_addr (str): Master node address for distributed training
        master_port (int): Master node port for distributed training
    """
    # If no_distributed is set, distributed is False
    if no_distributed:
        distributed = False
    runner = DockerRunner(name, docker_cmd, distributed=distributed, nproc_per_node=nproc_per_node,
                          nnodes=nnodes, node_rank=node_rank, master_addr=master_addr, master_port=master_port)
    runner()

def main():
    """
    Main entry point for the command-line tool.

    Parses command-line arguments and executes the appropriate build or run command.
    """
    default_base_path = os.getcwd()

    parser = argparse.ArgumentParser(description="Command-line tool for building and running Ryzen AI Dockers.")
    
    # Define subcommands
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    # 'build' command
    build_parser = subparsers.add_parser("build", help="Build the project")
    build_parser.add_argument("--base_path", default=default_base_path, help=f"Base path for the project (default: CWD)")
    build_parser.add_argument("--name", default="ryzerdocker", help="Name of the docker image to build")
    build_parser.add_argument("dockerfiles", nargs="*", help="List of Dockerfiles to combine for the build (optional)")
    build_parser.add_argument("--init_image", default=None, help=f"Initial base image to start with (default: None)")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run the project")
    run_parser.add_argument("--name", default=None, help="Name of the docker image to run")
    run_parser.add_argument("docker_cmd", nargs="?", default="", help="Overwrite the Docker CMD to run this command (optional)")

    # Distributed training arguments
    run_parser.add_argument("--distributed", action="store_true", default=None,
                            help="Enable distributed training (auto-detected by default)")
    run_parser.add_argument("--no-distributed", action="store_true", default=False,
                            help="Disable distributed training (force single-GPU)")
    run_parser.add_argument("--nproc-per-node", type=int, default=None,
                            help="Number of processes per node (default: auto-detect all GPUs)")
    run_parser.add_argument("--nnodes", type=int, default=1,
                            help="Number of nodes for multi-node training")
    run_parser.add_argument("--node-rank", type=int, default=0,
                            help="Rank of this node (0 for master, 1+ for workers)")
    run_parser.add_argument("--master-addr", default="localhost",
                            help="Master node address for distributed training")
    run_parser.add_argument("--master-port", type=int, default=29500,
                            help="Master node port for distributed training")

    # Parse the arguments
    args = parser.parse_args()

    if args.command == "build":
        build(args.base_path, args.name, args.dockerfiles, args.init_image)
    elif args.command == "run":
        run(args.name, args.docker_cmd, distributed=args.distributed, no_distributed=args.no_distributed,
            nproc_per_node=args.nproc_per_node, nnodes=args.nnodes, node_rank=args.node_rank,
            master_addr=args.master_addr, master_port=args.master_port)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)
