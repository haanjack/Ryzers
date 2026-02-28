# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import subprocess
import yaml
import os

class DockerRunner:
    """
    A class to execute Docker container scripts.

    Attributes:
        container_name (str): The name of the container.
        script_name (str): The name of the bash script to run the docker image.
        distributed (bool): Whether to enable distributed training.
        nproc_per_node (int): Number of processes per node.
        nnodes (int): Number of nodes for multi-node training.
        node_rank (int): Rank of this node.
        master_addr (str): Master node address.
        master_port (int): Master node port.
        policy (str): Policy type for training (act or groot).
    """

    def __init__(self, container_name = None, docker_cmd=None, script_name: str = None,
                 distributed=None, nproc_per_node=None, nnodes=1, node_rank=0,
                 master_addr="localhost", master_port=29500, policy="act"):
        """
        Initializes the DockerRunner with the container name and optional script name.

        Args:
            container_name (str): The name of the container.
            script_name (str, optional): The name of the script to execute. Defaults to None.
            distributed (bool, optional): Enable distributed training (auto-detected by default)
            nproc_per_node (int, optional): Number of processes per node (default: auto-detect all GPUs)
            nnodes (int): Number of nodes for multi-node training
            node_rank (int): Rank of this node (0 for master, 1+ for workers)
            master_addr (str): Master node address for distributed training
            master_port (int): Master node port for distributed training
            policy (str): Policy type for training (act or groot)
        """
        self.container_name = self.get_last_container_name() if container_name is None else container_name
        self.script_name = f"ryzers.run.{self.container_name}.sh" if script_name is None else script_name
        self.docker_cmdstr = docker_cmd if docker_cmd is not None else ""
        self.distributed = distributed
        self.nproc_per_node = nproc_per_node
        self.nnodes = nnodes
        self.node_rank = node_rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.policy = policy


    def __call__(self):
        """
        Executes the script for the specified container.

        The script to be executed is named 'ryzers.run.<container_name>.sh'.
        """
        # Check if the script exists
        if not os.path.exists(self.script_name):
            raise FileNotFoundError(f"Script {self.script_name} not found.")
        
        # Execute the script
        try:

            result = subprocess.run(
                ["bash", self.script_name, self.docker_cmdstr], check=True
            )
            print(f"Script output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing script {self.script_name}: {e.stderr}")
            raise

    def build_runscript(self, runflags, docker_cmd=""):
        """
        Generates a bash script that combines Docker CLI flags from package config.yaml files.

        Args:
            runflags (str): The Docker run flags.

        Returns:
            str: The path to the generated bash script.
        """
        # Build distributed training environment variables
        dist_env = ""
        if self.distributed is not None or self.nproc_per_node is not None:
            dist_value = '1' if self.distributed else '0'
            dist_env = f""" \\
    -e DISTRIBUTED={dist_value} \\
    -e NNODES={self.nnodes} \\
    -e NODE_RANK={self.node_rank} \\
    -e MASTER_ADDR={self.master_addr} \\
    -e MASTER_PORT={self.master_port}"""
            if self.nproc_per_node:
                dist_env += f" \\\n    -e NPROC_PER_NODE={self.nproc_per_node}"

        # Add policy environment variable
        policy_env = f" -e POLICY_TYPE={self.policy}"

        # Generate the bash script
        script_content = f"""#!/bin/bash
# Auto-generated script to run Docker with combined flags

# Enable X11 forwarding
xhost +local:docker

docker run {runflags}{dist_env}{policy_env} {self.container_name} $1
"""

        # Write the script to the specified file
        with open(self.script_name, 'w') as script_file:
            script_file.write(script_content)

        # Make the script executable
        os.chmod(self.script_name, 0o755)

        print(f"\nTo run this docker: ")
        print(f"# Run last ryzer docker built:")
        print(f"ryzers run [CMD_OVERRIDE] # will run last ryzer docker built.\n")
        print(f"# Run this ryzer docker by name:")
        print(f"ryzers run --name {self.container_name} [CMD_OVERRIDE]\n")

        print("\nTo inspect the docker run call, see contents of the script file: ")
        print(f"cat {self.script_name}")

    def get_last_container_name(self):
        last_built_file = os.path.join(os.path.dirname(__file__), '_ryzers.yaml')
        if os.path.exists(last_built_file):
            with open(last_built_file, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('last_built_image')
        else:
            raise FileNotFoundError("Please run 'ryzer build' succesfully first if not using the --name flag to get the last built container. Or use the --name flag to target a named container")


# Example usage
if __name__ == "__main__":
    runner = DockerRunner("my_container")
    try:
        runner()  # Executes 'ryzer.run.my_container.sh'
    except Exception as e:
        print(f"Execution failed: {e}")
