# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import subprocess
import yaml
import os
from .container import get_container_engine

class DockerRunner:
    """
    A class to execute Docker/Podman container scripts.

    Attributes:
        container_name (str): The name of the container.
        script_name (str): The name of the bash script to run the container image.
    """

    def __init__(self, container_name = None, docker_cmd=None, script_name: str = None):
        """
        Initializes the DockerRunner with the container name and optional script name.

        Args:
            container_name (str): The name of the container.
            script_name (str, optional): The name of the script to execute. Defaults to None.
        """
        self.container_name = self.get_last_container_name() if container_name is None else container_name
        self.script_name = f"ryzers.run.{self.container_name}.sh" if script_name is None else script_name
        self.docker_cmdstr = docker_cmd if docker_cmd is not None else ""
        self.engine = get_container_engine()


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
        Generates a bash script that combines container engine CLI flags from package config.yaml files.

        Args:
            runflags (str): The container run flags.

        Returns:
            str: The path to the generated bash script.
        """
        container_cmd = self.engine.get_command()

        # Generate the bash script
        script_content = f"""#!/bin/bash
# Auto-generated script to run {container_cmd.capitalize()} with combined flags

# Enable X11 forwarding
xhost +local:{container_cmd}

{container_cmd} run {runflags} {self.container_name} $1
"""

        # Write the script to the specified file
        with open(self.script_name, 'w') as script_file:
            script_file.write(script_content)

        # Make the script executable
        os.chmod(self.script_name, 0o755)

        print(f"\nTo run this container: ")
        print(f"# Run last ryzer container built:")
        print(f"ryzers run [CMD_OVERRIDE] # will run last ryzer container built.\n")
        print(f"# Run this ryzer container by name:")
        print(f"ryzers run --name {self.container_name} [CMD_OVERRIDE]\n")

        print("\nTo inspect the container run call, see contents of the script file: ")
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
