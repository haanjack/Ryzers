# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Container runtime detection and abstraction module.

Detects and provides the appropriate container runtime (Podman or Docker).
Podman is preferred if available, otherwise falls back to Docker.
"""

import subprocess
import shutil
from typing import Optional


class ContainerRuntime:
    """
    Abstraction for container runtimes (Docker or Podman).
    """

    def __init__(self):
        self._runtime: Optional[str] = None

    def detect(self) -> str:
        """
        Detect the available container runtime.

        Prefers Podman if available, otherwise falls back to Docker.

        Returns:
            str: The container runtime command ('podman' or 'docker')

        Raises:
            RuntimeError: If neither Podman nor Docker is found
        """
        if self._runtime is not None:
            return self._runtime

        # Check for podman first
        if shutil.which('podman') is not None:
            # Verify podman is working
            try:
                result = subprocess.run(
                    ['podman', 'info', '--format', '{{.Host.OS}}'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    self._runtime = 'podman'
                    return self._runtime
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        # Fall back to docker
        if shutil.which('docker') is not None:
            # Verify docker is working
            try:
                result = subprocess.run(
                    ['docker', 'info', '--format', '{{.OperatingSystem}}'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    self._runtime = 'docker'
                    return self._runtime
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        raise RuntimeError(
            "No container runtime found. Please install Podman or Docker."
        )

    @property
    def runtime(self) -> str:
        """Get the detected container runtime."""
        return self.detect()

    @property
    def is_podman(self) -> bool:
        """Check if using Podman."""
        return self.runtime == 'podman'

    @property
    def is_docker(self) -> bool:
        """Check if using Docker."""
        return self.runtime == 'docker'

    def build_cmd(self) -> str:
        """Get the build command for the runtime."""
        return self.runtime

    def run_cmd(self) -> str:
        """Get the run command for the runtime."""
        return self.runtime

    def get_run_flags(self, base_flags: str) -> str:
        """
        Adjust run flags based on the container runtime.

        Podman handles some flags differently than Docker.
        """
        if self.is_podman:
            # Podman doesn't need --group-add for video/render on most systems
            # but we keep them for compatibility
            pass
        return base_flags


# Global instance
_runtime: Optional[ContainerRuntime] = None


def get_runtime() -> ContainerRuntime:
    """Get the global container runtime instance."""
    global _runtime
    if _runtime is None:
        _runtime = ContainerRuntime()
    return _runtime


def get_runtime_name() -> str:
    """Get the name of the detected container runtime."""
    return get_runtime().runtime
