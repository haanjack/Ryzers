# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import subprocess
import shutil
from typing import Optional


class ContainerEngine:
    """
    A utility class to provide a container engine agnostic interface.

    Detects and uses podman if available, otherwise falls back to docker.
    Automatically handles Docker Hub registry URLs and image name normalization.
    """

    _instance: Optional['ContainerEngine'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ContainerEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._engine = self._detect_engine()
        self._initialized = True

    @staticmethod
    def _detect_engine() -> str:
        """
        Detect which container engine is available.

        Priority:
        1. podman (if available)
        2. docker (if available)

        Returns:
            str: The name of the available container engine ('podman' or 'docker')

        Raises:
            RuntimeError: If neither podman nor docker is available
        """
        # Check for podman first
        if shutil.which("podman") is not None:
            print("Using podman as container engine")
            return "podman"

        # Fall back to docker
        if shutil.which("docker") is not None:
            print("Using docker as container engine")
            return "docker"

        raise RuntimeError(
            "Neither podman nor docker is installed. "
            "Please install one of them to continue."
        )

    @property
    def engine(self) -> str:
        """Get the current container engine name."""
        return self._engine

    def get_command(self) -> str:
        """
        Get the command string for the detected engine.

        Returns:
            str: The command to use ('podman' or 'docker')
        """
        return self._engine

    def _normalize_image_name(self, image_name: str) -> str:
        """
        Normalize image name for the detected container engine.

        For podman, adds docker.io prefix to Docker Hub images if not already present.
        For docker, returns the name as-is (docker.io is implicit).

        Examples:
            - "ubuntu" -> podman: "docker.io/library/ubuntu", docker: "ubuntu"
            - "rocm/pytorch:rocm7" -> podman: "docker.io/rocm/pytorch:rocm7", docker: "rocm/pytorch:rocm7"
            - "docker.io/library/ubuntu" -> podman: "docker.io/library/ubuntu", docker: "docker.io/library/ubuntu"
            - "quay.io/image" -> podman: "quay.io/image", docker: "quay.io/image"

        Args:
            image_name (str): The image name to normalize

        Returns:
            str: The normalized image name
        """
        if self._engine != "podman":
            return image_name

        # If already has a registry (contains dots or is already docker.io), return as-is
        if ":" in image_name.split("/")[0] or image_name.startswith("docker.io"):
            return image_name

        # If there's a slash and it looks like a registry (contains dot), return as-is
        if "/" in image_name:
            parts = image_name.split("/")
            if "." in parts[0]:  # Registry server
                return image_name
            # This is a Docker Hub user/repo, add docker.io prefix
            return f"docker.io/{image_name}"

        # Single name like "ubuntu", add docker.io/library/ prefix
        return f"docker.io/library/{image_name}"

    def build(self, **kwargs) -> str:
        """
        Generate a build command string.

        Args:
            tag (str): Image tag
            buildflags (str): Build flags
            base_image (str): Base image to use
            path (str): Path to build context

        Returns:
            str: The build command string
        """
        tag = kwargs.get('tag', '')
        buildflags = kwargs.get('buildflags', '')
        base_image = kwargs.get('base_image', '')
        path = kwargs.get('path', '.')

        # Normalize the base image name for the container engine
        normalized_base_image = self._normalize_image_name(base_image)

        return (
            f"{self._engine} build -t {tag} {buildflags} "
            f"--build-arg BASE_IMAGE={normalized_base_image} {path}"
        )

    def run(self, runflags: str, container_name: str) -> str:
        """
        Generate a run command string.

        Args:
            runflags (str): Container run flags
            container_name (str): Container/image name

        Returns:
            str: The run command string
        """
        # Normalize the container/image name for the container engine
        normalized_container_name = self._normalize_image_name(container_name)

        return f"{self._engine} run {runflags} {normalized_container_name}"


def get_container_engine() -> ContainerEngine:
    """
    Get the singleton container engine instance.

    Returns:
        ContainerEngine: The container engine singleton
    """
    return ContainerEngine()
