# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

RUN apt-get update && apt-get install -y git cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config ffmpeg speech-dispatcher rocm-dev hipcc python3-pybind11 pybind11-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git config --global credential.helper store

# Install flash attention for ROCm (required for GR00T)
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
ENV BUILD_TYPE="rocm"
RUN git clone --recursive https://github.com/ROCm/flash-attention /opt/flash-attention && \
    cd /opt/flash-attention && \
    git checkout main_perf && \
    python setup.py install

# Install RCCL for distributed training on ROCm
RUN apt-get update && apt-get install -y librccl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy distributed training scripts
COPY detect_gpus.py /ryzers/detect_gpus.py
COPY train_policy_distributed.py /ryzers/train_policy_distributed.py
RUN chmod +x /ryzers/train_policy_distributed.py

# Install latest lerobot with GR00T support
RUN git clone https://github.com/huggingface/lerobot.git /opt/lerobot \
 && cd /opt/lerobot \
 && sed -i '/"torch[>=<~ ]*/d' pyproject.toml \
 && sed -i '/"torchvision[>=<~ ]*/d' pyproject.toml \
 && pip install -e ".[aloha,pusht,dynamixel,feetech,pi]"

# Patch GRoOT to pass device_map during from_pretrained to prevent meta tensor issues
COPY patch_groot.py /ryzers/patch_groot.py
RUN python3 /ryzers/patch_groot.py

# Install dependencies - use latest transformers with GRoOT compatibility patch
RUN pip install --no-cache-dir "numpy<2" pytest huggingface_hub num2words peft && \
    pip install --no-cache-dir --upgrade transformers

COPY test.sh /ryzers/test_lerobot.sh
RUN chmod +x /ryzers/test_lerobot.sh

# Create an entrypoint to autmatically login to Hugging Face on docker start
RUN printf '%s\n' \
  '#!/bin/bash' \
  'if [ -n "$HF_TOKEN" ]; then' \
  '  hf auth login --token "$HF_TOKEN" --add-to-git-credential' \
  'fi' \
  'export HF_USER=$(hf auth whoami | head -n 1 | cut -d" " -f3)' \
  'echo "🚀 Logged in as $HF_USER"' \
  '# Apply runtime patches for GR00T compatibility' \
  'python3 /ryzers/patch_groot.py 2>/dev/null || true' \
  'if [ $# -eq 0 ]; then' \
  '  exec /ryzers/test_lerobot.sh' \
  'elif [[ "$1" == -* ]]; then' \
  '  exec /ryzers/test_lerobot.sh "$@"' \
  'else' \
  '  exec "$@"' \
  'fi' \
  > /ryzers/entrypoint.sh \
  && chmod +x /ryzers/entrypoint.sh

ENTRYPOINT ["/ryzers/entrypoint.sh"]
CMD /ryzers/test_lerobot.sh
