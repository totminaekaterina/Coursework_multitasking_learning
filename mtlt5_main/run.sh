#!/bin/bash -l

#SBATCH --partition=a100
#SBATCH --job-name=mtlt5
#SBATCH --error=/userspace/tev/gitlab/MTLT5/main/run_err.log
#SBATCH --output=/userspace/tev/gitlab/MTLT5/main/run.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

########################  КЭШИ  ########################
export TEXTATTACK_CACHE_DIR=/userspace/tev/cache/textattack
export TEXTATTACK_DISABLE_POST_INSTALL=1     # отключает post-install
mkdir -p   "$TEXTATTACK_CACHE_DIR"
chmod 755  "$TEXTATTACK_CACHE_DIR"

# export TRANSFORMERS_OFFLINE=0  # 1 - для оффлайн-режима

# Create cache directories with proper permissions
# mkdir -p "$HF_HOME" "$MPLCONFIGDIR"
# chmod -R 755 "$HF_HOME" "$MPLCONFIGDIR"

# --- DNS Workaround ---
# Если есть проблемы с DNS, раскомментируйте:
# echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf

# --- Proxy Settings ---
# Если используете прокси, раскомментируйте:
# export HTTP_PROXY="http://your-proxy:port"
# export HTTPS_PROXY="http://your-proxy:port"

# --- CUDA and Environment Setup ---
. "/userspace/tev/conda/etc/profile.d/conda.sh"
conda activate /userspace/tev/gitlab/MTLT5/.venv
export PATH="/usr/local/cuda-11/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH"

# ----------  ТОЛЬКО ПОСЛЕ ЭТОГО  ----------------------------------
export TEXTATTACK_CACHE_DIR=/userspace/tev/cache/textattack
export TEXTATTACK_DISABLE_POST_INSTALL=1        # глушим post-install
export HF_HOME=/userspace/tev/cache/hf
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p  "$TEXTATTACK_CACHE_DIR" "$HF_HOME"
chmod 755 "$TEXTATTACK_CACHE_DIR" "$HF_HOME"
# ------------------------------------------------------------------


# --- Sanity Checks ---
nvidia-smi -L
nvcc --version
python -V
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# --- Network Test ---
# echo "Testing connection to Hugging Face Hub..."
# curl -I --retry 3 --connect-timeout 20 https://huggingface.co

# --- Main Execution ---
python -u /userspace/tev/gitlab/MTLT5/main/save_checkpoint.py