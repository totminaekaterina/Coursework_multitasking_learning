#!/bin/bash -l

#SBATCH --partition=a100
#SBATCH --job-name=mtlt5_train
#SBATCH --error=/userspace/tev/gitlab/MTLT5/main/run_err.log
#SBATCH --output=/userspace/tev/gitlab/MTLT5/main/run.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# --- CUDA and Environment Setup ---
. "/userspace/tev/conda/etc/profile.d/conda.sh"
conda activate /userspace/tev/gitlab/MTLT5/.venv
export PATH="/usr/local/cuda-11/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH"

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