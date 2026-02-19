#!/bin/bash
# SLURM submission script for context-based SFT

# Usage: sbatch run_sft_context_preceding.sh [context_window]
# Example: sbatch run_sft_context_preceding.sh 5
# Default context_window is 3 if not specified

#SBATCH --job-name=friends_context_preceiding
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=sft_context_%j.out


CONTEXT_WINDOW=${1:-3}

echo "Starting CONTEXT SFT training with context_window=${CONTEXT_WINDOW}..."

source ~/.bashrc
conda activate friends_sft

echo "Python used:"
which python
python -c "import transformers, trl; print('transformers', transformers.__version__); print('trl', trl.__version__)"


# Timestamp for unique output directory & wandb run name
TS=$(date +%Y%m%d-%H%M%S)

python train.py \
  --data_path cleaned_data.csv \
  --output_dir outputs/granite-friends-context${CONTEXT_WINDOW}-${TS} \
  --wandb_run_name granite-sft-context${CONTEXT_WINDOW}-${TS} \
  --context_window ${CONTEXT_WINDOW} \
  --epochs 3 \
  --batch_size 1 \
  --grad_accum 1 \
  --lr 2e-4 \
  --max_seq_len 512 \
  --use_wandb \
  --wandb_project cse595-friends-granite

echo "Context-based SFT training finished."
