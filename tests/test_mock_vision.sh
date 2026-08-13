#!/bin/bash
#SBATCH --job-name=anonlib-vision-test
#SBATCH --chdir=$ANONLIB_PATH/src/anonlib
#SBATCH --output=reports/R-%x.%A_%a.out
#SBATCH --error=reports/R-%x.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=11:59:59
#SBATCH -A my_account
#SBATCH --array=0-3

# --- outputs & config ---
export ROOT=$SCRATCH/anonlib_vision_test
export SHARDS_ROOT="$ROOT/shards"
export MERGED_DIR="$ROOT/merged"
export CFG=$ANONLIB_PATH/configs/config_mock_vision.yaml

# HF cache/home
export HF_HOME=$SCRATCH/hf

mkdir -p "$SHARDS_ROOT"
mkdir -p "$MERGED_DIR"

export CMD="python $ANONLIB_PATH/src/anonlib/shard_process.py --config $CFG"

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60
  "
# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$CMD"
echo "END TIME: $(date)"
