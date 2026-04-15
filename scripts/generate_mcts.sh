#!/bin/bash

HOME_LOC=/path/to/your/home/PepTune
SCRIPT_LOC=$HOME_LOC/src
LOG_LOC=$HOME_LOC/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='mcts'
PYTHON_EXECUTABLE=python

# ===================================================================
# Default parameters (can be overridden by command line arguments)
# Available proteins: amhr, tfr, gfap, glp1, glast, ncam, cereblon, ligase, skp2, p53, egfp
PROT_NAME1=${1:-"gfap"}
PROT_NAME2=${2:-""}
MODE=${3:-"2"}
MODEL=${4:-"mcts"}
LENGTH=${5:-"100"}
EPOCH=${6:-"7"}
CKPT=$HOME_LOC/checkpoints/epoch13-new-tokenizer.ckpt

# ===================================================================
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate peptune

mkdir -p "${HOME_LOC}/${PROT_NAME1}"
mkdir -p "${LOG_LOC}"

echo "Running MCTS generation with parameters:"
echo "  Protein Name 1: $PROT_NAME1"
echo "  Protein Name 2: $PROT_NAME2"
echo "  Mode: $MODE"
echo "  Model: $MODEL"
echo "  Length: $LENGTH"
echo "  Epoch: $EPOCH"

# Build Hydra override arguments
mkdir -p "${LOG_LOC}"

HYDRA_ARGS="+prot_name1=$PROT_NAME1 ++mode=$MODE +model_type=$MODEL +length=$LENGTH +epoch=$EPOCH"
if [ -n "$PROT_NAME2" ]; then
    HYDRA_ARGS="$HYDRA_ARGS +prot_name2=$PROT_NAME2"
fi

cd "$SCRIPT_LOC"

# Run the MCTS generation script with Hydra overrides
$PYTHON_EXECUTABLE $SCRIPT_LOC/generate_mcts.py \
    --config-path "$SCRIPT_LOC" \
    --config-name config \
    base_path="$HOME_LOC" \
    eval.checkpoint_path="$CKPT" \
    $HYDRA_ARGS >> ${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}_generate.log 2>&1

echo "Generation complete. Check logs at: ${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}_generate.log"

conda deactivate
