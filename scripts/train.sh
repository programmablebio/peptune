#!/bin/bash

HOME_LOC=/path/to/your/home/PepTune
ENV_LOC=/path/to/your/envs/peptune
SCRIPT_LOC=$HOME_LOC/src
LOG_LOC=$HOME_LOC/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='11M-ablation-all-losses'
# set 3 have skip connection
PYTHON_EXECUTABLE=$ENV_LOC/bin/python

# ===================================================================
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_LOC

$PYTHON_EXECUTABLE $SCRIPT_LOC/train_peptune.py >> ${DATE}_${SPECIAL_PREFIX}_train.log 2>&1

conda deactivate