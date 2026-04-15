#!/bin/bash

HOME_LOC=/path/to/your/home/PepTune
SCRIPT_LOC=$HOME_LOC/src
LOG_LOC=$HOME_LOC/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='unconditional'
PYTHON_EXECUTABLE=python

# ===================================================================
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate peptune

$PYTHON_EXECUTABLE $SCRIPT_LOC/generate_unconditional.py >> ${DATE}_${SPECIAL_PREFIX}_generate.log 2>&1

conda deactivate