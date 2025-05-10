#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ../..

RESULTS_DIR='results'
TASKNET_SAVE_PATH='tasknet_weights/tasknet'

EXPERIMENT_DIR='experiments_1class' #used to store the backup results
mkdir -p $EXPERIMENT_DIR
USE_DATASET_SUBSET=0

NUM_EPOCHS_TASKNET=10
BATCH_SIZE_TASKNET=8
LR_TASKNET=1e-3

TASKNET_WEIGHTS_FW_TO_SAVE1="${TASKNET_SAVE_PATH}_${NUM_EPOCHS_TASKNET}.pt"
TASKNET_LOAD="${EXPERIMENT_DIR}/tasknet/tasknet.pt"
HALF_WAY=$((NUM_EPOCHS_TASKNET/2))
if [ "$HALF_WAY" -eq 0 ]; then
   HALF_WAY=1
fi
TASKNET_WEIGHTS_FW_TO_SAVE2="${TASKNET_SAVE_PATH}_${HALF_WAY}.pt"

python3 main.py --val_forward_batch1 --tasknet_get_indoor_AP --tasknet_weights_load $TASKNET_LOAD --use_dataset_subset $USE_DATASET_SUBSET --batch_size_tasknet 1 --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/tasknet/indoor_validation_coco"
echo "Completed Tasknet Indoor Validation with Batch Size 1"

python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --test_tasknet --tasknet_weights_load $TASKNET_LOAD --batch_size_tasknet 1 --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/tasknet/validation_pascalvoc"
echo "Completed Tasknet test"
