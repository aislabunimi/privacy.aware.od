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
HALF_WAY=$((NUM_EPOCHS_TASKNET/2))
if [ "$HALF_WAY" -eq 0 ]; then
   HALF_WAY=1
fi
TASKNET_WEIGHTS_FW_TO_SAVE2="${TASKNET_SAVE_PATH}_${HALF_WAY}.pt"

#If you want to use five classes, add flag --five_classes to all commands

python3 main.py --train_tasknet --tasknet_save_path $TASKNET_SAVE_PATH --use_dataset_subset $USE_DATASET_SUBSET --num_epochs_tasknet $NUM_EPOCHS_TASKNET --batch_size_tasknet $BATCH_SIZE_TASKNET --lr_tasknet $LR_TASKNET --results_dir $RESULTS_DIR
mkdir -p "${EXPERIMENT_DIR}/tasknet"
cp $TASKNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/tasknet"
mv "${EXPERIMENT_DIR}/tasknet/tasknet_${NUM_EPOCHS_TASKNET}.pt" "${EXPERIMENT_DIR}/tasknet/tasknet.pt"
mv $TASKNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/tasknet"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/tasknet/allpeople_val_results"
echo "Completed Tasknet training"

python3 main.py --val_forward_batch1 --tasknet_get_indoor_AP --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --use_dataset_subset $USE_DATASET_SUBSET --batch_size_tasknet 1 --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/tasknet/indoor_validation_coco"
echo "Completed Tasknet Indoor Validation with Batch Size 1"

python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --test_tasknet --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --batch_size_tasknet 1 --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/tasknet/validation_pascalvoc"
echo "Completed Tasknet test"

mv $TASKNET_WEIGHTS_FW_TO_SAVE1 "${TASKNET_SAVE_PATH}.pt" #rename needed for other scripts
