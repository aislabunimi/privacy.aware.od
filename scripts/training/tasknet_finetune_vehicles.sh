#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ../..

RESULTS_DIR='results'
TASKNET_SAVE_PATH='tasknet_weights/tasknet'

EXPERIMENT_DIR='experiments_5classes' #used to store the backup results
mkdir -p $EXPERIMENT_DIR
USE_DATASET_SUBSET=0

NUM_EPOCHS_TASKNET=10
BATCH_SIZE_TASKNET=2
LR_TASKNET=4e-4

TASKNET_WEIGHTS_FW_TO_SAVE1="${TASKNET_SAVE_PATH}_${NUM_EPOCHS_TASKNET}.pt"
HALF_WAY=$((NUM_EPOCHS_TASKNET/2))
if [ "$HALF_WAY" -eq 0 ]; then
   HALF_WAY=1
fi
TASKNET_WEIGHTS_FW_TO_SAVE2="${TASKNET_SAVE_PATH}_${HALF_WAY}.pt"

#If you want to use five classes, add flag --five_classes to all commands

python3 main.py --five_classes --train_tasknet --tasknet_save_path $TASKNET_SAVE_PATH --use_dataset_subset $USE_DATASET_SUBSET --num_epochs_tasknet $NUM_EPOCHS_TASKNET --batch_size_tasknet $BATCH_SIZE_TASKNET --results_dir $RESULTS_DIR --lr_tasknet $LR_TASKNET
mkdir -p "${EXPERIMENT_DIR}/tasknet"
cp $TASKNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/tasknet_5classes"
mv "${EXPERIMENT_DIR}/tasknet/tasknet_${NUM_EPOCHS_TASKNET}.pt" "${EXPERIMENT_DIR}/tasknet/tasknet.pt"
mv $TASKNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/tasknet"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/tasknet/results"
echo "Completed Tasknet_5classes training"

python3 main.py --five_classes --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --use_dataset_subset $USE_DATASET_SUBSET --batch_size_tasknet 1 --results_dir $RESULTS_DIR --val_forward_batch1 --train_tasknet
mv $RESULTS_DIR "${EXPERIMENT_DIR}/tasknet/validation_coco"
echo "Completed Tasknet_5classes Validation with Batch Size 1"

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --test_tasknet --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/tasknet/validation_pascal_voc"
echo "Completed Tasknet_5classes test"

mv $TASKNET_WEIGHTS_FW_TO_SAVE1 "${TASKNET_SAVE_PATH}.pt" #rename needed for other scripts
