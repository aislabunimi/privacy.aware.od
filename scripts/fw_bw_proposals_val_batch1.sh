#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ..

RESULTS_DIR='results'
UNET_SAVE_PATH='model_weights/model'
UNET_NAME='model' #must be only the last part of the variable above
TASKNET_WEIGHTS_LOAD='tasknet_weights/tasknet.pt'

NUM_EPOCHS_UNET_FORWARD=50
UNET_WEIGHTS_LOAD_FW="${UNET_SAVE_PATH}_fw_${NUM_EPOCHS_UNET_FORWARD}.pt"
NUM_EPOCHS_UNET_BACKWARD=80
UNET_WEIGHTS_LOAD_BW="${UNET_SAVE_PATH}_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt"

UNET_FW="${UNET_NAME}_fw_${NUM_EPOCHS_UNET_FORWARD}.pt"
UNET_BW="${UNET_NAME}_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt"

EXPERIMENT_DIR='experiments' #used to store the backup results
mkdir -p $EXPERIMENT_DIR

USE_DATASET_SUBSET=0
BATCH_SIZE_UNET=1

UNET_WEIGHTS_FW_TO_SAVE1="${UNET_SAVE_PATH}_fw_${NUM_EPOCHS_UNET_FORWARD}.pt"
HALF_WAY=$((NUM_EPOCHS_UNET_FORWARD/2))
if [ "$HALF_WAY" -eq 0 ]; then
   HALF_WAY=1
fi
UNET_WEIGHTS_FW_TO_SAVE2="${UNET_SAVE_PATH}_fw_${HALF_WAY}.pt"

UNET_WEIGHTS_BW_TO_SAVE1="${UNET_SAVE_PATH}_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt"
HALF_WAY=$((NUM_EPOCHS_UNET_BACKWARD/2))
if [ "$HALF_WAY" -eq 0 ]; then
   HALF_WAY=1
fi
UNET_WEIGHTS_BW_TO_SAVE2="${UNET_SAVE_PATH}_bw_${HALF_WAY}.pt"

#If you want to use five classes, add flag --five_classes to all commands

python3 main.py --tasknet_get_indoor_AP --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --use_dataset_subset $USE_DATASET_SUBSET --batch_size_tasknet 1 --results_dir $RESULTS_DIR --val_forward_batch1
mv $RESULTS_DIR "${EXPERIMENT_DIR}/tasknet/val_results_batch1"
echo "Completed Tasknet Indoor Validation with Batch Size 1"

python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --not_use_custom_filter_prop --unet_fw_weights_load "${EXPERIMENT_DIR}/all_proposals/forward/${UNET_FW}" --val_forward_batch1
mv $RESULTS_DIR "${EXPERIMENT_DIR}/all_proposals/val_results_batch1"
echo "Completed all_proposals Validation with Batch Size 1"

for i in "4 4" "3 3" "2 2" "1 1" "4 3" "4 2" "4 1" "3 4" "3 2" "3 1" "2 4" "2 3" "2 1" "1 4" "1 3" "1 2" "1 0" "2 0" "3 0" "4 0"; do
   a=( $i )
   PROP_POS=${a[0]}
   PROP_NEG=${a[1]}
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --not_use_custom_filter_prop --unet_fw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward/${UNET_FW}" --val_forward_batch1
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/val_results_batch1"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg Validation with Batch Size 1"
done
