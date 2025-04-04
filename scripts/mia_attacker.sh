#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ..

RESULTS_DIR='results'
UNET_SAVE_PATH='model_weights/model'
UNET_NAME='model' #must be only the last part of the variable above
TASKNET_WEIGHTS_LOAD='experiments/tasknet/tasknet.pt'
UNET_WEIGHTS_LOAD_FW="${UNET_SAVE_PATH}_fw_50.pt"
NUM_EPOCHS_UNET_BACKWARD=80
UNET_WEIGHTS_LOAD_BW="${UNET_SAVE_PATH}_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt"

UNET_BW="${UNET_NAME}_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt"

EXPERIMENT_DIR='experiments' #used to store the backup results
mkdir -p $EXPERIMENT_DIR

USE_DATASET_SUBSET=0
BATCH_SIZE_UNET=4
LR_UNET=5e-4


UNET_WEIGHTS_BW_TO_SAVE1="${UNET_SAVE_PATH}_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt"
HALF_WAY=$((NUM_EPOCHS_UNET_BACKWARD/2))
if [ "$HALF_WAY" -eq 0 ]; then
   HALF_WAY=1
fi
UNET_WEIGHTS_BW_TO_SAVE2="${UNET_SAVE_PATH}_bw_${HALF_WAY}.pt"

#If you want to use five classes, add flag --five_classes to all commands

# Generate the obfuscated dataset
python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load ${EXPERIMENT_DIR}/all_proposals/forward/model_fw_50.pt

# Train backward with mia
python3 main.py --mia --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --lr_unet $LR_UNET --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --train_model_backward
python3 main.py --mia --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --results_dir $RESULTS_DIR --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
mkdir -p "${EXPERIMENT_DIR}/all_proposals/backward_mia"
mv $UNET_WEIGHTS_BW_TO_SAVE1 "${EXPERIMENT_DIR}/all_proposals/backward_mia"
mv $UNET_WEIGHTS_BW_TO_SAVE2 "${EXPERIMENT_DIR}/all_proposals/backward_mia"
#rm -rf $RESULTS_DIR "${EXPERIMENT_DIR}/all_proposals/backward/${RESULTS_DIR}_bw"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/all_proposals/backward_mia/${RESULTS_DIR}_bw"
echo "Completed all_proposals backward experiment and copied results to ${EXPERIMENT_DIR} folder"

python3 main.py --mia --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/all_proposals/forward/model_fw_50.pt" --unet_bw_weights_load "${EXPERIMENT_DIR}/all_proposals/backward_mia/model_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt" --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/all_proposals/test_results_mia"
echo "Completed all_proposals test"

for i in "4 4" "3 3" "2 2" "1 1"; do
   a=( $i )
   PROP_POS=${a[0]}
   PROP_NEG=${a[1]}
   
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load ${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward/model_fw_50.pt

   python3 main.py --mia --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --lr_unet $LR_UNET --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --train_model_backward
   python3 main.py --mia --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --results_dir $RESULTS_DIR --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
   mkdir -p "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward_mia"
   mv $UNET_WEIGHTS_BW_TO_SAVE1 "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward_mia"
   mv $UNET_WEIGHTS_BW_TO_SAVE2 "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward_mia"

   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward_mia/${RESULTS_DIR}_bw"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg backward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --mia --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward/model_fw_50.pt" --unet_bw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward_mia/model_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt" --results_dir $RESULTS_DIR
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/test_results"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg test"
done
