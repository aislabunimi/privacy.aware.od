#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
RESULTS_DIR='results'
UNET_SAVE_PATH='model_weights/model'
UNET_NAME='model' #must be only the last part of the variable above
TASKNET_WEIGHTS_LOAD='tasknet_weights/tasknet_1norm_myresize.pt'

NUM_EPOCHS_UNET_FORWARD=50
UNET_WEIGHTS_LOAD_FW="${UNET_SAVE_PATH}_fw_${NUM_EPOCHS_UNET_FORWARD}.pt"
NUM_EPOCHS_UNET_BACKWARD=80
UNET_WEIGHTS_LOAD_BW="${UNET_SAVE_PATH}_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt"

UNET_FW="${UNET_NAME}_fw_${NUM_EPOCHS_UNET_FORWARD}.pt"
UNET_BW="${UNET_NAME}_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt"

EXPERIMENT_DIR='experiments' #used to store the backup results
mkdir -p $EXPERIMENT_DIR

USE_DATASET_SUBSET=0
BATCH_SIZE_UNET=4

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

python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --not_use_custom_filter_prop
mkdir -p "${EXPERIMENT_DIR}/all_proposals/forward"
cp $UNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/all_proposals/forward"
mv $UNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/all_proposals/forward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/all_proposals/forward/${RESULTS_DIR}_fw"
echo "Completed all_proposals forward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW

python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --train_model_backward
python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --results_dir $RESULTS_DIR --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
mkdir -p "${EXPERIMENT_DIR}/all_proposals/backward"
mv $UNET_WEIGHTS_BW_TO_SAVE1 "${EXPERIMENT_DIR}/all_proposals/backward"
mv $UNET_WEIGHTS_BW_TO_SAVE2 "${EXPERIMENT_DIR}/all_proposals/backward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/all_proposals/backward/${RESULTS_DIR}_bw"
echo "Completed all_proposals backward experiment and copied results to ${EXPERIMENT_DIR} folder"

python3 main.py --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/all_proposals/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/all_proposals/backward/${UNET_BW}" --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/all_proposals/test_results"
echo "Completed all_proposals test"

for i in "4 4" "3 3" "2 2" "1 1" "4 3" "4 2" "4 1" "3 4" "3 2" "3 1" "2 4" "2 3" "2 1" "1 4" "1 3" "1 2" "1 0" "2 0" "3 0" "4 0"; do
   a=( $i )
   PROP_POS=${a[0]}
   PROP_NEG=${a[1]}
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --prop_pos $PROP_POS --prop_neg $PROP_NEG
   mkdir -p "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward"
   cp $UNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward"
   mv $UNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward/${RESULTS_DIR}_fw"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg forward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW

   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --train_model_backward
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --results_dir $RESULTS_DIR --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
   mkdir -p "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE1 "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE2 "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward/${RESULTS_DIR}_bw"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg backward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward/${UNET_BW}" --results_dir $RESULTS_DIR
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/test_results"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg test"
done
