#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ..

RESULTS_DIR='results'
SAVE_DIR='plotted_results'
UNET_NAME='model' #must be only the last part of the variable above

NUM_EPOCHS_UNET_FORWARD=50
NUM_EPOCHS_UNET_BACKWARD=80

UNET_FW="${UNET_NAME}_fw_${NUM_EPOCHS_UNET_FORWARD}.pt"
UNET_BW="${UNET_NAME}_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt"

EXPERIMENT_DIR='experiments' #used to store the backup results
mkdir -p $EXPERIMENT_DIR

TASKNET="${EXPERIMENT_DIR}/tasknet/tasknet_1class.pt"

PASCAL_SET="dataset/pascalVOC2012/images"

#If you want to use five classes, add flag --five_classes to all commands

python3 plot_results.py --tasknet_weights_load $TASKNET --results_dir $RESULTS_DIR --save_dir $SAVE_DIR --plot_single_image --plot_tasknet --pascal_dataset $PASCAL_SET
mv $SAVE_DIR "${EXPERIMENT_DIR}/tasknet/examples"
echo "Completed Tasknet example saving"

python3 plot_results.py --tasknet_weights_load $TASKNET --unet_fw_weights_load "${EXPERIMENT_DIR}/all_proposals/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/all_proposals/backward/${UNET_BW}" --results_dir $RESULTS_DIR --save_dir $SAVE_DIR --plot_single_image --plot_fw --pascal_dataset $PASCAL_SET
mv $SAVE_DIR "${EXPERIMENT_DIR}/all_proposals/forward/examples_fw"
echo "Completed all_proposals forward example saving"

python3 plot_results.py --tasknet_weights_load $TASKNET --unet_fw_weights_load "${EXPERIMENT_DIR}/all_proposals/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/all_proposals/backward/${UNET_BW}" --results_dir $RESULTS_DIR --save_dir $SAVE_DIR --plot_single_image --plot_bw --pascal_dataset $PASCAL_SET
mv $SAVE_DIR "${EXPERIMENT_DIR}/all_proposals/backward/examples_bw"
echo "Completed all_proposals backward example saving"

for i in "4 4" "3 3" "2 2" "1 1" "4 3" "4 2" "4 1" "3 4" "3 2" "3 1" "2 4" "2 3" "2 1" "1 4" "1 3" "1 2" "1 0" "2 0" "3 0" "4 0"; do
   a=( $i )
   PROP_POS=${a[0]}
   PROP_NEG=${a[1]}
   
   python3 plot_results.py --tasknet_weights_load $TASKNET --unet_fw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward/${UNET_BW}" --results_dir $RESULTS_DIR --save_dir $SAVE_DIR --plot_single_image --plot_fw --pascal_dataset $PASCAL_SET
   mv $SAVE_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward/examples_fw"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg forward example saving"
   
   python3 plot_results.py --tasknet_weights_load $TASKNET --unet_fw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward/${UNET_BW}" --results_dir $RESULTS_DIR --save_dir $SAVE_DIR --plot_single_image --plot_bw --pascal_dataset $PASCAL_SET
   mv $SAVE_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward/examples_bw"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg backward example saving"
done
