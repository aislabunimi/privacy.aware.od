#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ../..

RESULTS_DIR='results'
SAVE_DIR='plotted_results'
UNET_NAME='model' #must be only the last part of the variable above
SAVE_BW='attacker_enhanced_examples'

NUM_EPOCHS_UNET_FORWARD=50
NUM_EPOCHS_UNET_BACKWARD=80

UNET_FW="${UNET_NAME}_fw_${NUM_EPOCHS_UNET_FORWARD}.pt"
UNET_BW="${UNET_NAME}_bw_${NUM_EPOCHS_UNET_BACKWARD}.pt"

EXPERIMENT_DIR='experiments_1class' #used to store the backup results
mkdir -p $EXPERIMENT_DIR

TASKNET="${EXPERIMENT_DIR}/tasknet/tasknet.pt"

PASCAL_SET="dataset/pascalVOC2012/images"

#If you want to use five classes, add flag --five_classes to all commands


python3 plot_results_final.py --plot_subset --tasknet_weights_load $TASKNET --unet_fw_weights_load "${EXPERIMENT_DIR}/allprop/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/allprop/backward_mia/${UNET_BW}" --results_dir $RESULTS_DIR --save_dir $SAVE_DIR --plot_single_image --plot_bw --pascal_dataset $PASCAL_SET
mv $SAVE_DIR "${EXPERIMENT_DIR}/allprop/backward_mia/${SAVE_BW}"
echo "Completed all_proposals backward example saving"


for i in "4 4" "3 3" "2 2" "1 1"; do
   a=( $i )
   PROP_POS=${a[0]}
   PROP_NEG=${a[1]}
   
   python3 plot_results_final.py --plot_subset --tasknet_weights_load $TASKNET --unet_fw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward_mia/${UNET_BW}" --results_dir $RESULTS_DIR --save_dir $SAVE_DIR --plot_single_image --plot_bw --pascal_dataset $PASCAL_SET
   mv $SAVE_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward_mia/${SAVE_BW}"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg backward example saving"

done