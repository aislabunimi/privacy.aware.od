#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ../..

RESULTS_DIR='results'


IMAGES_PATH="dataset/robot/images"


EXPERIMENT_DIR='experiments_1class'
TASKNET_WEIGHTS_LOAD=${EXPERIMENT_DIR}/tasknet/tasknet.pt
USE_DATASET_SUBSET=0
BATCH_SIZE_UNET=1

ANN_FILE='dataset/annotations/robot/giraff_person.json'

#first test tasknet plain on Pascal
python3 main.py --test_tasknet --pascal_ann_path $ANN_FILE --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --pascal_img_path $IMAGES_PATH
mkdir -p "${EXPERIMENT_DIR}/tasknet"
mv "results" "${EXPERIMENT_DIR}/tasknet/validation_robot_dataset"
echo "Completed Tasknet test"

python3 main.py --test_model --pascal_ann_path $ANN_FILE --pascal_img_path $IMAGES_PATH --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/allprop/forward/model_fw_50.pt" --unet_bw_weights_load "${EXPERIMENT_DIR}/allprop/backward/model_bw_80.pt"
#mkdir -p "${EXPERIMENT_DIR}/all_proposals"
mv "results" "${EXPERIMENT_DIR}/allprop/validation_robot_dataset"
echo "Completed all_proposals test"

for i in "4 4" "3 3" "2 2" "1 1"; do
   a=( $i )
   PROP_POS=${a[0]}
   PROP_NEG=${a[1]}
   python3 main.py --test_model --pascal_ann_path $ANN_FILE --pascal_img_path $IMAGES_PATH --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward/model_fw_50.pt" --unet_bw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward/model_bw_80.pt"
   #mkdir -p "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg"
   mv "results" "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/validation_robot_dataset"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg test"
done