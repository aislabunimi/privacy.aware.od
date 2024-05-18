#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
RESULTS_DIR='results'
TASKNET_WEIGHTS_LOAD='tasknet_weights/tasknet_1norm_myresize.pt'

EXPERIMENT_DIR='experiments' #used to store the backup results
USE_DATASET_SUBSET=0
BATCH_SIZE_UNET=1

#first test tasknet plain on Pascal
python3 main.py --test_tasknet --tasknet_weights_load $TASKNET_WEIGHTS_LOAD
mkdir -p "${EXPERIMENT_DIR}/tasknet"
mv "results" "${EXPERIMENT_DIR}/tasknet/test_results"
echo "Completed Tasknet test"

python3 main.py --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/all_proposals/forward/model_fw_50.pt" --unet_bw_weights_load "${EXPERIMENT_DIR}/all_proposals/backward/model_bw_80.pt"
#mkdir -p "${EXPERIMENT_DIR}/all_proposals"
mv "results" "${EXPERIMENT_DIR}/all_proposals/test_results"
echo "Completed all_proposals test"

for i in "4 4" "3 3" "2 2" "1 1" "4 3" "4 2" "4 1" "3 4" "3 2" "3 1" "2 4" "2 3" "2 1" "1 4" "1 3" "1 2" "1 0" "2 0" "3 0" "4 0"; do
   a=( $i )
   PROP_POS=${a[0]}
   PROP_NEG=${a[1]}
   python3 main.py --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/forward/model_fw_50.pt" --unet_bw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/backward/model_bw_80.pt"
   #mkdir -p "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg"
   mv "results" "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg/test_results"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg test"
done
