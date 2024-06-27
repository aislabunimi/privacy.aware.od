#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ..

RESULTS_DIR='results'

EXPERIMENT_DIR='experiments' #used to store the backup results
mkdir -p $EXPERIMENT_DIR
MODEL_FW="model_weights/model_fw_50.pt"
MODEL_BW="model_weights/model_bw_80.pt"
TASKNET="tasknet_weights/tasknet_5classes.pt"

COCO_FIVECLASSES_PATH='/home/alberti/Tesi_magistrale/coco_vehicles'
COCO_ALLCLASSES_PATH='/home/alberti/Tesi_magistrale/coco2017'
PASCAL_IMG_PATH='/home/alberti/Tesi_magistrale/VOCdevkit/VOC2012/images'

python3 main.py --train_tasknet --five_classes --tasknet_weights_load $TASKNET --use_dataset_subset 0 --batch_size_tasknet 1 --results_dir $RESULTS_DIR --val_forward_batch1
#mv $RESULTS_DIR "${EXPERIMENT_DIR}/tasknet/val_forward_batch1"
mkdir -p "${EXPERIMENT_DIR}/5classes_val_batch_1/tasknet"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/5classes_val_batch_1/tasknet/val_results_batch1"
echo "Completed Tasknet Validation with Batch Size 1"

python3 main.py --use_dataset_subset 0 --batch_size_unet 1 --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET --not_use_custom_filter_prop --unet_fw_weights_load "${EXPERIMENT_DIR}/allprop_5classes/forward/model_fw_50.pt" --val_forward_batch1 --five_classes
#mkdir -p "${EXPERIMENT_DIR}/all_proposals/val_forward_batch1"
mkdir -p "${EXPERIMENT_DIR}/5classes_val_batch_1/allprop_5classes"
#mv $RESULTS_DIR "${EXPERIMENT_DIR}/all_proposals/forward/${RESULTS_DIR}_fw"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/5classes_val_batch_1/allprop_5classes/val_results_batch1"
echo "Completed all_proposals Validation with Batch Size 1"

python3 main.py --use_dataset_subset 0 --batch_size_unet 1 --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET --not_use_custom_filter_prop --unet_fw_weights_load "${EXPERIMENT_DIR}/3pos3neg_5classes/forward/model_fw_50.pt" --val_forward_batch1 --five_classes
mkdir -p "${EXPERIMENT_DIR}/5classes_val_batch_1/3pos3neg_5classes"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/5classes_val_batch_1/3pos3neg_5classes/val_results_batch1"
echo "Completed 3pos3neg Validation with Batch Size 1"

python3 main.py --use_dataset_subset 0 --batch_size_unet 1 --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET --not_use_custom_filter_prop --unet_fw_weights_load "${EXPERIMENT_DIR}/2pos2neg_5classes/forward/model_fw_50.pt" --val_forward_batch1 --five_classes
mkdir -p "${EXPERIMENT_DIR}/5classes_val_batch_1/2pos2neg_5classes"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/5classes_val_batch_1/2pos2neg_5classes/val_results_batch1"
echo "Completed 2pos2neg Validation with Batch Size 1"
