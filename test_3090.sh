#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
RESULTS_DIR='results'

EXPERIMENT_DIR='experiments' #used to store the backup results
mkdir -p $EXPERIMENT_DIR
MODEL_FW="model_weights/model_fw_50.pt"
MODEL_BW="model_weights/model_bw_80.pt"

python3 main.py --five_classes --train_tasknet --batch_size_tasknet 2 --lr_tasknet 4e-4
mkdir -p "${EXPERIMENT_DIR}/tasknet5classes"
cp "tasknet_weights/tasknet_10.pt" "${EXPERIMENT_DIR}/tasknet5classes"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/tasknet5classes/${RESULTS_DIR}"
echo "Completed Tasknet experiment and copied results to ${EXPERIMENT_DIR} folder"

python3 main.py --five_classes --batch_size_unet 2 --lr_unet 3e-4 --not_use_custom_filter_prop
mkdir -p "${EXPERIMENT_DIR}/allprop_5classes/forward"
cp "model_weights/model_fw_50.pt" "${EXPERIMENT_DIR}/allprop_5classes/forward"
mv "model_weights/model_fw_25.pt" "${EXPERIMENT_DIR}/allprop_5classes/forward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/allprop_5classes/forward/${RESULTS_DIR}_fw"
echo "Completed Allprop forward and copied folder"

python3 main.py --five_classes --batch_size_unet 2 --save_disturbed_dataset --unet_fw_weights_load $MODEL_FW

python3 main.py --five_classes --batch_size_unet 2 --train_model_backward
python3 main.py --five_classes --batch_size_unet 1 --compute_similarity_metrics --unet_bw_weights_load $MODEL_BW
mkdir -p "${EXPERIMENT_DIR}/allprop_5classes/backward"
mv "model_weights/model_bw_80.pt" "${EXPERIMENT_DIR}/allprop_5classes/backward"
mv "model_weights/model_bw_40.pt" "${EXPERIMENT_DIR}/allprop_5classes/backward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/allprop_5classes/backward/${RESULTS_DIR}_bw"
echo "Completed Allprop backward experiment and copied folder"

python3 main.py --five_classes --batch_size_unet 2 --lr_unet 3e-4 --prop_pos 2 --prop_neg 2
mkdir -p "${EXPERIMENT_DIR}/2pos2neg_5classes/forward"
cp "model_weights/model_fw_50.pt" "${EXPERIMENT_DIR}/2pos2neg_5classes/forward"
mv "model_weights/model_fw_25.pt" "${EXPERIMENT_DIR}/2pos2neg_5classes/forward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/2pos2neg_5classes/forward/${RESULTS_DIR}_fw"
echo "Completed 2pos2neg forward and copied folder"

python3 main.py --five_classes --batch_size_unet 2 --save_disturbed_dataset --unet_fw_weights_load $MODEL_FW

python3 main.py --five_classes --batch_size_unet 2 --train_model_backward
python3 main.py --five_classes --batch_size_unet 1 --compute_similarity_metrics --unet_bw_weights_load $MODEL_BW
mkdir -p "${EXPERIMENT_DIR}/2pos2neg_5classes/backward"
mv "model_weights/model_bw_80.pt" "${EXPERIMENT_DIR}/2pos2neg_5classes/backward"
mv "model_weights/model_bw_40.pt" "${EXPERIMENT_DIR}/2pos2neg_5classes/backward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/2pos2neg_5classes/backward/${RESULTS_DIR}_bw"
echo "Completed 2pos2neg backward experiment and copied folder"

python3 main.py --five_classes --batch_size_unet 2 --lr_unet 3e-4 --prop_pos 3 --prop_neg 3
mkdir -p "${EXPERIMENT_DIR}/3pos3neg_5classes/forward"
cp "model_weights/model_fw_50.pt" "${EXPERIMENT_DIR}/3pos3neg_5classes/forward"
mv "model_weights/model_fw_25.pt" "${EXPERIMENT_DIR}/3pos3neg_5classes/forward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/3pos3neg_5classes/forward/${RESULTS_DIR}_fw"
echo "Completed 3pos3neg forward and copied folder"

python3 main.py --five_classes --batch_size_unet 2 --save_disturbed_dataset --unet_fw_weights_load $MODEL_FW

python3 main.py --five_classes --batch_size_unet 2 --train_model_backward
python3 main.py --five_classes --batch_size_unet 1 --compute_similarity_metrics --unet_bw_weights_load $MODEL_BW
mkdir -p "${EXPERIMENT_DIR}/3pos3neg_5classes/backward"
mv "model_weights/model_bw_80.pt" "${EXPERIMENT_DIR}/3pos3neg_5classes/backward"
mv "model_weights/model_bw_40.pt" "${EXPERIMENT_DIR}/3pos3neg_5classes/backward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/3pos3neg_5classes/backward/${RESULTS_DIR}_bw"
echo "Completed 3pos3neg backward experiment and copied folder"
