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
BATCH_SIZE_UNET=2
LR_UNET=3e-4

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

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --lr_unet $LR_UNET --not_use_custom_filter_prop --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --tasknet_weights_load $TASKNET_WEIGHTS_LOAD
mkdir -p "${EXPERIMENT_DIR}/allprop_5classes/forward"
cp $UNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/allprop_5classes/forward"
mv $UNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/allprop_5classes/forward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/allprop_5classes/forward/${RESULTS_DIR}_fw"
echo "Completed Allprop forward and copied folder"

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --not_use_custom_filter_prop --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW --val_forward_batch1
mv $RESULTS_DIR "${EXPERIMENT_DIR}/allprop_5classes/forward/val_results_batch1"
echo "Completed allprop_5classes Validation with Batch Size 1"

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --lr_unet $LR_UNET --train_model_backward --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH
python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
mkdir -p "${EXPERIMENT_DIR}/allprop_5classes/backward"
mv $UNET_WEIGHTS_BW_TO_SAVE1 "${EXPERIMENT_DIR}/allprop_5classes/backward"
mv $UNET_WEIGHTS_BW_TO_SAVE2 "${EXPERIMENT_DIR}/allprop_5classes/backward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/allprop_5classes/backward/${RESULTS_DIR}_bw"
echo "Completed Allprop backward experiment and copied folder"

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/allprop_5classes/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/allprop_5classes/backward/${UNET_BW}"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/allprop_5classes/test_results"
echo "Completed allprop_5classes test"

for i in "3 3" "2 2"; do
   a=( $i )
   PROP_POS=${a[0]}
   PROP_NEG=${a[1]}
   python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --lr_unet $LR_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --prop_pos $PROP_POS --prop_neg $PROP_NEG
   mkdir -p "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/forward"
   cp $UNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/forward"
   mv $UNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/forward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/forward/${RESULTS_DIR}_fw"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg_5classes forward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --not_use_custom_filter_prop --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW --val_forward_batch1
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/forward/val_results_batch1"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg_5classes Validation with Batch Size 1"
   
   python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW

   python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --lr_unet $LR_UNET --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --train_model_backward
   python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --results_dir $RESULTS_DIR --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
   mkdir -p "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE1 "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE2 "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/backward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/backward/${RESULTS_DIR}_bw"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg_5classes backward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --five_classes --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/backward/${UNET_BW}" --results_dir $RESULTS_DIR
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${PROP_POS}pos${PROP_NEG}neg_5classes/test_results"
   echo "Completed ${PROP_POS}pos${PROP_NEG}neg_5classes test"
done

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --lr_unet $LR_UNET --not_use_custom_filter_prop --weight 0.6 --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --tasknet_weights_load $TASKNET_WEIGHTS_LOAD
mkdir -p "${EXPERIMENT_DIR}/0.6_MAE_5classes/forward"
cp "model_weights/model_fw_50.pt" "${EXPERIMENT_DIR}/0.6_MAE_5classes/forward"
mv "model_weights/model_fw_25.pt" "${EXPERIMENT_DIR}/0.6_MAE_5classes/forward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/0.6_MAE_5classes/forward/${RESULTS_DIR}_fw"
echo "Completed 0.6_MAE_5classes forward and copied folder"

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --not_use_custom_filter_prop --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW --val_forward_batch1 --weight 0.6
mv $RESULTS_DIR "${EXPERIMENT_DIR}/0.6_MAE_5classes/forward/val_results_batch1"
echo "Completed 0.6_MAE_5classes Validation with Batch Size 1"

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --lr_unet $LR_UNET --train_model_backward --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH
python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
mkdir -p "${EXPERIMENT_DIR}/0.6_MAE_5classes/backward"
mv "model_weights/model_bw_80.pt" "${EXPERIMENT_DIR}/0.6_MAE_5classes/backward"
mv "model_weights/model_bw_40.pt" "${EXPERIMENT_DIR}/0.6_MAE_5classes/backward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/0.6_MAE_5classes/backward/${RESULTS_DIR}_bw"
echo "Completed 0.6_MAE_5classes backward experiment and copied folder"

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/0.6_MAE_5classes/forward/model_fw_50.pt" --unet_bw_weights_load "${EXPERIMENT_DIR}/0.6_MAE_5classes/backward/model_bw_80.pt"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/0.6_MAE_5classes/test_results"
echo "Completed 0.6_MAE_5classes test"

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --lr_unet $LR_UNET --not_use_custom_filter_prop --weight 0.7 --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --tasknet_weights_load $TASKNET_WEIGHTS_LOAD
mkdir -p "${EXPERIMENT_DIR}/0.7_MAE_5classes/forward"
cp "model_weights/model_fw_50.pt" "${EXPERIMENT_DIR}/0.7_MAE_5classes/forward"
mv "model_weights/model_fw_25.pt" "${EXPERIMENT_DIR}/0.7_MAE_5classes/forward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/0.7_MAE_5classes/forward/${RESULTS_DIR}_fw"
echo "Completed 0.7_MAE_5classes forward and copied folder"

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --not_use_custom_filter_prop --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW --val_forward_batch1 --weight 0.7
mv $RESULTS_DIR "${EXPERIMENT_DIR}/0.7_MAE_5classes/forward/val_results_batch1"
echo "Completed 0.7_MAE_5classes Validation with Batch Size 1"

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --lr_unet $LR_UNET --train_model_backward --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH
python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
mkdir -p "${EXPERIMENT_DIR}/0.7_MAE_5classes/backward"
mv "model_weights/model_bw_80.pt" "${EXPERIMENT_DIR}/0.7_MAE_5classes/backward"
mv "model_weights/model_bw_40.pt" "${EXPERIMENT_DIR}/0.7_MAE_5classes/backward"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/0.7_MAE_5classes/backward/${RESULTS_DIR}_bw"
echo "Completed 0.7_MAE_5classes backward experiment and copied folder"

python3 main.py --five_classes --use_dataset_subset $USE_DATASET_SUBSET --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/0.7_MAE_5classes/forward/model_fw_50.pt" --unet_bw_weights_load "${EXPERIMENT_DIR}/0.7_MAE_5classes/backward/model_bw_80.pt"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/0.7_MAE_5classes/test_results"
echo "Completed 0.7_MAE_5classes test"
