#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ..

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

for i in "0.5" "0.6" "0.7" "0.8" "0.9"; do
   a=( $i )
   WEIGHT=${a[0]}
   python3 main.py --not_use_custom_filter_prop --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --weight $WEIGHT
   mkdir -p "${EXPERIMENT_DIR}/${WEIGHT}_MAE/forward"
   cp $UNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/${WEIGHT}_MAE/forward"
   mv $UNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/${WEIGHT}_MAE/forward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_MAE/forward/${RESULTS_DIR}_fw"
   echo "Completed ${WEIGHT}_MAE forward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW

   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --train_model_backward
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --results_dir $RESULTS_DIR --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
   mkdir -p "${EXPERIMENT_DIR}/${WEIGHT}_MAE/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE1 "${EXPERIMENT_DIR}/${WEIGHT}_MAE/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE2 "${EXPERIMENT_DIR}/${WEIGHT}_MAE/backward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_MAE/backward/${RESULTS_DIR}_bw"
   echo "Completed ${WEIGHT}_MAE backward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/${WEIGHT}_MAE/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/${WEIGHT}_MAE/backward/${UNET_BW}" --results_dir $RESULTS_DIR
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_MAE/test_results"
   echo "Completed ${WEIGHT}_MAE test"
done

for i in "0.6" "0.8"; do
   a=( $i )
   WEIGHT=${a[0]}
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --weight $WEIGHT --prop_pos 2 --prop_neg 2
   mkdir -p "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/forward"
   cp $UNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/forward"
   mv $UNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/forward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/forward/${RESULTS_DIR}_fw"
   echo "Completed ${WEIGHT}_2pos2neg_MAE forward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW

   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --train_model_backward
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --results_dir $RESULTS_DIR --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
   mkdir -p "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE1 "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE2 "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/backward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/backward/${RESULTS_DIR}_bw"
   echo "Completed ${WEIGHT}_2pos2neg_MAE backward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/backward/${UNET_BW}" --results_dir $RESULTS_DIR
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/test_results"
   echo "Completed ${WEIGHT}_2pos2neg_MAE test"
done

for i in "0.6" "0.8"; do
   a=( $i )
   WEIGHT=${a[0]}
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --weight $WEIGHT --prop_pos 4 --prop_neg 4
   mkdir -p "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/forward"
   cp $UNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/forward"
   mv $UNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/forward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/forward/${RESULTS_DIR}_fw"
   echo "Completed ${WEIGHT}_4pos4neg_MAE forward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW

   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --train_model_backward
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --results_dir $RESULTS_DIR --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
   mkdir -p "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE1 "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE2 "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/backward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/backward/${RESULTS_DIR}_bw"
   echo "Completed ${WEIGHT}_4pos4neg_MAE backward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/backward/${UNET_BW}" --results_dir $RESULTS_DIR
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/test_results"
   echo "Completed ${WEIGHT}_4pos4neg_MAE test"
done

for i in "0.5" "0.7" "0.9"; do
   a=( $i )
   WEIGHT=${a[0]}
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --weight $WEIGHT --prop_pos 2 --prop_neg 2
   mkdir -p "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/forward"
   cp $UNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/forward"
   mv $UNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/forward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/forward/${RESULTS_DIR}_fw"
   echo "Completed ${WEIGHT}_2pos2neg_MAE forward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW

   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --train_model_backward
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --results_dir $RESULTS_DIR --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
   mkdir -p "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE1 "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE2 "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/backward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/backward/${RESULTS_DIR}_bw"
   echo "Completed ${WEIGHT}_2pos2neg_MAE backward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/backward/${UNET_BW}" --results_dir $RESULTS_DIR
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_2pos2neg_MAE/test_results"
   echo "Completed ${WEIGHT}_2pos2neg_MAE test"
done

for i in "0.5" "0.7" "0.9"; do
   a=( $i )
   WEIGHT=${a[0]}
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_forward $NUM_EPOCHS_UNET_FORWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --weight $WEIGHT --prop_pos 4 --prop_neg 4
   mkdir -p "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/forward"
   cp $UNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/forward"
   mv $UNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/forward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/forward/${RESULTS_DIR}_fw"
   echo "Completed ${WEIGHT}_4pos4neg_MAE forward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --save_disturbed_dataset --unet_fw_weights_load $UNET_WEIGHTS_LOAD_FW

   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet $BATCH_SIZE_UNET --num_epochs_unet_backward $NUM_EPOCHS_UNET_BACKWARD --unet_save_path $UNET_SAVE_PATH --results_dir $RESULTS_DIR --train_model_backward
   python3 main.py --use_dataset_subset $USE_DATASET_SUBSET --batch_size_unet 1 --results_dir $RESULTS_DIR --compute_similarity_metrics --unet_bw_weights_load $UNET_WEIGHTS_LOAD_BW
   mkdir -p "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE1 "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/backward"
   mv $UNET_WEIGHTS_BW_TO_SAVE2 "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/backward"
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/backward/${RESULTS_DIR}_bw"
   echo "Completed ${WEIGHT}_4pos4neg_MAE backward experiment and copied results to ${EXPERIMENT_DIR} folder"
   
   python3 main.py --test_model --tasknet_weights_load $TASKNET_WEIGHTS_LOAD --unet_fw_weights_load "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/forward/${UNET_FW}" --unet_bw_weights_load "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/backward/${UNET_BW}" --results_dir $RESULTS_DIR
   mv $RESULTS_DIR "${EXPERIMENT_DIR}/${WEIGHT}_4pos4neg_MAE/test_results"
   echo "Completed ${WEIGHT}_4pos4neg_MAE test"
done
