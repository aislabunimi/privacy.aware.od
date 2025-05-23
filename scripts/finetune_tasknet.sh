#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ..

RESULTS_DIR='results'
TASKNET_SAVE_PATH='tasknet_weights/tasknet'

EXPERIMENT_DIR='experiments' #used to store the backup results
mkdir -p $EXPERIMENT_DIR
USE_DATASET_SUBSET=0

NUM_EPOCHS_TASKNET=10
BATCH_SIZE_UNET=4 #controls the one also for tasknet only in this case
LR_TASKNET=5e-4

TASKNET_WEIGHTS_FW_TO_SAVE1="${TASKNET_SAVE_PATH}_finetuned_${NUM_EPOCHS_TASKNET}.pt"
HALF_WAY=$((NUM_EPOCHS_TASKNET/2))
if [ "$HALF_WAY" -eq 0 ]; then
   HALF_WAY=1
fi
TASKNET_WEIGHTS_FW_TO_SAVE2="${TASKNET_SAVE_PATH}_finetuned_${HALF_WAY}.pt"

#If you want to use five classes, add flag --five_classes to all commands

#weights to load for finetuning: the tasknet on 1 class and the UNet already trained

#Need to cp tasknet weights to keep them as initial weights for finetuning
#if you have trained it yourself
[[ -e "tasknet_weights/tasknet_10.pt" ]] && cp "tasknet_weights/tasknet_10.pt" "${EXPERIMENT_DIR}/tasknet/tasknet_1class.pt"
#or you have download it
[[ -e "tasknet_weights/tasknet_1class.pt" ]] && cp "tasknet_weights/tasknet_1class.pt" "${EXPERIMENT_DIR}/tasknet/tasknet_1class.pt"

TASKNET="${EXPERIMENT_DIR}/tasknet/tasknet_1class.pt"
UNET="${EXPERIMENT_DIR}/all_proposals/forward/model_fw_50.pt"

python3 main.py --finetune_tasknet --tasknet_save_path $TASKNET_SAVE_PATH --use_dataset_subset $USE_DATASET_SUBSET --num_epochs_tasknet $NUM_EPOCHS_TASKNET --batch_size_unet $BATCH_SIZE_UNET --lr_tasknet $LR_TASKNET --results_dir $RESULTS_DIR --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET
mkdir -p "${EXPERIMENT_DIR}/finetuning_tasknet_on_unetallprop"
cp $TASKNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/finetuning_tasknet_on_unetallprop"
mv $TASKNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/finetuning_tasknet_on_unetallprop"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_unetallprop/results"
echo "Completed Finetuning Tasknet On UNet trained with all proposals"

python3 main.py --finetune_tasknet --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --use_dataset_subset $USE_DATASET_SUBSET --batch_size_tasknet 1 --batch_size_unet 1 --results_dir $RESULTS_DIR --val_forward_batch1 --unet_fw_weights_load $UNET
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_unetallprop/val_results_batch1"
echo "Completed Finetuning Tasknet On UNet trained with all proposals Validation with Batch Size 1"

python3 main.py --test_tasknet --finetune_tasknet --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_unetallprop/test_results"
echo "Completed Finetuning Tasknet On UNet trained with all proposals test"


UNET="${EXPERIMENT_DIR}/2pos2neg/forward/model_fw_50.pt"

python3 main.py --finetune_tasknet --tasknet_save_path $TASKNET_SAVE_PATH --use_dataset_subset $USE_DATASET_SUBSET --num_epochs_tasknet $NUM_EPOCHS_TASKNET --batch_size_unet $BATCH_SIZE_UNET --lr_tasknet $LR_TASKNET --results_dir $RESULTS_DIR --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET
mkdir -p "${EXPERIMENT_DIR}/finetuning_tasknet_on_2pos2neg"
cp $TASKNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/finetuning_tasknet_on_2pos2neg"
mv $TASKNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/finetuning_tasknet_on_2pos2neg"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_2pos2neg/results"
echo "Completed Finetuning Tasknet On UNet trained with 2pos2neg"

python3 main.py --finetune_tasknet --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --use_dataset_subset $USE_DATASET_SUBSET --batch_size_tasknet 1 --batch_size_unet 1 --results_dir $RESULTS_DIR --val_forward_batch1 --unet_fw_weights_load $UNET
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_2pos2neg/val_results_batch1"
echo "Completed Finetuning Tasknet On UNet trained with 2pos2neg Validation with Batch Size 1"

python3 main.py --test_tasknet --finetune_tasknet --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_2pos2neg/test_results"
echo "Completed Finetuning Tasknet On UNet trained with 2pos2neg test"


UNET="${EXPERIMENT_DIR}/0.8_MAE/forward/model_fw_50.pt"

python3 main.py --finetune_tasknet --tasknet_save_path $TASKNET_SAVE_PATH --use_dataset_subset $USE_DATASET_SUBSET --num_epochs_tasknet $NUM_EPOCHS_TASKNET --batch_size_unet $BATCH_SIZE_UNET --lr_tasknet $LR_TASKNET --results_dir $RESULTS_DIR --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET
mkdir -p "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.8_MAE"
cp $TASKNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.8_MAE"
mv $TASKNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.8_MAE"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.8_MAE/results"
echo "Completed Finetuning Tasknet On UNet trained with 0.8_MAE"

python3 main.py --finetune_tasknet --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --use_dataset_subset $USE_DATASET_SUBSET --batch_size_tasknet 1 --batch_size_unet 1 --results_dir $RESULTS_DIR --val_forward_batch1 --unet_fw_weights_load $UNET
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.8_MAE/val_results_batch1"
echo "Completed Finetuning Tasknet On UNet trained with 0.8_MAE Validation with Batch Size 1"

python3 main.py --test_tasknet --finetune_tasknet --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.8_MAE/test_results"
echo "Completed Finetuning Tasknet On UNet trained with 0.8_MAE test"


UNET="${EXPERIMENT_DIR}/1pos1neg/forward/model_fw_50.pt"

python3 main.py --finetune_tasknet --tasknet_save_path $TASKNET_SAVE_PATH --use_dataset_subset $USE_DATASET_SUBSET --num_epochs_tasknet $NUM_EPOCHS_TASKNET --batch_size_unet $BATCH_SIZE_UNET --lr_tasknet $LR_TASKNET --results_dir $RESULTS_DIR --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET
mkdir -p "${EXPERIMENT_DIR}/finetuning_tasknet_on_1pos1neg"
cp $TASKNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/finetuning_tasknet_on_1pos1neg"
mv $TASKNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/finetuning_tasknet_on_1pos1neg"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_1pos1neg/results"
echo "Completed Finetuning Tasknet On UNet trained with 1pos1neg"

python3 main.py --finetune_tasknet --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --use_dataset_subset $USE_DATASET_SUBSET --batch_size_tasknet 1 --batch_size_unet 1 --results_dir $RESULTS_DIR --val_forward_batch1 --unet_fw_weights_load $UNET
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_1pos1neg/val_results_batch1"
echo "Completed Finetuning Tasknet On UNet trained with 1pos1neg Validation with Batch Size 1"

python3 main.py --test_tasknet --finetune_tasknet --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_1pos1neg/test_results"
echo "Completed Finetuning Tasknet On UNet trained with 1pos1neg test"


UNET="${EXPERIMENT_DIR}/0.9_MAE/forward/model_fw_50.pt"

python3 main.py --finetune_tasknet --tasknet_save_path $TASKNET_SAVE_PATH --use_dataset_subset $USE_DATASET_SUBSET --num_epochs_tasknet $NUM_EPOCHS_TASKNET --batch_size_unet $BATCH_SIZE_UNET --lr_tasknet $LR_TASKNET --results_dir $RESULTS_DIR --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET
mkdir -p "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.9_MAE"
cp $TASKNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.9_MAE"
mv $TASKNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.9_MAE"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.9_MAE/results"
echo "Completed Finetuning Tasknet On UNet trained with 0.9_MAE"

python3 main.py --finetune_tasknet --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --use_dataset_subset $USE_DATASET_SUBSET --batch_size_tasknet 1 --batch_size_unet 1 --results_dir $RESULTS_DIR --val_forward_batch1 --unet_fw_weights_load $UNET
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.9_MAE/val_results_batch1"
echo "Completed Finetuning Tasknet On UNet trained with 0.9_MAE Validation with Batch Size 1"

python3 main.py --test_tasknet --finetune_tasknet --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.9_MAE/test_results"
echo "Completed Finetuning Tasknet On UNet trained with 0.9_MAE test"


UNET="${EXPERIMENT_DIR}/0.5_MAE/forward/model_fw_50.pt"

python3 main.py --finetune_tasknet --tasknet_save_path $TASKNET_SAVE_PATH --use_dataset_subset $USE_DATASET_SUBSET --num_epochs_tasknet $NUM_EPOCHS_TASKNET --batch_size_unet $BATCH_SIZE_UNET --lr_tasknet $LR_TASKNET --results_dir $RESULTS_DIR --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET
mkdir -p "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.5_MAE"
cp $TASKNET_WEIGHTS_FW_TO_SAVE1 "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.5_MAE"
mv $TASKNET_WEIGHTS_FW_TO_SAVE2 "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.5_MAE"
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.5_MAE/results"
echo "Completed Finetuning Tasknet On UNet trained with 0.5_MAE"

python3 main.py --finetune_tasknet --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --use_dataset_subset $USE_DATASET_SUBSET --batch_size_tasknet 1 --batch_size_unet 1 --results_dir $RESULTS_DIR --val_forward_batch1 --unet_fw_weights_load $UNET
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.5_MAE/val_results_batch1"
echo "Completed Finetuning Tasknet On UNet trained with 0.5_MAE Validation with Batch Size 1"

python3 main.py --test_tasknet --finetune_tasknet --unet_fw_weights_load $UNET --tasknet_weights_load $TASKNET_WEIGHTS_FW_TO_SAVE1 --results_dir $RESULTS_DIR
mv $RESULTS_DIR "${EXPERIMENT_DIR}/finetuning_tasknet_on_0.5_MAE/test_results"
echo "Completed Finetuning Tasknet On UNet trained with 0.5_MAE test"
