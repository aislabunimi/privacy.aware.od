# Privacy-oriented deep learning for object detection in curious cloud-based robot ecosystems

You can find the trained models and the datasets [here](https://mega.nz/folder/QXYwyawI#h0xKjDmgAUamI61ekhmyMg).

## Installation and Setup

Simply run the scripts called `install.sh` under the folder scripts:
```bash
scripts/install.sh
```
If some shell script doesn't have the permissions to run:
```bash
chmod +x scripts/*.sh
```
As you may see by inspecting the install.sh file, the scripts installs the Pytorch version used in the paper (`torch==2.2.1, torchvision==0.17.1, torchaudio==2.2.1 cu118`).

A newer Pytorch version should work without any problem (the same goes for the other packages in `requirements.txt`). If you want to use a different Pytorch version, you can install only the requirements by running this command instead:
```bash
pip install -r requirements.txt
```

For downloading the dataset, run the scripts called `download_dataset.sh` under the folder scripts:
```bash
scripts/download_datasets.sh
```
Manually download additional files from OneDrive:
* Annotations: download from this [link](https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/ETBSSzNN_-lGk2gLD37elNQBKVHg9KqfoXgQL7GJRCdIbA?e=wGgVuz) and extract it inside the folder [dataset](./dataset)
* Robot dataset: download from this [link](https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/ETR0ryc4_35BsWtrbMLRk6wBnciDHdiooDssGCAlWZeLEQ?e=eQacRL) and extract it inside the folder [dataset](./dataset)


## Usage
You can run the code used in the paper using the files contained in [scripts](./scripts).
* **Training the tasknet:** use [tasknet_training.sh](scripts/training/tasknet_finetune_people.sh) and [tasknet_train_fiveclasses.sh](scripts/training/tasknet_finetune_vehicles.sh) to train the Faster R-CNN using the people and vehicle dataset



## Using the project
It's suggested to use the helper scripts in the scripts folder for this project, as it uses a particular training pipeline.

With TASKNET we refer to a Faster R-CNN model, with UNET to an encoder decoder similar to the UNet in layers, but without the skip connections.

Each scripts contains some variables. The ones that you may be interested in are:
- `USE_DATASET_SUBSET`: default is 0, meaning all the samples from dataset will be used. If you want to train on a smaller set, you can specify a number: 1000 means you are training the model with 1000 images.
- `NUM_EPOCHS_{MODEL}`: specifies the number of epochs. {MODEL} is either TASKNET or UNET.
- `BATCH_SIZE_{MODEL}`: the batch size used. {MODEL} is either TASKNET or UNET.
- `LR_{MODEL}`: the learning rate. {MODEL} is either TASKNET or UNET.
- The values of these variables in these scripts are the ones used in the results reported in the paper.

All the other options can be seen using `python3 main.py -h`. You may also use this file alone if you want, but it's highly unrecommended (it's easier to edit the .sh scripts).

For running one of the scripts reported in next subsections, just use the command:
```bash
scripts/{script_name}.sh
```
where {script_name} is one of `tasknet.sh`, `tasknet_fiveclasses.sh`, `fw_bw_proposals.sh`, `five_classes.sh`, `finetune_tasknet.sh`.

### Training Tasknet
The `tasknet.sh` script is needed only if you want to train again the tasknet (the script `tasknet_fiveclasses.sh` is the corresponding one for five classes experiments).
The training loop is training tasknet, validation on batch=1 (needed for comparable results with testing), testing. The Tasknet training is done on a bigger dataset (all people of COCO dataset), while validation on the reduced indoor dataset used with proposals selection method.

### Training with Proposals Selection
The script `fw_bw_proposals.sh` contains all the experiments with all settings reported in the paper. First, an experiment using all proposals is done. Then there are the experiments will all possible combinations of positive and negative proposals.
The training loop in the script is:
1. Training forward: the UNet forward is trained to optimize the performance of the Tasknet. In Training, the Tasknet behaves with the custom proposal method producing a weakened loss. In Validation, the Tasknet works normally.
2. Validation forward with batch size=1: this is done only to obtain comparable results with testing.
3. Saving disturbed dataset: using the UNet forward trained before, the privatized dataset is generated. This simulates an attacker that collects many sample of privatized images with corresponding plain ones.
4. Train Backward: the UNet backward is trained to reduce a reconstruction error between the reconstructed image and the original one, learning to reconstruct the plain image from the privatized ones.
5. Compute similarity metrics: similar to validation with batch size=1, need to have comparable results with testing. Also, for computing the right similarity metrics, batch size 1 is mandatory to avoid any introduced padding.
6. Test Model: this last step computes the test results on the PascalVOC2012ValSet for both the forward and backward experiments.

If you want to avoid the "all proposals" setting, you can just make a copy of the shell script and remove that part. If you want to try only some settings or different ones with custom proposals method, you can delete some settings in the "for i in" loop.

### Training with Four classes (Five Classes)
The script `five_classes.sh` contains the experiments with the settings used for four classes vehicles (the experiments and variable names are called five classes, but in reality only four classes are used). This scripts has the same training loop of the script above. You can use it as it is, or modify it if you want to test with different proposals.
You can also use the script `fw_bw_proposals.sh` by adding flag `--five_classes` to all commands in the script. In that case, it's best if you change the name of the folders where you save the results (change the name of the folder after `${EXPERIMENT_DIR}`).

### Finetuning Tasknet on UNet generated images
The script `finetune_tasknet.sh` contains an example of how you can finetune the tasknet on the UNet generated images. The training loop is finetune tasknet, validation on batch=1, testing. To use this script, you need to have done before the UNet forward experiments with corresponding setting.
The scripts does the finetuning with the settings: "all proposals, 2pos2neg, 1pos1neg". You can modify it based on your needs if you want to finetune on different settings:
1. Set the UNET variable to the path of the desired forward weights
2. Change the name of the folder after `${EXPERIMENT_DIR}` for all the corresponding experiments till the next UNET variable appears in the script.

### Plotting the Results
For faster plotting of all experiments, you can use the scripts `plot.sh` or `plot_fiveclasses.sh` to plot example images from PascalVOC2012ValSet with 1class and 5class respectively.
The scripts plots the following files:
1. For Tasknet: images contained in `examples`, in tasknet folder with the predictions.
2. For the other experiments (including all proposals):
	1. For forward weights: images contained in `forward/examples_fw`, reconstructed by the UNet forward with the predictions made by the Tasknet. In `forward/graphs_fw` you will find some graphs obtained from validation epochs.
	2. For backward weights: images contained in `backward/examples_bw`, reconstructed by the UNet backward. In `backward/graphs_bw` you will find some graphs obtained from validation epochs.

Alternatively, you can run `python3 plot_results.py -h`, showing the possible options for the plotting.
You can modify these bash scripts by adding the flag `--plot_my_test_data` if you want to plot examples with your own data, or trying other plotting methods.
