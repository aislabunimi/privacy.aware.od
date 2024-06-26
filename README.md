# Privacy-oriented deep learning for object detection in curious cloud-based robot ecosystems

This repository contains the code for my master thesis (and future paper).

## Installation
### Packages
Simply run the scripts called "install.sh" under the folder scripts:
```bash
scripts/install.sh
```
If the file doesn't have the permissions to run:
```bash
chmod +x scripts/*.sh
```
As you may see by inspecting the install.sh file, the scripts installs the Pytorch version used in the paper (torch==2.2.1, torchvision==0.17.1, torchaudio==2.2.1 cu118). A newer Pytorch version should work without any problem (the same goes for the other packages in requirements.txt).

### Dataset download
For downloading the dataset, run the scripts called "download_dataset.sh" under the folder scripts:
```bash
scripts/download_dataset.sh
```
It may take a while depending on your Internet speed.

## Using the project
### Helper scripts
The tasknet.sh script is needed only if you want to train again the tasknet (the script tasknet_fiveclasses is the corresponding one for five classes experiments).

The script fw_bw_proposals.sh contains all the experiments with all settings reported in the paper.

The script five_classes.sh contains the experiments with the settings used for five classes.

These scripts can be used as helper to define your own script (for example if you want to tweak the batch size, or execute the experiments only for certain examples).

You can also use the main.py file. Run 'python3 main.py -h' to see all possible options with their explanation.

The plot_results.py file contains the code for plotting the results. Running with 'python3 plot_results.py -h' shows the possible options.
