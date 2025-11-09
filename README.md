# Privacy-oriented deep learning for object detection in curious cloud-based robot ecosystems

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
* Models for the people experiments: donwload from [here](https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/EQ5N8j82LpFCgqLHmfxrG2ABSU7A0QPh-uQaBb3i_BY9Ug?e=wSGPy5)
* Models trained for the vehicles experiments: download from [here](https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/Ea1QEFWjA-hKuExp1OGUyaMBJZgTnhjUh1ykPX6mqaNsSQ?e=rEH8uj)


## Usage
You can run the code used in the paper using the files contained in [scripts](./scripts).
* **Training**
  * Training the tasknet: use [tasknet_training.sh](scripts/training/tasknet_finetune_people.sh) and [tasknet_train_fiveclasses.sh](scripts/training/tasknet_finetune_vehicles.sh) to train the Faster R-CNN using the people and vehicle dataset
  * Training the obfuscator and the attacker with the proposals configurations: use [training_fw_bw_proposals_people.sh](scripts/training/training_fw_bw_proposals_people.sh) and  [training_fw_bw_proposals_vehicles.sh](scripts/training/training_fw_bw_proposals_vehicles.sh)
* **Testing:**
  * Test the tasknet alone: use [tasknet_test_people.sh](scripts/test/tasknet_test_people.sh) and [tasknet_test_vehicles.sh](scripts/test/tasknet_test_vehicles.sh) to test the Faster R-CNN using the people and vehicle dataset
  * Test obfuscator and attacker with different proposals configurations: use [test_proposals_people.sh](scripts/test/test_proposals_people.sh) and [test_proposals_vehicles.sh](scripts/test/test_proposals_vehicles.sh) for testing with people and vehicle datasets
  * Plot scripts: some plotting scripts are in this [folder](scripts/plot)
