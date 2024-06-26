#Privacy-oriented deep learning for object detection in curious cloud-based robot ecosystems

This repository contains the code for my master thesis (and future paper).

##Installation
###Packages
Simply run the scripts called "install.sh" under the folder scripts:
```bash
scripts/install.sh
```

If the file doesn't have the permissions to run:
```bash
chmod +x scripts/*.sh
```

As you may see by inspecting the install.sh file, the scripts installs the Pytorch version used in the paper (torch==2.2.1, torchvision==0.17.1, torchaudio==2.2.1 cu118). A newer Pytorch version should work without any problem (the same goes for the other packages in requirements.txt).

###Dataset download
For downloading the dataset, run the scripts called "download_dataset.sh" under the folder scripts:
```bash
scripts/download_dataset.sh
```

It may take a while depending on your Internet speed.

##Using the project
###Helper scripts


In the scripts folder you you can find an example of how to do various experiments with different settings.

The main is found in main.py file. Run 'python3 main.py -h' to see all possible options.

The plot_results.py file contains the code for plotting the results. Running with 'python3 plot_results.py' shows the possible options.
