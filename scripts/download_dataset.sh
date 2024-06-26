#!/usr/bin/env bash
set -u -e #safety net to avoid using not setted variables
cd ..

download_and_extract() {
    local url=$1
    local output_dir=$2
    local file_name=$3

    echo "Downloading $file_name..."
    wget -c "$url" -O "$file_name"

    echo "Extracting $file_name..."
    if [[ $file_name == *.zip ]]; then
        unzip -q "$file_name" -d "$output_dir"
    elif [[ $file_name == *.tar ]]; then
        tar -xf "$file_name" -C "$output_dir"
    fi

    echo "Removing $file_name..."
    rm "$file_name"
}

mkdir -p "dataset/coco2017"
mkdir -p "dataset/pascalVOC2012"
mkdir -p "tasknet_weights"

COCO2017_TRAIN="http://images.cocodataset.org/zips/train2017.zip"
COCO2017_VAL="http://images.cocodataset.org/zips/val2017.zip"
COCO2017_ANNOTATIONS="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
PASCAL_VOC2012="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
ANNOTATIONS="https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/EfYfEsdXWv9EmVFLQ7gAB9MBenIi1GoNw2GneonuznF7CQ?download=1" #place microsoft download link, then after ? substitute the rest of the text with download=1
TASKNET_1CLASS="https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/EWxaodFEquZMll6E1OnP5SIBpw-XofLmNiSAPlKHeRGQnw?download=1"
TASKNET_5CLASS="https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/EVbEPAF4pdRIoYkj-f4IkycBQiKvepADuLUxieZHcU4B2Q?download=1"

if [ -d "dataset/coco2017/train2017" ]; then
   echo "COCO2017 Train dataset already exists, skipping download."
else
   download_and_extract "$COCO2017_TRAIN" "dataset/coco2017" "train2017.zip"
fi

if [ -d "dataset/coco2017/val2017" ]; then
   echo "COCO2017 Val dataset already exists, skipping download."
else
download_and_extract "$COCO2017_VAL" "dataset/coco2017" "val2017.zip"
fi

if [ -d "dataset/coco2017/annotations" ]; then
   echo "COCO2017 Annotations already exists, skipping download."
else
   download_and_extract "$COCO2017_ANNOTATIONS" "dataset/coco2017" "annotations_trainval2017.zip"
fi

if [ -d "dataset/pascalVOC2012/images" ]; then
   echo "PascalVOC2012 Images already exists, skipping download."
else
   download_and_extract "$PASCAL_VOC2012" "dataset/pascalVOC2012" "VOCtrainval_11-May-2012.tar"
fi

mv "dataset/pascalVOC2012/VOCdevkit/VOC2012/JPEGImages" "dataset/pascalVOC2012/images"
rm -r "dataset/pascalVOC2012/VOCdevkit"

if [ -d "dataset/annotations/coco_people_indoor" ]; then
   echo "Annotations already exists, skipping download."
else
   download_and_extract "$ANNOTATIONS" "dataset" "annotations.zip"
fi

if [ -f "tasknet_weights/tasknet_1class.pt" ]; then
   echo "Tasknet weights for 1 class already exists, skipping download."
else
   download_and_extract "$TASKNET_1CLASS" "tasknet_weights" "tasknet_1class.zip"
fi

if [ -f "tasknet_weights/tasknet_5classes.pt" ]; then
   echo "Tasknet weights for 5 classes already exists, skipping download."
else
   download_and_extract "$TASKNET_5CLASSES" "tasknet_weights" "tasknet_5classes.zip"
fi

echo "Datasets, Annotations and Tasknet Weights downloaded and extracted"
