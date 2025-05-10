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


  mv "dataset/pascalVOC2012/VOCdevkit/VOC2012/JPEGImages" "dataset/pascalVOC2012/images"
  rm -r "dataset/pascalVOC2012/VOCdevkit"
fi

echo "Datasets downloaded and extracted"
