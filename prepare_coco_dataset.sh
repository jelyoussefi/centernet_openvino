#!/bin/bash

# Define the COCO dataset directory
COCO_DIR="/workspace/datasets/COCO"

# Check if the dataset already exists
if [ -d "$COCO_DIR/val2017" ] && [ -d "$COCO_DIR/annotations" ]; then
    exit 0
fi

# Create the COCO dataset directory
mkdir -p "$COCO_DIR"
cd "$COCO_DIR" || exit

# Download the dataset
echo "Downloading COCO dataset..."
wget -q http://images.cocodataset.org/zips/val2017.zip
wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract the dataset
echo "Extracting COCO dataset..."
unzip -q ./val2017.zip
unzip -q ./annotations_trainval2017.zip

# Clean up zip files
echo "Cleaning up zip files..."
rm -f *.zip

echo "COCO dataset is ready at $COCO_DIR."
