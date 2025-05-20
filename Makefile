# ----------------------------------
# General Settings
# ----------------------------------
SHELL := /bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# Directories
DATASET_DIR := ./dataset
MODELS_DIR := ./models
DATASET_DIR := /workspace/datasets/COCO

CHECKPOINT_PATH ?= ./model_epoch_8.pth
CONFIG_PATH ?= ./model_centernet_r18_8xb16-crop512-140e_coco.py

# Default Parameters
DEVICE      ?= GPU
INPUT_SIZE  ?= 640
PRECISION   ?= FP32

#MODEL_PATH ?= $(MODELS_DIR)/$(PRECISION)/ctdet_coco_dlav0_512.xml
#INPUT_PATH ?= ./streams/streat.mp4

MODEL_PATH ?= $(MODELS_DIR)/$(PRECISION)/centernet.xml
INPUT_PATH ?= ./streams/tube_3.jpg
# ----------------------------------
# Docker Configuration
# ----------------------------------
DOCKER_IMAGE_NAME := centernet_openvino
export DOCKER_BUILDKIT := 1

# Docker Build Parameters
DOCKER_BUILD_PARAMS := \
    --rm \
    --network=host \
    --build-arg http_proxy=$(HTTP_PROXY) \
    --build-arg https_proxy=$(HTTPS_PROXY) \
    --build-arg no_proxy=$(NO_PROXY) \
    -t $(DOCKER_IMAGE_NAME) .

# Docker Run Parameters
DOCKER_RUN_PARAMS := \
    -it --rm \
    --network=host \
    -a stdout -a stderr \
    --privileged \
    -v /dev:/dev \
    -e DISPLAY=$(DISPLAY) \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(CURRENT_DIR):/workspace \
    -w /workspace \
    -e http_proxy=$(HTTP_PROXY) \
    -e https_proxy=$(HTTPS_PROXY) \
    -e no_proxy=$(NO_PROXY) \
    $(DOCKER_IMAGE_NAME)

# ----------------------------------
# Targets
# ----------------------------------
.PHONY: default build bash dataset


default: run

# Build the Docker image
build:
	@echo "📦 Building Docker image $(DOCKER_IMAGE_NAME) ..."
	@docker build $(DOCKER_BUILD_PARAMS)

run: build
	@echo "🚀 Running CenterNet Inference demo in $(PRECISION) ..."
	@[ -n "$$DISPLAY" ] && xhost +local:root > /dev/null 2>&1 || true
	@docker run $(DOCKER_RUN_PARAMS) bash -c "python3 ./detector.py \
			-m $(MODEL_PATH) \
			-i $(INPUT_PATH) \
			-d $(DEVICE) \
			-p $(PRECISION)"


dataset: build
	@echo "🚀 Preparing the COCO dataset ..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c ./prepare_coco_dataset.sh
			
openvino_export: build
	@echo "🚀 Exporting PyTorch model to OpenVINO ..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c "python3 ./openvino_export.py \
			--config $(CONFIG_PATH) \
			--checkpoint $(CHECKPOINT_PATH) \
			--resolution $(INPUT_SIZE)  \
			--output_dir $(MODELS_DIR)/FP32/"

quantize: build dataset
	@echo "⚙️ Quantizing model to INT8 ..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c "python3 ./quantize.py \
			--model $(MODELS_DIR)/FP32/centernet.xml \
			--dataset $(DATASET_DIR) \
			--resize $(INPUT_SIZE) $(INPUT_SIZE) \
			--output $(MODELS_DIR)/INT8/centernet.xml"


bash: build
	@docker run $(DOCKER_RUN_PARAMS) bash
	
	
