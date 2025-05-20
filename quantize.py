#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm

import openvino as ov
import nncf
from nncf.quantization import quantize

from utils.centernet_openvino import CenterNet


def cv2_preprocess(image_path, input_size=(640, 640)):
    """Preprocess image using OpenCV, similar to CenterNet class preprocessing"""
    if isinstance(image_path, str):
        # Load image if path is provided
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image at {image_path}")
            return None
    else:
        # Use the image directly if already loaded
        image = image_path
        
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, input_size, interpolation=cv2.INTER_LINEAR)
    
    # Apply normalization (same as in CenterNet class)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    input_image = (input_image - mean) / std
    
    # Add batch dimension and transpose to NCHW format
    input_image = input_image[np.newaxis, ...]  # (1, H, W, C)
    input_image = input_image.transpose(0, 3, 1, 2)  # (1, C, H, W)
    
    return input_image


def quantize_model(model_path, dataset_path, output_path, input_size=(640, 640), target_device=nncf.TargetDevice.GPU):
    """Quantize the OpenVINO model to INT8 precision using NNCF"""
    print(f"Loading model from {model_path}")

    # Read the model
    core = ov.Core()
    model = core.read_model(model_path)
    
    # Validate dataset path
    images_dir = os.path.join(dataset_path, 'val2017')
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"COCO val2017 directory not found at: {images_dir}")
    
    # Use a limited subset of images for calibration (300 by default)
    max_calibration_samples = 300
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
                  if f.endswith(('.jpg', '.jpeg', '.png'))][:max_calibration_samples]
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    
    print(f"Using {len(image_files)} images for calibration")
    
    # Create a list of dictionaries (one per sample)
    input_name = model.inputs[0].get_any_name()
    print(f"Model input name: {input_name}")
    
    calibration_data = []
    for img_path in tqdm(image_files, desc="Preparing calibration data"):
        # Preprocess image similar to how it's done in CenterNet inference
        try:
            input_tensor = cv2_preprocess(img_path, input_size)
            if input_tensor is not None:
                calibration_data.append({input_name: input_tensor})
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Prepared {len(calibration_data)} valid calibration samples")
    if len(calibration_data) == 0:
        raise ValueError("No valid calibration samples")
    
    # Set up quantization parameters
    print("Starting quantization with target device:", target_device)
    quantization_dataset = nncf.Dataset(calibration_data)
    
    # Perform quantization
    print("Running quantization...")
    quantized_model = quantize(
        model=model,
        calibration_dataset=quantization_dataset,
        target_device=target_device
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Saving quantized model to {output_path}")
    ov.serialize(quantized_model, output_path)
    print("Model saved successfully")
    
    return quantized_model
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Saving quantized model to {output_path}")
    ov.serialize(quantized_model, output_path)
    print("Model saved successfully")
    
    return quantized_model


def main():
    parser = argparse.ArgumentParser(description='Quantize CenterNet model to INT8 using COCO dataset')
    
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to input model (.pt, .onnx, or .xml)')
    
    parser.add_argument('-d', '--dataset', type=str, default='/workspace/dataset/coco/',
                        help='Path to the COCO dataset directory')
    
    parser.add_argument('--resize', type=int, nargs=2, default=[640, 640],
                        help='Resolution to resize frames to (width height)')
    
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path for quantized model (directory for OpenVINO, file for PyTorch)')
    
    parser.add_argument('--target', type=str, default='GPU', choices=['CPU', 'GPU', 'NPU'],
                        help='Target device for quantization: CPU, GPU or NPU')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.model))[0]
        args.output = f"{base_name}_int8.xml"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    try:
        # Quantize the model
        target_device = getattr(nncf.TargetDevice, args.target)
        quantize_model(args.model, args.dataset, args.output, 
                       input_size=tuple(args.resize), target_device=target_device)
        print(f"✅ Quantization successful. Model saved to: {args.output}")
    except Exception as e:
        print(f"❌ Error during quantization: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
