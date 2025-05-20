import os
import cv2
import torch
import numpy as np
import argparse
import time
import logging
import imghdr  # Used to check if a file is an image

from utils.centernet_openvino import CenterNet


# Print a nice ASCII art banner with application parameters in green
def print_banner(model_path, input_path, device, precision):
    """Print a nice ASCII art banner with application parameters."""
    banner_width = 60
    title = "CenterNet Detector"
    padding = (banner_width - len(title) - 2) // 2
    backend = 'OpenVINO' if model_path.endswith(('.onnx', '.xml')) else 'PyTorch'
    green = "\033[32m"  
    red = "\033[31m"
    reset = "\033[0m"  
    
    print("\n" + "-" * banner_width)
    print(" " * padding + title + " " * padding)
    print("-" * banner_width)
    print(f"  | Model      : {green}{model_path}{reset}")
    print(f"  | Input      : {green}{input_path}{reset}")
    print(f"  | Device     : {red}{device}{reset}")
    print(f"  | Precision  : {green}{precision}{reset}")
    print("-" * banner_width + "\n")


def is_image_file(file_path):
    """Check if the input file is an image based on extension and content."""
    # First check by extension
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    if any(file_path.lower().endswith(ext) for ext in image_extensions):
        return True
    
    # Then try to check by content using imghdr
    try:
        img_type = imghdr.what(file_path)
        return img_type is not None
    except Exception:
        return False


def draw_detections(frame, detections, labels):
    vis_frame = frame.copy()
    colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8).tolist()
    for detection in detections:
        xmin, ymin, xmax, ymax, score, class_id = detection

        x1, y1 = int(xmin), int(ymin)
        x2, y2 = int(xmax), int(ymax)
        cls_id = int(class_id)
        color = (255,0,0) #colors[cls_id % len(colors)]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_frame, '{} {:.1%}'.format(cls_id, score),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 1)

    return vis_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video tracking with TAPIR model')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to the model ')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input video file or image file')
    parser.add_argument('-d', '--device', type=str, default='GPU', choices=['CPU', 'GPU', 'NPU'], help='Device to run the model on: CPU or GPU')
    parser.add_argument('-p', '--precision', type=str, default='FP32', choices=['FP32', 'INT8'], help='Model precision: FP32 or INT8')
    args = parser.parse_args()

    # Display banner with application parameters
    print_banner(args.model, args.input, args.device, args.precision)

    
    labels = ["Capped", "TTSC", "Uncapped", "Foil"]
    
    # Initialize the model
    model = CenterNet(args.model, args.device, confidence_threshold=0.5)
    
    # Check if input is an image file
    is_image = is_image_file(args.input)
    
    gui_enabled = bool(os.environ.get('DISPLAY'))
    
    # FPS calculation variables
    prev_time = time.time()
    fps_avg = 0.0
    alpha = 0.1  # Smoothing factor for moving average
    frame_count = 0

    if is_image:
        pass
       
    else:
        # Handle video input (original behavior)
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open video file or stream: {args.input}")
            exit(1)
            
        start_time = 0  # skip first {start_time} seconds

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                        
            frame_count += 1
            
            start_time = time.time()
            
            # Process the image with the model
            detections = model(frame)
            #print("------------------------------------------")
            #print(detections)
            # Draw bounding boxes on frame
            display_frame = draw_detections(frame, detections, labels)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            fps_avg = alpha * fps + (1 - alpha) * fps_avg if frame_count > 1 else fps
            prev_time = current_time
            
            # Add FPS overlay in top-middle (red text with black background)
            fps_text = f"FPS: {fps_avg:.1f}"
            if gui_enabled:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_thickness = 2
                text_color = (0, 0, 255)  # Red in BGR
                text_size, _ = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
                text_x = (display_frame.shape[1] - text_size[0]) // 2  # Center horizontally
                text_y = text_size[1] + 10  # 10 pixels from top
                # Draw black background rectangle
                bg_padding = 5
                cv2.rectangle(
                    display_frame,
                    (text_x - bg_padding, text_y - text_size[1] - bg_padding),
                    (text_x + text_size[0] + bg_padding, text_y + bg_padding),
                    (0, 0, 0),  # Black in BGR
                    -1
                )
                # Draw FPS text
                cv2.putText(
                    frame,
                    fps_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA
                )

                cv2.imshow('frame', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print(f"\t{fps_text}")        

        cap.release()
    
    # Clean up resources
    cv2.destroyAllWindows()