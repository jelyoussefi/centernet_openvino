# export mmdetection model to OpenVINO
# tested with Python 3.10, OpenVINO 2025.1

# PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" pip install torch==2.1.0 torchvision numpy==1.26.3
# pip install mmcv==2.1.* -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
# pip install openvino==2025.1 mmdet

# usage: python convert_centernet.py
import argparse

import openvino as ov
import torch
from mmdet.apis import init_detector

def get_parser():
	parser = argparse.ArgumentParser(description="Centernet OpenVINO Export")
	parser.add_argument("--config", default="./model_centernet_r18_8xb16-crop512-140e_coco.py", help="Ckeckpoint path")
	parser.add_argument("--checkpoint", default="./model_epoch_8.pth", help="Ckeckpoint path")
	parser.add_argument("--resolution", default=640, type=int, help="Input resolution")
	parser.add_argument("--output_dir", default="./", type=str, help="Output ONNX file")
	return parser


if __name__ == '__main__':
	parser = get_parser()
	args = parser.parse_args()

	config = args.config
	checkpoint = args.checkpoint
	resolution = args.resolution
	output_dir = args.output_dir
	ov_model_path = f"{output_dir}/centernet.xml"
	example_image = torch.randn(1, 3, resolution, resolution)

	#config = "mmdetection/configs/centernet/centernet_r18_8xb16-crop512-140e_coco.py"
	model = init_detector(config=config,
						  checkpoint=checkpoint,
						  device="cpu")

	# veryify that PyTorch inference works as expected
	torch_result = model(example_image)
	#print(torch_result)

	# convert to OpenVINO
	core = ov.Core()
	ov_model = ov.convert_model(model,
								input=[("inputs",[1,3,resolution,resolution])],
								example_input=example_image
								)

	for i, output in enumerate(ov_model.outputs):
		if i == 0:
			output.get_tensor().set_names({"center_heatmap"})
		elif i == 1:
			output.get_tensor().set_names({"width_height"})
		elif i == 2:
			output.get_tensor().set_names({"regression"})

	ov.save_model(ov_model, ov_model_path )

	print(f"OpenVINO model was succesfully exported to {ov_model_path}")

	compiled_model = core.compile_model(ov_model, "CPU")

	# compare OpenVINO and PyTorch inference
	ov_result = compiled_model(example_image)
	#print(ov_result)


