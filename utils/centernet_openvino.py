"""
 Copyright (c) 2019-2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2
import time
import threading
import queue
from collections import deque
from time import perf_counter
import numpy as np
from numpy.lib.stride_tricks import as_strided
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints


class CenterNet():
    def __init__(self, model_path, device, callback, confidence_threshold=0.5):

        self.device = device
        self.callback = callback
        self.confidence_threshold = confidence_threshold

        self.ov_core = ov.Core()
        
        config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}

        self.model = self.ov_core.compile_model(model_path, device, config)
        _, _, self.h, self.w = self.model.inputs[0].shape
        self.input_name = self.model.inputs[0]
        self.output_names = [output.any_name for output in self.model.outputs]
        self.nb_outputs = len(self.output_names)

        self.latencies = deque(maxlen=100)
        self.ireqs = ov.AsyncInferQueue(self.model)
        self.ireqs.set_callback(self.ov_callback)
        
        self.frames_number = 0
        self.start_time = perf_counter()
        
        # Queue for postprocessing results
        self.postprocess_queue = queue.Queue()
        
        # Flag to control the postprocessing thread
        self.running = True
        
        # Start the postprocessing thread
        self.postprocess_thread = threading.Thread(target=self.postprocess_worker, daemon=True)
        self.postprocess_thread.start()

    def infer(self, frame):

        if  self.frames_number == 0:
            self.start_time = perf_counter()

        self.frames_number += 1

        while not self.ireqs.is_ready() and self.running :
            time.sleep(0.001)

        if self.running:
            input_frame = self.preprocess(frame)

            self.ireqs.start_async(inputs={self.input_name: input_frame}, userdata=frame)

        return self.running

    def ov_callback(self, infer_request, userdata):
        """OpenVINO callback that puts results in queue for postprocessing"""
        original_frame = userdata  
        self.latencies.append(infer_request.latency)
        
        # Put inference results in queue for postprocessing thread
        self.postprocess_queue.put((infer_request.results, original_frame))
           
    def postprocess_worker(self):
        """Worker thread for postprocessing and callback execution"""
        while self.running:
            try:
                # Get results from queue with timeout
                results, original_frame = self.postprocess_queue.get(timeout=0.1)
                
                # Perform postprocessing
                dets = self.postprocess(results, original_frame.shape)
                
                # Call the callback with results
                self.running = self.callback(dets, original_frame, self.fps(), self.latency())

                # Mark task as done
                self.postprocess_queue.task_done()
                
            except queue.Empty:
                # Continue if queue is empty
                continue
            except Exception as e:
                print(f"Error in postprocessing thread: {e}")
                # Mark task as done even if there was an error
                try:
                    self.postprocess_queue.task_done()
                except:
                    pass

    def stop(self):
        """Stop the postprocessing thread"""
        self.running = False
        if self.postprocess_thread.is_alive():
            self.postprocess_thread.join(timeout=1.0)

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop()
         
    def preprocess(self, frame):
        input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input = cv2.resize(input, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        input = (input - mean) / std  # Applied per channel
        input = input[np.newaxis, ...]  # (1, H, W, C)
        input = input.transpose(0, 3, 1, 2)  # (1, C, H, W)
        return input

    def postprocess(self, outputs, original_shape):

        detections = []
        if self.nb_outputs == 3:
            heat = outputs[next(k for k in outputs if k.get_any_name() == self.output_names[0])][0]
            wh = outputs[next(k for k in outputs if k.get_any_name() == self.output_names[1])][0]
            reg = outputs[next(k for k in outputs if k.get_any_name() == self.output_names[2])][0]

            heat = np.exp(heat)/(1 + np.exp(heat))
            height, width = heat.shape[1:3]
            num_predictions = 100

            heat = self.nms(heat)
            scores, inds, clses, ys, xs = self.topk(heat, K=num_predictions)
            reg = self.tranpose_and_gather_feat(reg, inds)

            reg = reg.reshape((num_predictions, 2))
            xs = xs.reshape((num_predictions, 1)) + reg[:, 0:1]
            ys = ys.reshape((num_predictions, 1)) + reg[:, 1:2]

            wh = self.tranpose_and_gather_feat(wh, inds)
            wh = wh.reshape((num_predictions, 2))
            clses = clses.reshape((num_predictions, 1))
            scores = scores.reshape((num_predictions, 1))
            bboxes = np.concatenate((xs - wh[..., 0:1] / 2,
                                     ys - wh[..., 1:2] / 2,
                                     xs + wh[..., 0:1] / 2,
                                     ys + wh[..., 1:2] / 2), axis=1)
            dets = np.concatenate((bboxes, scores, clses), axis=1)
            mask = dets[..., 4] >= self.confidence_threshold
            filtered_detections = dets[mask]

            scale = max(original_shape)
            center = np.array(original_shape[:2])/2.0

            dets = self.transform(filtered_detections, np.flip(center, 0), scale, height, width)
            for det in dets:
                xmin, ymin, xmax, ymax, score, class_id = det
                xmin = max(int(xmin), 0)
                ymin = max(int(ymin), 0)
                xmax = min(int(xmax), original_shape[1])
                ymax = min(int(ymax), original_shape[0])
                class_id = int(class_id)
                detections.append([xmin, ymin, xmax, ymax, score, class_id])
        else:
            np.set_printoptions(threshold=np.inf, linewidth=80, suppress=True, precision=2)
            dets = outputs[self.output_names[0]][0]
            labels = outputs[self.output_names[1]][0]
            for i, det in enumerate(dets):
                score = det[4]  # Score is the 5th element
                if score < self.confidence_threshold:
                    continue
                print(det, labels[i])  # Fixed typo: lables -> labels
            
        return detections 

    def get_affine_transform(self, center, scale, rot, output_size, inv=False):

        def get_dir(src_point, rot_rad):
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            src_result = [0, 0]
            src_result[0] = src_point[0] * cs - src_point[1] * sn
            src_result[1] = src_point[0] * sn + src_point[1] * cs
            return src_result

        def get_3rd_point(a, b):
            direct = a - b
            return b + np.array([-direct[1], direct[0]], dtype=np.float32)

        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w, dst_h = output_size

        rot_rad = np.pi * rot / 180
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

        dst = np.zeros((3, 2), dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :], src[1, :] = center, center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def gather_feat(self, feat, ind):
        dim = feat.shape[1]
        ind = np.expand_dims(ind, axis=1)
        ind = np.repeat(ind, dim, axis=1)
        feat = feat[ind, np.arange(feat.shape[1])]
        return feat

    def tranpose_and_gather_feat(self, feat, ind):
        feat = np.transpose(feat, (1, 2, 0))
        feat = feat.reshape((-1, feat.shape[2]))
        feat = self.gather_feat(feat, ind)
        return feat

    def topk(self, scores, K=40):
        cat, _, width = scores.shape

        scores = scores.reshape((cat, -1))
        topk_inds = np.argpartition(scores, -K, axis=1)[:, -K:]
        topk_scores = scores[np.arange(scores.shape[0])[:, None], topk_inds]

        topk_ys = (topk_inds / width).astype(np.int32).astype(float)
        topk_xs = (topk_inds % width).astype(np.int32).astype(float)

        topk_scores = topk_scores.reshape((-1))
        topk_ind = np.argpartition(topk_scores, -K)[-K:]
        topk_score = topk_scores[topk_ind]
        topk_clses = topk_ind / K
        topk_inds = self.gather_feat(
            topk_inds.reshape((-1, 1)), topk_ind).reshape((K))
        topk_ys = self.gather_feat(topk_ys.reshape((-1, 1)), topk_ind).reshape((K))
        topk_xs = self.gather_feat(topk_xs.reshape((-1, 1)), topk_ind).reshape((K))

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def nms(self, heat, kernel=3):
        def max_pool2d(A, kernel_size, padding=1, stride=1):
            A = np.pad(A, padding, mode='constant')
            output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                            (A.shape[1] - kernel_size)//stride + 1)
            kernel_size = (kernel_size, kernel_size)
            A_w = as_strided(A, shape=output_shape + kernel_size,
                             strides=(stride*A.strides[0],
                                      stride*A.strides[1]) + A.strides)
            A_w = A_w.reshape(-1, *kernel_size)

            return A_w.max(axis=(1, 2)).reshape(output_shape)

        pad = (kernel - 1) // 2

        hmax = np.array([max_pool2d(channel, kernel, pad) for channel in heat])
        keep = (hmax == heat)
        return heat * keep

    def transform_preds(self, coords, center, scale, output_size):
        def affine_transform(pt, t):
            new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
            new_pt = np.dot(t, new_pt)
            return new_pt[:2]

        target_coords = np.zeros(coords.shape)
        trans = self.get_affine_transform(center, scale, 0, output_size, inv=True)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
        return target_coords

    def transform(self, dets, center, scale, height, width):
        dets[:, :2] = self.transform_preds(
            dets[:, 0:2], center, scale, (width, height))
        dets[:, 2:4] = self.transform_preds(
            dets[:, 2:4], center, scale, (width, height))
        return dets


    def fps(self):
        return self.frames_number/(perf_counter() - self.start_time)

    def latency(self):
        if len(self.latencies) > 0:
            return np.mean(self.latencies)
        else:
            return 0