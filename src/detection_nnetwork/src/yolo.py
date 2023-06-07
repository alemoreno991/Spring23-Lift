import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class Yolo:
    def __init__(self, cfg) -> None:
        self._cfg = cfg

        # Initialize
        set_logging()
        self._device = select_device(cfg.device)

        # Load model
        self._model = attempt_load(cfg.weights, map_location=self._device)  # load FP32 model
        stride = int(self._model.stride.max())  # model stride
        imgsz = check_img_size(cfg.img_size, s=stride)  # check img_size

        if cfg.trace:
            self._model = TracedModel(self._model, self._device, cfg.img_size)

        if self._isHalf():
            self._model.half()  # to FP16

        # Get names and colors
        names = self._model.module.names if hasattr(self._model, 'module') else self._model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self._device.type != 'cpu':
            self._model(torch.zeros(1, 3, imgsz, imgsz).to(self._device).type_as(next(self._model.parameters())))  # run once

        self._old_img_w = self._old_img_h = imgsz
        self._old_img_b = 1

    def detect(self, bgr_frame):
        im0, img = self.bgr2rgb(bgr_frame) 
        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._isHalf() else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self._device.type != 'cpu' and (self._old_img_b != img.shape[0] or self._old_img_h != img.shape[2] or self._old_img_w != img.shape[3]):
            self._old_img_b = img.shape[0]
            self._old_img_h = img.shape[2]
            self._old_img_w = img.shape[3]
            for _ in range(3):
                self._model(img, augment=self._cfg.augment)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self._model(img, augment=self._cfg.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
                    pred, 
                    conf_thres=self._cfg.conf_thres, 
                    iou_thres=self._cfg.iou_thres, 
                    classes=self._cfg.classes, 
                    agnostic=self._cfg.agnostic_nms
                )

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        return pred

    def process_prediction(self, pred):
        confidences = []
        classes = []
        centers = []
        boxes = []

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    left, top = int(xyxy[0]), int(xyxy[1])
                    right, bottom = int(xyxy[2]), int(xyxy[3])
                    cx = (left + right) / 2
                    cy = (top + bottom) / 2

                    confidences.append(conf)
                    classes.append(cls)
                    centers.append([cx, cy])
                    boxes.append(xyxy)

        return classes, boxes, centers, confidences


    #---------------------------------------------------------------------------
    # bgr2rgb: 
    #   - Transforms from BGR to RGB
    #   - Outputs a numpy array 
    #---------------------------------------------------------------------------
    def bgr2rgb(self, bgr_frame):
        img = np.asanyarray(bgr_frame.get_data())

        # Letterbox
        original = img.copy()
        img = img[np.newaxis, :, :, :]        

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        rgb_frame = np.ascontiguousarray(img)

        return original, rgb_frame

    def _isHalf(self):
        return self._device.type != 'cpu' # half precision only supported on CUDA
