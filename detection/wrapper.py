import sys
import os

dir_name = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dir_name, "yolov5"))

import numpy as np
import torch
from .yolov5.models.experimental import attempt_load
from .yolov5.utils.datasets import letterbox
from .yolov5.utils.general import non_max_suppression, scale_coords


class VehicleDetector:
    def __init__(self, weights, conf_thres=0.3, iou_thres=0.5, device="cpu"):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.weights = weights
        self.device = torch.device(device)  # torch.device('cuda:0')
        self.model = attempt_load(self.weights, map_location=self.device)

    def detect(self, img0):
        img = letterbox(img0, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = img.astype(np.float32)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img /= 255.0
        img = torch.from_numpy(img).to(self.device)

        # Inference
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        # Process detections
        result = []
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for obj in det:
                    obj = obj.tolist()
                    x_min, y_min, x_max, y_max, conf, cls = obj
                    result.append((x_min, y_min, x_max, y_max, conf, cls))
        return result

    def detect_multiple(self, images):
        """
        :param images: list of numpy image (n images)
        :return: [detections_image_1, ...,detections_image_n]
                detections_per_image = [x_min, y_min, x_max, y_max, conf, cls]
        """
        batch = []
        original_shape = images[0].shape[:2]
        for img in images:
            img = letterbox(img, new_shape=640, auto=True)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = img.astype(np.float32)  # BGR to RGB
            img = np.ascontiguousarray(img)
            img /= 255.0
            batch.append(img)
        batch = np.array(batch)
        new_shape = batch.shape[2:]

        batch = torch.from_numpy(batch).to(self.device)
        predicts = self.model(batch, augment=False)[0]

        # Apply NMS
        predicts = non_max_suppression(predicts, self.conf_thres, self.iou_thres)

        result = []
        for dets_per_image in predicts:
            if dets_per_image is not None and len(dets_per_image):
                dets_per_image[:, :4] = scale_coords(new_shape, dets_per_image[:, :4], original_shape).round()
                dets_per_image_list = []
                for obj in dets_per_image:
                    obj = obj.tolist()
                    x_min, y_min, x_max, y_max, conf, cls = obj
                    dets_per_image_list.append((x_min, y_min, x_max, y_max, conf, cls))
                result.append(dets_per_image_list)
            else:
                result.append([])    # ensure the number of elements in input batch and output batch are the same
        return result
