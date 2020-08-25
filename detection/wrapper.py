import sys
import os

dir_name = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dir_name, "yolov3"))

import torch
import numpy as np
import cv2

from .yolov3.models import Darknet, parse_data_cfg
from .yolov3.utils.datasets import letterbox
from .yolov3.utils.utils import scale_coords, load_classes, torch_utils, non_max_suppression

model_cfg = os.path.join(dir_name, "yolov3/cfg/yolov3-spp.cfg")
weights = os.path.join(dir_name, "yolov3/weights/yolov3-spp.pt")
data_path = os.path.join(dir_name, "yolov3/data/coco.data")
class_names = os.path.join(dir_name, "yolov3/data/coco.names")


class VehicleDetector:
    def __init__(self, img_size=(512, 512), conf_thres=0.3, nms_thres=0.6, device="gpu"):
        # Initialize model
        self.img_size = img_size
        self.model = Darknet(model_cfg, img_size)
        self.device = torch_utils.select_device(device=device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        self.model.to(self.device).eval()
        self.classes = load_classes(class_names)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
    
    def detect(self, img0):
        # preprocess image
        img = self.preprocess(img0)
        
        # Get detections
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.nms_thres)
        result = []
        for i, det in enumerate(pred):
        # Process detections
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                for obj in det:
                    obj = obj.tolist()
                    x_min, y_min, x_max, y_max, conf, cls = obj                                       # only append object with class 'car'
                    result.append((x_min, y_min, x_max, y_max, conf, cls))
        return result
    
    def preprocess(self, img0):
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = img.astype(np.float32)  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img


if __name__ == '__main__':
    image_file = "/media/thorpham/PROJECT/AIC_2020_Challenge_Track-1/thor/yolov3_detect_car/yolov3/images/000001.jpg"
    image = cv2.imread(image_file)
    detector = VehicleDetector(device='cpu')
    pred = detector.detect(image)
    print(pred)
