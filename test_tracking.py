import os
import sys
import time
import numpy as np
import cv2
from tracking.sort import Sort


class Detector:
    """
    OpenCV YOLO - modified from PyImageSearch's code
    """
    def __init__(self, conf_threshold=0.5, nms_threshold=0.3, yolo_dir="saved_models"):
        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([yolo_dir, "coco.names"])
        self.classes = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([yolo_dir, "yolov3.weights"])
        configPath = os.path.sep.join([yolo_dir, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def detect(self, img):
        """
        :param img:
        :return: boxes [box1, box2, ..., boxn], box format: [x, y, w, h], (x,y) top-left corner
        """
        (H, W) = img.shape[:2]
        # determine only the *output* layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()
        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.conf_threshold:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        # return [[*boxes[i], confidences[i], classIDs[i]] for i in idxs.flatten()]
        return [[*boxes[i], confidences[i], self.classes[classIDs[i]]] for i in idxs.flatten()]


def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    return [x, y, x+w, y+h]


def draw_boxes(img, boxes_xyxy, color, thickness=1):
    for x_min, y_min, x_max, y_max in boxes_xyxy.astype(np.int):
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


if __name__ == '__main__':
    detector = Detector()
    tracker = Sort()

    test_video = sys.argv[1]
    vs = cv2.VideoCapture(test_video)

    while True:
        ret, frame = vs.read()
        if ret:
            detections = detector.detect(frame)
            detected_boxes = []
            for det in detections:
                box_xywh = det[:4]
                box_xyxy = xywh_to_xyxy(box_xywh)
                detected_boxes.append(box_xyxy)
            detected_boxes = np.array(detected_boxes)
            tracks = tracker.update(detected_boxes)
            frame = draw_boxes(frame, detected_boxes, color=[0, 255, 0])
            frame = draw_boxes(frame, tracks[:, :4], color=[0, 255, 255])
            cv2.imshow("track", frame)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
            break
    vs.release()