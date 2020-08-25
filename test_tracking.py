import os
import sys
import time
import numpy as np
import cv2
from tracking.sort import Sort
from detection.wrapper import VehicleDetector


def draw_boxes(img, boxes_xyxy, color, thickness=1):
    for x_min, y_min, x_max, y_max in boxes_xyxy.astype(np.int):
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


if __name__ == '__main__':
    detector = VehicleDetector(device='0')  # select gpu:0
    tracker = Sort()

    test_video = sys.argv[1]
    vs = cv2.VideoCapture(test_video)

    frame_count = 0
    while True:
        ret, frame = vs.read()
        if ret:
            frame_count += 1
            start = time.time()
            detections = detector.detect(frame)
            detect_timestamp = time.time()

            detected_boxes = []
            for det in detections:
                box_xyxy = det[:4]
                detected_boxes.append(box_xyxy)
            detected_boxes = np.array(detected_boxes)
            tracks = tracker.update(detected_boxes)
            track_timestamp = time.time()

            print("frame {}: detection time: {} s, trackin time: {} s".format(frame_count, detect_timestamp - start,
                                                                              track_timestamp - detect_timestamp))

            frame = draw_boxes(frame, detected_boxes, color=[0, 255, 0])
            frame = draw_boxes(frame, tracks[:, :4], color=[0, 255, 255])
            cv2.imshow("track", frame)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
            break
    vs.release()

