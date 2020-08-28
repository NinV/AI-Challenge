import os
import sys
import time
import numpy as np
import cv2
from tracking.sort import Sort
from detection.wrapper import VehicleDetector


def draw_boxes(img, boxes_xyxy, texts, color, thickness=1):
    if texts is not None:
        for (x_min, y_min, x_max, y_max), t in zip(boxes_xyxy.astype(np.int), texts):
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
            cv2.putText(img, str(t), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    else:
        for x_min, y_min, x_max, y_max in boxes_xyxy.astype(np.int):
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    return img


class_names = ["motorbike", "car", "bus", "truck"]


def track_text(trackIds, classIds):
    texts = []
    for trkid, clsid in zip(trackIds, classIds):
        texts.append("{} - {}".format(int(trkid), class_names[int(clsid)]))

    return texts


if __name__ == '__main__':
    detector = VehicleDetector(device='0')  # select gpu:0
    tracker = Sort(max_age=5)

    test_video = sys.argv[1]
    vs = cv2.VideoCapture(test_video)

    while True:
        ret, frame = vs.read()
        if ret:
            start = time.time()
            detections = detector.detect(frame)
            detect_timestamp = time.time()

            tracks = tracker.update(detections)
            track_timestamp = time.time()

            print("frame {}: detection time: {} s, trackin time: {} s".format(tracker.frame_count, detect_timestamp - start,
                                                                              track_timestamp - detect_timestamp))

            # frame = draw_boxes(frame, detected_boxes, color=[0, 255, 0])
            if tracks.any():
                frame = draw_boxes(frame, tracks[:, :4], track_text(tracks[:, 4], tracks[:, 5]), color=[0, 255, 255])

            # print(tracks)
            cv2.imshow("track", frame)

            # fix frame rate at 30 fps
            stop_time = time.time()
            wait_time = int(33.33 - (stop_time - start)*1000)
            cv2.waitKey(max(wait_time, 1))

            # cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
            break
    vs.release()

