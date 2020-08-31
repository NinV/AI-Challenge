import os
import sys
import time
from typing import List
import numpy as np
import cv2
# from tracking.sort import Sort
from tracking.DeepSORT import Tracker, Detection, Track
from detection.wrapper import VehicleDetector
from tracking.appearance import histogram

class_names = ["motorbike", "car", "bus", "truck"]


def draw_detection(img, detections: List[Detection], color, thickness=1):
    for det in detections:
        x_min, y_min, x_max, y_max = np.round(det.box).astype(np.int)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        cv2.putText(img, class_names[det.classId], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


color_codes = {"matched_confirmed": [0, 255, 0],        # green
               "matched_unconfirmed": [255, 0, 0],      # blue
               "unmatched_confirmed": [0, 255, 255],    # yellow
               "unmatched_unconfirmed": [0, 0, 255]}    # red


def draw_track(img, tracks: List[Track], thickness=1):
    for trk in tracks:
        x_min, y_min, x_max, y_max = trk.get_box()

        if trk.time_since_update == 0:
            if trk.status == 0:
                status = "matched_unconfirmed"
            else:
                status = "matched_confirmed"
        else:
            if trk.status == 0:
                status = "unmatched_unconfirmed"
            else:
                status = "unmatched_confirmed"

        color = color_codes[status]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        cv2.putText(img, "{} - {}".format(trk.trackId, class_names[trk.classId]), (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


if __name__ == '__main__':
    detector = VehicleDetector(device='0', conf_thres=0.5)  # select gpu:0
    tracker = Tracker()

    test_video = sys.argv[1]
    vs = cv2.VideoCapture(test_video)

    while True:
        ret, frame = vs.read()
        if ret:
            start = time.time()
            detections = detector.detect(frame)
            # detections = [Detection(det) for det in detections]
            detections = [Detection(det, histogram(frame, det[:4])) for det in detections]
            detect_timestamp = time.time()

            tracker.update(detections)
            track_timestamp = time.time()

            # print("frame {}: detection time: {} s, trackin time: {} s".format(tracker.frame_count, detect_timestamp - start,
            #                                                                   track_timestamp - detect_timestamp))

            # frame = draw_detection(frame, detections, color=[0, 255, 0])
            frame = draw_track(frame, tracker.active_tracks)
            cv2.imshow("track", frame)

            # fix frame rate at 30 fps
            stop_time = time.time()
            wait_time = int(33.33 - (stop_time - start)*1000)
            # cv2.waitKey(max(wait_time, 1))
            cv2.waitKey(0)

            # cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
            break
    vs.release()

