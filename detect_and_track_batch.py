import sys
import time
import cv2
from detection.wrapper import VehicleDetector
from tracking.DeepSORT import Tracker, Detection


if __name__ == '__main__':
    detector = VehicleDetector(weights="detection/yolov5/weights/best_yolov5l.pt", device="cuda:0")
    tracker = Tracker(max_age=5)
    test_video = sys.argv[1]
    vs = cv2.VideoCapture(test_video)

    frame_count = 0
    total = 1000
    batch_size = 14

    start = time.time()
    frame_list = []
    while frame_count < total:
        ret, frame = vs.read()
        if ret:
            frame_count += 1
            frame_list.append(frame)

            if frame_count % batch_size == 0:
                detections = detector.detect_multiple(frame_list)
                for detection_per_frame in detections:
                    detection_per_frame = [Detection(det) for det in detection_per_frame]
                    tracker.update(detection_per_frame, visual_tracking=False, verbose=False)
                frame_list = []
        else:
            break
    vs.release()
    print(time.time() - start)
