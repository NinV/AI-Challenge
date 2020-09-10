import sys
import time
import cv2
from detection.wrapper import VehicleDetector


if __name__ == '__main__':
    detector = VehicleDetector(weights="detection/yolov5/weights/best_yolov5l.pt", device="cuda:0")
    test_video = sys.argv[1]
    vs = cv2.VideoCapture(test_video)

    frame_count = 0
    total = 100
    batch_size = 12

    start = time.time()
    frame_list = []
    while frame_count < total:
        ret, frame = vs.read()
        if ret:
            frame_count += 1
            frame_list.append(frame)

            if frame_count % batch_size == 0:
                detector.detect_multiple(frame_list)
                frame_list = []
        else:
            break
    vs.release()
    print(time.time() - start)
