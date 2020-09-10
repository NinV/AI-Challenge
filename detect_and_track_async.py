import sys
import time
from multiprocessing import Process, Queue
import cv2

from tracking.DeepSORT import Tracker, Detection
from detection.wrapper import VehicleDetector


def detect(detector, frame_batch):
    global detections_queue
    start = time.time()
    detections = detector.detect_multiple(frame_batch)
    # print("detection time:", time.time() - start)

    for detection_per_frame in detections:
        detection_per_frame = [Detection(det) for det in detection_per_frame]
        detections_queue.put(detection_per_frame)
    # print("new batch processed")


def track():
    global detections_queue, tracker

    while True:
        if not detections_queue.empty():
            # print("Update tracking. Queue size:", detections_queue.qsize())
            detection_per_frame = detections_queue.get()
            if detection_per_frame:
                tracker.update(detection_per_frame, visual_tracking=False, verbose=False)
            else:
                break


if __name__ == '__main__':
    detector = VehicleDetector(weights="detection/yolov5/weights/best_yolov5l.pt", device="cuda:0")
    tracker = Tracker(max_age=5)
    detections_queue = Queue()

    test_video = sys.argv[1]
    vs = cv2.VideoCapture(test_video)

    frame_count = 0
    total = 1000
    batch_size = 14

    frame_batch = []
    track_p = Process(target=track)
    track_p.start()
    start = time.time()
    while frame_count < total:
        ret, frame = vs.read()
        if ret:
            frame_count += 1
            frame_batch.append(frame)
            if len(frame_batch) == batch_size:
                detect(detector, frame_batch)
                frame_batch = []
        else:
            break

    detections_queue.put(False)
    vs.release()
    track_p.join()
    print(time.time() - start)
