import argparse
from glob import glob
import os
from time import time
from multiprocessing import Process, Queue
import cv2
from tracking.DeepSORT import Tracker, Detection
from detection.wrapper import VehicleDetector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="path to videos folder")
    parser.add_argument("-w", "--weights", default="detection/yolov5/weights/best_yolov5l.pt")
    parser.add_argument("-o", "--output", default="tmp", help="output destination")
    parser.add_argument("--device", default="cuda:0", help="device for detector")
    parser.add_argument("--conf", default=0.5, help="detector confidence threshold", type=float)
    parser.add_argument("--max_age", default=5, help="tracker max age", type=int)
    parser.add_argument("--batch_size", default=14, type=int)
    return parser.parse_args()


def get_camId(path):
    return os.path.split(path)[-1].split(".")[0]


def write_track(fp, camId, frameId, track):
    x_min, y_min, x_max, y_max, trackId, classId = track
    fp.write(
        "{} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {}\n".format(camId, frameId, int(trackId), x_min, y_min, x_max, y_max,
                                                           int(classId)))


def detect(detector, frame_batch):
    global detections_queue
    detections = detector.detect_multiple(frame_batch)

    for detection_per_frame in detections:
        detection_per_frame = [Detection(det) for det in detection_per_frame]
        detections_queue.put(detection_per_frame)


def track():
    global detections_queue, tracker

    while True:
        if not detections_queue.empty():
            detection_per_frame = detections_queue.get()

            # detection_per_frame can be a list or False
            if isinstance(detection_per_frame, list):
                tracker.update(detection_per_frame, visual_tracking=False, verbose=False)
                for trk in tracker.active_tracks:
                    if trk.time_since_update == 0 and trk.status == 1:
                        x_min, y_min, x_max, y_max = trk.get_box()
                        track = x_min[0], y_min[0], x_max[0], y_max[0], trk.trackId, trk.classId,
                        write_track(f, camId, tracker.frame_count, track)
            else:   # end of video
                break


if __name__ == '__main__':
    start = time()
    args = parse_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    detector = VehicleDetector(weights=args.weights, device=args.device)  # select gpu:0
    video_files = sorted(glob(os.path.join(args.input, "*")))

    for vf in video_files:
        camId = get_camId(vf)
        print("[INFO] processing", camId)
        with open(os.path.join(args.output, "{}.txt".format(camId)), "w") as f:
            vs = cv2.VideoCapture(vf)
            tracker = Tracker(max_age=args.max_age)
            detections_queue = Queue()
            track_p = Process(target=track)
            track_p.start()

            frame_count = 0
            frame_batch = []

            while True:
                ret, frame = vs.read()
                if ret:
                    frame_count += 1
                    frame_batch.append(frame)
                    if len(frame_batch) == args.batch_size:
                        detect(detector, frame_batch)
                        frame_batch = []
                else:
                    break
            detections_queue.put(False)     # send signal to tracker at the end of video
            vs.release()
            track_p.join()
    print("Total time spent:", time() - start)
