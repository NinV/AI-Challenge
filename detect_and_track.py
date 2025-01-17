import argparse
from glob import glob
import os
from time import time
import cv2
from tracking.DeepSORT import Tracker, Detection
from detection.wrapper import VehicleDetector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="path to videos folder")
    parser.add_argument("-o", "--output", default="tmp", help="output destination")
    parser.add_argument("-w", "--weights", default="detection/yolov5/weights/best_yolov5l.pt")
    parser.add_argument("--device", default="cuda:0", help="device for detector")
    parser.add_argument("--conf", default=0.5, help="detector confidence threshold")
    parser.add_argument("--max_age", default=5, help="tracker max age")
    return parser.parse_args()


def get_camId(path):
    return os.path.split(path)[-1].split(".")[0]


def write_track(fp, camId, frameId, track):
    x_min, y_min, x_max, y_max, trackId, classId = track
    fp.write(
        "{} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {}\n".format(camId, frameId, int(trackId), x_min, y_min, x_max, y_max,
                                                           int(classId)))


if __name__ == '__main__':
    start = time()
    args = parse_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    detector = VehicleDetector(weights="detection/yolov5/weights/best_yolov5l.pt", device=args.device)  # select gpu:0
    video_files = sorted(glob(os.path.join(args.input, "*")))

    for vf in video_files:
        camId = get_camId(vf)
        print("[INFO] processing", camId)
        with open(os.path.join(args.output, "{}.txt".format(camId)), "w") as f:
            vs = cv2.VideoCapture(vf)
            tracker = Tracker(max_age=args.max_age)

            # frame_count = 0
            # total_frame = 100
            # while frame_count < total_frame:
            while True:
                ret, frame = vs.read()
                if ret:
                    # frame_count += 1
                    detections = detector.detect(frame)
                    detections = [Detection(det) for det in detections]
                    tracker.update(detections, visual_tracking=False, verbose=False)
                    for trk in tracker.active_tracks:
                        if trk.time_since_update == 0 and trk.status == 1:
                            x_min, y_min, x_max, y_max = trk.get_box()
                            track = x_min[0], y_min[0], x_max[0], y_max[0], trk.trackId, trk.classId,
                            write_track(f, camId, tracker.frame_count, track)
                else:
                    break
            vs.release()
    print("Total time spent:", time() - start)