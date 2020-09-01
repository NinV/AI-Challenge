import argparse
from glob import glob
import os
import cv2
from tracking.DeepSORT import Tracker, Detection
from detection.wrapper import VehicleDetector
from tracking.appearance import histogram


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="path to videos folder")
    parser.add_argument("-o", "--output", default="tmp", help="output destination")
    parser.add_argument("--device", default='0', help="device for detector")
    parser.add_argument("--conf", default=0.5, help="detector confidence threshold")
    parser.add_argument("--max_age", default=20, help="tracker max age")
    return parser.parse_args()


def get_camId(path):
    return os.path.split(path)[-1].split(".")[0]


def write_track(fp, camId, frameId, track):
    x_min, y_min, x_max, y_max, trackId, classId = track
    fp.write(
        "{} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {}\n".format(camId, frameId, int(trackId), x_min, y_min, x_max, y_max,
                                                           int(classId)))


if __name__ == '__main__':
    args = parse_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    detector = VehicleDetector(device=args.device, conf_thres=args.conf, img_size=(608, 608))  # select gpu:0
    video_files = sorted(glob(os.path.join(args.input, "*")))

    for vf in video_files:
        camId = get_camId(vf)
        print("[INFO] processing", camId)
        with open(os.path.join(args.output, "{}.txt".format(camId)), "w") as f:
            vs = cv2.VideoCapture(vf)
            tracker = Tracker()
            while True:
                ret, frame = vs.read()
                if ret:
                    detections = detector.detect(frame)
                    detections = [Detection(det) for det in detections]
                    tracker.update(detections, visual_tracking=False)
                    for trk in tracker.active_tracks:
                        if trk.time_since_update == 0 and trk.status == 1:
                            x_min, y_min, x_max, y_max = trk.get_box()
                            track = x_min[0], y_min[0], x_max[0], y_max[0], trk.trackId, trk.classId,
                            write_track(f, camId, tracker.frame_count, track)
                else:
                    break
            vs.release()