from typing import List
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial.distance import mahalanobis, cosine
from tabulate import tabulate
from filterpy.kalman import KalmanFilter


def convert_bbox_to_z(bbox):
    """
    :param bbox: [x_min, y_min, x_max, y_max]
    :return: [cx, cy, w/h, h]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.
    cy = bbox[1] + h / 2.
    return np.array([cx, cy, w/h, h]).reshape((4, 1))


def convert_x_to_bbox(x):
    """
    :param x: [cx, cy, w/h, h]
    :return bbox: [x_min, y_min, x_max, y_max]
    """
    cx, cy, a, h = x
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class Detection:
    detection_count = 1

    def __init__(self, detection, feature=None):
        """
        :param detection: [x_min, y_min, x_max, y_max, conf, cls]
        """
        self.classId = int(detection[5])
        self.box = detection[:4]
        self.confidence = detection[4]
        self.feature = feature
        self.z = convert_bbox_to_z(self.box)
        self.detID = Detection.detection_count
        Detection.detection_count += 1

    def __str__(self):
        return "Detection {} - {}".format(self.detID, self.box)

    def __repr__(self):
        return self.__str__() + "\n"


class Track:
    track_count = 1

    def __init__(self, detection: Detection):
        self.classId = detection.classId
        self.feature = detection.feature

        self.status = 0  # 0: unconfirmed, 1: confirmed, 2: to be deleted
        self.time_since_update = 0
        self.age = 0
        self.hit_streak = 0  # number of consecutive get matched to a detection
        self.trackId = Track.track_count
        Track.track_count += 1

        # kalman filter
        # state vector x: [cx, cy, aspect_ratio, h, dcx, dcy, da, dh], where aspect_ratio = w/h
        # measurement vector z: [cx, cy, aspect_ratio, h]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # state transition matrix
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0, 0],      # cx_pred = cx_prev + dcx
             [0, 1, 0, 0, 0, 1, 0, 0],      # cy_pred = cy_prev + dcy
             [0, 0, 1, 0, 0, 0, 1, 0],      # a_pred = a_prev + da
             [0, 0, 0, 1, 0, 0, 0, 1],      # h_pred = h_prev + dh
             [0, 0, 0, 0, 1, 0, 0, 0],      # dcx_pred = dcx_prev
             [0, 0, 0, 0, 0, 1, 0, 0],      # dcy_pred = dcy_prev
             [0, 0, 0, 0, 0, 0, 1, 0],      # da_pred = da_prev
             [0, 0, 0, 0, 0, 0, 0, 1]])     # da_pred = da_prev

        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0],      # z = Hx
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # initialize state
        self.kf.x[:4] = convert_bbox_to_z(detection.box)
        self.time_since_update = 0

    def update(self, det: Detection):
        self.time_since_update = 0
        self.hit_streak += 1
        self.feature = det.feature
        self.kf.update(convert_bbox_to_z(det.box))
        self.classId = det.classId

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def get_box(self):
        return convert_x_to_bbox(self.kf.x[:4])

    def __str__(self):
        return "Track {} - x_min = {}, y_min = {}, x_max = {}, y_max ={}".format(self.trackId, *self.get_box()) + \
               "dx = {}, dy = {}".format(self.kf.x[4], self.kf.x[5])

    def __repr__(self):
        return self.__str__() + "\n"


class Tracker(object):
    def __init__(self, max_age=15, min_hits=3, max_distance=50.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.active_tracks = set()
        self.confirmed_tracks = set()
        self.frame_count = 0

    def matching(self, dets: List[Detection], verbose=False):
        active_tracks = list(self.active_tracks)
        matched = [[], []]
        unmatched_dets = []
        unmatched_tracks = []

        distance_matrix = np.zeros((len(active_tracks), len(dets)))
        appearance_matrix = np.zeros((len(active_tracks), len(dets)))
        for row, trk in enumerate(active_tracks):
            for col, det in enumerate(dets):
                distance_matrix[row, col] = mahalanobis(trk.kf.x[:4], det.z, np.linalg.inv(trk.kf.P[:4, :4]))
                appearance_matrix[row, col] = cosine(trk.feature, det.feature)

        # normalize cost matrix
        # distance_matrix /= np.sum(distance_matrix)
        # appearance_matrix /= np.sum(appearance_matrix)

        # if distance_matrix.any():
        #     distance_matrix /= np.max(distance_matrix)
        if verbose:
            # print("distance_matrix:\n", distance_matrix)
            print("Mahalanobis distance")
            print_cost_matrix(distance_matrix, active_tracks, dets)
            print("cosine distance (histogram)")
            print_cost_matrix(appearance_matrix, active_tracks, dets)
            # print_cost_matrix(np.log(distance_matrix), active_tracks, dets)

        trackIds, detectionIds = linear_assignment(distance_matrix)
        # trackIds, detectionIds = linear_assignment(appearance_matrix)

        # custom_matrix = 0.5 * distance_matrix + 0.5 * appearance_matrix
        # trackIds, detectionIds = linear_assignment(custom_matrix)

        for trkId, detId in zip(trackIds, detectionIds):
            if distance_matrix[trkId, detId] < self.max_distance:
                matched[0].append(active_tracks[trkId])
                matched[1].append(dets[detId])
            else:
                unmatched_tracks.append(active_tracks[trkId])
                unmatched_dets.append(dets[detId])

        for i, det in enumerate(dets):
            if i not in detectionIds:
                unmatched_dets.append(det)

        for i, trk in enumerate(active_tracks):
            if i not in trackIds:
                unmatched_tracks.append(active_tracks[i])
        if not matched[0]:
            matched = []
        return matched, unmatched_dets, unmatched_tracks

    def update(self, dets: List[Detection], verbose=False):
        self.frame_count += 1
        if verbose:
            print("Processing frame ", self.frame_count)
            print("detections:\n", dets)
        for trk in self.active_tracks:
            trk.predict()

        matched, unmatched_dets, unmatched_trks = self.matching(dets, verbose=verbose)
        if matched:
            for trk, det in zip(matched[0], matched[1]):
                trk.update(det)
                if verbose:
                    print("match track {} to {}".format(trk.trackId, det))

                # set track status to "confirmed" if it gets match for "min_hits" consecutive frames
                if trk.hit_streak >= self.min_hits:
                    trk.status = 1
                    self.confirmed_tracks.add(trk)

        for det in unmatched_dets:
            new_track = Track(det)
            self.active_tracks.add(new_track)
            if verbose:
                print("create new track:\n", new_track)

        for trk in unmatched_trks:
            if trk.time_since_update >= self.max_age:
                self.active_tracks.remove(trk)
        if verbose:
            print("active track (Updated)\n", self.active_tracks)
            print("___________________________________________\n")


def print_cost_matrix(cost_matrix, tracks: List[Track], detections: List[Detection]):
    headers = [""] + [str(det.detID) for det in detections]
    table = [[trk.trackId, *values] for trk, values in zip(tracks, cost_matrix)]
    print(tabulate(table, headers=headers))