####################################################################
# sort ---> copy
####################################################################

from __future__ import print_function
from numba import jit
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment    <-- deprecated
from scipy.optimize import linear_sum_assignment as linear_assignment
from filterpy.kalman import KalmanFilter


@jit
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (
                bb_gt[3] - bb_gt[1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    :param bbox: [x_min, y_min, x_max, y_max]
    :return: [cx, cy, area_surface, w/h]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox):
        """
        :param bbox: [x_min, y_min, x_max, y_max]
        """
        # state vector x: [cx, cy, area_surface, ratio, dcx, dcy, ds], where ratio = w/h
        # measurement vector z: [cx, cy, area_surface, w/h]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # state transition matrix
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0],                 # cx_pred = cx_prev + dcx
             [0, 1, 0, 0, 0, 1, 0],                 # cy_pred = cy_prev + dcy
             [0, 0, 1, 0, 0, 0, 1],                 # s_pred = s_prev + ds
             [0, 0, 0, 1, 0, 0, 0],                 # ratio_pred = ratio_prev
             [0, 0, 0, 0, 1, 0, 0],                 # dcx_pred = dcx_prev
             [0, 0, 0, 0, 0, 1, 0],                 # dcy_pred = dcy_prev
             [0, 0, 0, 0, 0, 0, 1]])                # ds_pred = ds_prev

        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # initialize state
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.classId = bbox[5]

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if (len(trackers) == 0) or (len(detections) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[1]:
            unmatched_trackers.append(t)

    matches = []
    for m in zip(matched_indices[0], matched_indices[1]):
        m = np.array(m)
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        if not isinstance(dets, np.ndarray):
            dets = np.array(dets)

        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                if dets[d].any():
                    trk.update(dets[d][:][0])
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1, trk.classId])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret), matched, unmatched_dets, unmatched_trks
        return np.empty((0, 5)), matched, unmatched_dets, unmatched_trks
