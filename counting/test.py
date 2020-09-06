import os
import sys
import json
import cv2
import numpy as np
from hausdorff_dist import hausdorff_distance

def check_bbox_inside_with_roi(bbox, mask):
    #check if four point of bbox all in roi area
    is_inside = True

    x_tl = bbox[1]
    y_tl = bbox[2]
    x_br = bbox[3]
    y_br = bbox[4]

    for x in [x_tl, x_br]:
        if x <= 0 or x >= mask.shape[1]:
            return False

    for y in [y_tl, y_br]:
        if y <= 0 or y >= mask.shape[0]:
            return False

    vertexs = [[x_tl, y_tl], [x_tl, y_br], [x_br, y_tl], [x_br, y_br]]
    for v in vertexs:
        (g, b, r) = mask[v[1], v[0]]
        if (g, b, r) < (128, 128, 128):
            is_inside = False
            return is_inside

    return is_inside

def check_tracks_with_roi(tracks, mask):
    tracks_end_in_roi = []
    tracks_start_in_roi = []
    tracks_too_short = []

    for trackid, track in tracks.items():
        start_bbox = track['bbox'][0]
        end_bbox = track['bbox'][-1]

        if check_bbox_inside_with_roi(start_bbox, mask) == True:
            if track['startframe'] > 3:
                tracks_start_in_roi.append(trackid)

        if check_bbox_inside_with_roi(end_bbox, mask) == True:
            tracks_end_in_roi.append(trackid)

        if track['endframe'] - track['startframe'] < 10:
            if trackid not in tracks_start_in_roi:
                tracks_too_short.append(trackid)
    return tracks_end_in_roi, tracks_start_in_roi, tracks_too_short


def check_bbox_overlap_with_roi(bbox, mask):
    is_overlap = False
    if bbox[1] >= mask.shape[1] or bbox[2] >= mask.shape[0] \
            or bbox[3] < 0 or bbox[4] < 0:
        return is_overlap

    x_tl = bbox[1] if bbox[1] > 0 else 0
    y_tl = bbox[2] if bbox[2] > 0 else 0
    x_br = bbox[3] if bbox[3] < mask.shape[1] else mask.shape[1] - 1
    y_br = bbox[4] if bbox[4] < mask.shape[0] else mask.shape[0] - 1
    vertexs = [[x_tl, y_tl], [x_tl, y_br], [x_br, y_tl], [x_br, y_br]]
    for v in vertexs:
        (g, b, r) = mask[v[1], v[0]]
        if (g, b, r) > (128, 128, 128):
            is_overlap = True
            return is_overlap

    return is_overlap

def is_same_direction(traj1, traj2, angle_thr):
    vec1 = np.array([traj1[-1][0] - traj1[0][0], traj1[-1][1] - traj1[0][1]])
    vec2 = np.array([traj2[-1][0] - traj2[0][0], traj2[-1][1] - traj2[0][1]])
    L1 = np.sqrt(vec1.dot(vec1))
    L2 = np.sqrt(vec2.dot(vec2))
    if L1 == 0 or L2 == 0:
        return False
    cos = vec1.dot(vec2)/(L1*L2)
    angle = np.arccos(cos) * 360/(2*np.pi)
    if angle < angle_thr:
        return True
    else:
        return False

def calc_angle(vec1, vec2):
    vec1 = np.array([traj1[-1][0] - traj1[-5][0], traj1[-1][1] - traj1[-5][1]])
    vec2 = np.array([traj2[-1][0] - traj2[-5][0], traj2[-1][1] - traj2[-5][1]])
    L1 = np.sqrt(vec1.dot(vec1))
    L2 = np.sqrt(vec2.dot(vec2))
    if L1 == 0 or L2 == 0:
        return 90
    cos = vec1.dot(vec2)/(L1*L2)
    if cos > 1:
        return 90
    angle = np.arccos(cos) * 360/(2*np.pi)
    return angle
# load config
cam_conf = "cam_configs/cam_01.json"
tipical_trajs = {}
with open(cam_conf, 'r') as fc:
    movements = json.load(fc)
    for movement_id, movement_info in movements.items():
        tracklets = movement_info['tracklets']
        tipical_trajs[movement_id] = tracklets
#load mask image
cam_mask = os.path.join('mask', "cam_01.png")
mask = cv2.imread(cam_mask)
h, w, c = mask.shape
tracks[trackid] = {'startframe' : frameid,
                'endframe' : frameid,
                'bbox' : [[frameid, x1, y1, x2, y2, label]],
                'tracklet' : [[cx, cy]]}
tracks_end_in_roi, tracks_start_in_roi, tracks_too_short = check_tracks_with_roi(tracks, mask)
trackids = sorted([k for k in tracks.keys()])
#start counting
dist_thr = 300
angle_thr = 30
min_length = 10
results = []

for trackid in trackids:
    if len(tracks[trackid]['tracklet']) < min_length:
        continue
    track_traj = tracks[trackid]['tracklet']
    #calc hausdorff dist with tipical trajs, assign the movement with the min dist
    all_dists_dict = {k: float('inf') for k in tipical_trajs}
    for m_id, m_t in tipical_trajs.items():
        for t in m_t:
            tmp_dist = hausdorff_distance(np.array(track_traj), np.array(t), distance='euclidean')
            if tmp_dist < all_dists_dict[m_id]:
                all_dists_dict[m_id] = tmp_dist

    #check direction
    all_dists = sorted(all_dists_dict.items(), key=lambda k: k[1])
    min_idx, min_dist = None, dist_thr
    for i in range(0, len(all_dists)):
        m_id = all_dists[i][0]
        m_dist = all_dists[i][1]
        if m_dist >= dist_thr: #if min dist > dist_thr, will not assign to any movement
            break
        else:
            if is_same_direction(track_traj, tipical_trajs[m_id][0], angle_thr): #check direction
                min_idx = m_id
                min_dist = m_dist
                break #if match, end
            else:
                continue #direction not matched, find next m_id