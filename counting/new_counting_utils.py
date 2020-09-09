import json
import os

import cv2
import numpy as np


def load_tracking(track_file):
    tracks = {}
    with open(track_file, 'r') as ft:
        lines = [line.strip('\n').split(' ') for line in ft]
        for line in lines:
            frameid = int(line[1])
            trackid = int(line[2])
            x1 = max(0, int(float(line[3])))
            y1 = max(0, int(float(line[4])))
            x2 = min(int(float(line[5])), 1280-1)
            y2 = min(int(float(line[6])), 720-1)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)


            label = line[7]
            if trackid in tracks:
                tracks[trackid]['endframe'] = frameid
                tracks[trackid]['bbox'].append([frameid, x1, y1, x2, y2, label])
                tracks[trackid]['tracklet'].append([cx, cy])
            else:
                tracks[trackid] = {'startframe': frameid,
                                   'endframe': frameid,
                                   'bbox': [[frameid, x1, y1, x2, y2, label]],
                                   'tracklet': [[cx, cy]],
                                   'video_idx': line[0]}
    return tracks


if __name__ == '__main__':
    track_file = '/home/thorpham/AI-Challenge/counting/track_results/cam_09.txt'
    tracks = load_tracking(track_file)


def check_center_inside_with_roi(center, mask):
    if np.sum(mask[center[1], center[0]]) > 100:
        return True
    else:
        return False


def check_bbox_inside_with_roi(bbox, mask):
    # check if four point of bbox all in roi area
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


def check_tracks_with_roi(tracks, mask, center=True):
    tracks_end_in_roi = []
    tracks_start_in_roi = []
    tracks_too_short = []
    checker = check_center_inside_with_roi if center else check_bbox_inside_with_roi
    for trackid, track in tracks.items():
        if center:
            start_frame = track['tracklet'][0]
            end_frame = track['tracklet'][-1]
        else:
            start_frame = track['bbox'][0]
            end_frame = track['bbox'][-1]

        if checker(start_frame, mask):
            if track['startframe'] > 1:
                tracks_start_in_roi.append(trackid)

        if checker(end_frame, mask):
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


def check_only_bbox_overlap_with_roi(bbox, mask):
    x1 = bbox[1]
    y1 = bbox[2]
    x2 = bbox[3]
    y2 = bbox[4]
    points = [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]
    conner = [check_center_inside_with_roi(p, mask) for p in points]
    sum_conner = np.sum(conner)
    if sum_conner > 0 and sum_conner < 4:
        return True
    else:
        return False


def distance_of_2_points(p1, p2):
    return np.sqrt(np.sum(np.square(np.array(p2) - np.array(p1))))


def is_same_direction_by_dist(checked_dir, base_dir):
    endpoint_checked = checked_dir[-1]
    startpoint_base = base_dir[0]
    endpoint_base = base_dir[-1]

    end_dis = distance_of_2_points(endpoint_checked, endpoint_base)
    start_dis = distance_of_2_points(endpoint_checked, startpoint_base)
    if end_dis < start_dis:
        return True
    else:
        return False


def is_same_direction(traj1, traj2, angle_thr):
    vec1 = np.array([traj1[-1][0] - traj1[0][0], traj1[-1][1] - traj1[0][1]])
    vec2 = np.array([traj2[-1][0] - traj2[0][0], traj2[-1][1] - traj2[0][1]])
    L1 = np.sqrt(vec1.dot(vec1))
    L2 = np.sqrt(vec2.dot(vec2))
    if L1 == 0 or L2 == 0:
        return False
    cos = vec1.dot(vec2) / (L1 * L2)
    if abs(cos) > 1:
        cos = np.sign(cos)
    angle = np.arccos(cos) * 360 / (2 * np.pi)
    if angle < angle_thr:
        return True
    else:
        return False


def get_nearest_movement(traj, cand_move, base_movements, angle_thr=30):
    """
    tim huong di gan nhat voi traj trong list of cand_move
    """
    in_vect = np.array([traj[-1][0] - traj[0][0], traj[-1][1] - traj[0][1]])
    L1 = np.linalg.norm(in_vect)

    cand_angle = []
    movements = []
    for movement in cand_move:
        traj_as = base_movements[movement]
        for traj_a in traj_as:
            movements.append(movement)
            vect = np.array([traj_a[-1][0] - traj_a[0][0], traj_a[-1][1] - traj_a[0][1]])
            L2 = np.linalg.norm(vect)
            if L1 == 0 or L2 == 0:
                cand_angle.append(400)
            else:
                cos = in_vect.dot(vect) / (L1 * L2)
                angle = np.arccos(cos) * 360 / (2 * np.pi)
                cand_angle.append(angle)

    if min(cand_angle) > angle_thr:
        return None
    else:
        idx = np.argmin(np.array(cand_angle))
        return movements[idx]


def get_config_mask(data_root, name):
    mask_conf = os.path.join(data_root, 'mask_for_config', name + '.json')
    with open(mask_conf, 'r') as fc:
        base = json.load(fc)
    roi_movements = {}
    height,width = 720,1280
    for m in base["shapes"]:
        name  = m["label"]
        zone = m["points"]
        ROI = [[int(x),int(y)] for x,y in zone]
        mask_mat = np.zeros((height,width,3),np.uint8)
        cv2.fillPoly(mask_mat, [np.array(ROI)], (255, 255, 255))
        roi_movements[name] = mask_mat
    return roi_movements


def get_config_cam_load_tracking(data_root, name):
    # load movements tipical trajs
    cam_conf = os.path.join(data_root, 'cam_configs', name + '.json')
    # with open(cam_conf, 'r') as fc:
    #     tipical_trajs = json.load(fc)

    #  old code
    tipical_trajs = {}
    with open(cam_conf, 'r') as fc:
        movements = json.load(fc)
        for movement_id, movement_info in movements.items():
            tracklets = movement_info['tracklets']
            tipical_trajs[movement_id] = tracklets

    # load mask
    cam_mask = os.path.join(data_root, 'mask', name + '.png')
    mask = cv2.imread(cam_mask)

    # load tracking
    track_file = os.path.join(data_root, 'track_results', name + '.txt')
    tracks = load_tracking(track_file)
    return tipical_trajs, mask, tracks


# def out_of_roi(center, poly):
#     path_array = []
#     for poly_point in poly:
#         path_array.append([poly_point[0], poly_point[1]])
#     path_array = np.asarray(path_array)
#     polyPath = mplPath.Path(path_array)
#     return polyPath.contains_point(center, radius = 0)



# def check_list_point(points, mask):
#     intersection = False
#     p1 = points[0]
#     x0,y0 = p1[0],p1[1]
#     for i in range(1,len(points)) :
#         x1,y1 = points[i][0], points[i][1]
#         check_1 = check_center_inside_with_roi((x1,y1),mask)
#         check_2 = check_center_inside_with_roi((x0,y0),mask)
#         if (check_1==True and check_2==False ) or  (check_1==False and check_2==True ):
#             intersection = True
#             return intersection
#         else :
#             x0,y0 = x1,y1
#     return intersection


# def check_bbox_overlap_with_roi(bbox, mask):
#     is_overlap = False
#     if bbox[1] >= mask.shape[1] or bbox[2] >= mask.shape[0] \
#             or bbox[3] < 0 or bbox[4] < 0:
#         return is_overlap
#
#     x_tl = bbox[1] if bbox[1] > 0 else 0
#     y_tl = bbox[2] if bbox[2] > 0 else 0
#     x_br = bbox[3] if bbox[3] < mask.shape[1] else mask.shape[1] - 1
#     y_br = bbox[4] if bbox[4] < mask.shape[0] else mask.shape[0] - 1
#     vertexs = [[x_tl, y_tl], [x_tl, y_br], [x_br, y_tl], [x_br, y_br]]
#     for v in vertexs:
#         (g, b, r) = mask[v[1], v[0]]
#         if (g, b, r) > (128, 128, 128):
#             is_overlap = True
#             return is_overlap
#
#     return is_overlap


# def calc_angle(vec1, vec2):
#     vec1 = np.array([traj1[-1][0] - traj1[-5][0], traj1[-1][1] - traj1[-5][1]])
#     vec2 = np.array([traj2[-1][0] - traj2[-5][0], traj2[-1][1] - traj2[-5][1]])
#     L1 = np.sqrt(vec1.dot(vec1))
#     L2 = np.sqrt(vec2.dot(vec2))
#     if L1 == 0 or L2 == 0:
#         return 90
#     cos = vec1.dot(vec2) / (L1 * L2)
#     if cos > 1:
#         return 90
#     angle = np.arccos(cos) * 360 / (2 * np.pi)
#     return angle


# def filter_tracks(tracks,mask,d = 20):
#     track_new = {}
#     trackids = sorted([k for k in tracks.keys()])
#     for trackid in trackids :
#         track_center = tracks[trackid]['tracklet']
#         if check_list_point(track_center,mask) :
#             last_bb = tracks[trackid]['bbox'][-1]
#             c_x,c_y = track_center[-1]
#             frameID, xmin,ymin,xmax,ymax, classes = last_bb
#             # front_point = (int(c_x + (xmax-xmin)/2),int(c_y + (ymax-ymin)/2))
#             # points  = [trackid,int(c_x - d),int(c_y - d),int(c_x + d),int(c_y + d)]
#             points = [trackid,xmin,ymin,xmax,ymax]
#             if not check_center_inside_with_roi((c_x,c_y),mask):#
#                 track_new[trackid] = tracks[trackid]
#     return track_new