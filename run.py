import os
import sys
import time
import numpy as np
import cv2
import numba
from tracking.sort import Sort
from detection.wrapper import VehicleDetector
import json
from counting.hausdorff_dist import hausdorff_distance

def draw_boxes(img, boxes_xyxy, texts, color, thickness=1):
    if texts is not None:
        for (x_min, y_min, x_max, y_max), t in zip(boxes_xyxy.astype(np.int), texts):
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
            cv2.putText(img, str(t), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    else:
        for x_min, y_min, x_max, y_max in boxes_xyxy.astype(np.int):
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    return img
def draw_arrow(img,point,color):
    cv2.polylines(img,[np.array(point)],True,color,thickness=2)
    p0_x = point[-2][0]
    p0_y = point[-2][1]
    p1_x = point[-1][0]
    p1_y = point[-1][1]
    cv2.arrowedLine(img, (int(p0_x), int(p0_y)), (int(p1_x), int(p1_y)), (0,255,0),5)
    return img

def draw_one_boxes(img, bbox, text, color, thickness=1):
    (x_min, y_min,x_max, y_max) = bbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    cv2.putText(img, "ID" + str(text), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

# load file config take ROI
with open("counting/zones-movement_paths/cam_01.json","r") as file :
    data = json.load(file)
    # get param image
    path_image = data["imagePath"]
    width = data["imageWidth"]
    height = data["imageHeight"]
    # get ROI
    zone = data["shapes"][0]["points"]
ROI = [[int(x),int(y)] for x,y in zone]

# movement_1 
tracklets_1 =  [[347, 213],[338, 230],[329, 251],[320, 271],[309, 294],[300, 315],[291, 336],
 [283, 355],[276, 373],[265, 393],[255, 417],[249, 434],[244, 451],[232, 467],[225, 485]]
# movement_2
tracklets_2 = [[832, 465],[827, 451],[820, 434],[812, 415],[803, 391],[797, 374],[789, 359],[786, 341],
                [777, 321],[770, 302],[764, 287],[756, 271],[751, 256],[744, 244],[739, 234]]
# load mask image
cam_conf = "counting/cam_configs/cam-1.json"
tipical_trajs = {}
with open(cam_conf, 'r') as fc:
    movements = json.load(fc)
    for movement_id, movement_info in movements.items():
        tracklets = movement_info['tracklets']
        tipical_trajs[movement_id] = tracklets
#load mask image
cam_mask = os.path.join('counting/mask', "cam_01.png")
mask = cv2.imread(cam_mask)
h, w, c = mask.shape

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


# save tracking from tracks
def get_track(tracks,frameID):
    global trackings
    if len(tracks) > 0 :
        for idx in range(len(tracks)):
            xmin, ymin, xmax, ymax = tracks[idx, :4]
            xmin, ymin, xmax, ymax = int(xmin),int(ymin),int(xmax),int(ymax)
            trackID = tracks[idx, 4].astype(int)
            cx = int((xmax-xmin)/2)
            cy = int((ymax-ymin)/2)
            if trackID not in trackings :
                trackings[trackID] = {'startframe' : frameID,
                                     'endframe': frameID,
                                     'bbox' : [[frameID, xmin, ymin, xmax, ymax, 2]],
                                      'tracklet' : [[cx, cy]]
                                               
                                     }# fixed classes
            else :
                trackings[trackID]['endframe'] = frameID
                trackings[trackID]["bbox"].append([frameID, xmin, ymin, xmax, ymax, 2])
                trackings[trackID]["tracklet"].append([cx, cy])

# PARAM
dist_thr = 300
angle_thr = 30
min_length = 10

detector = VehicleDetector(device='0')  # select gpu:0
tracker = Sort(max_age=15)

test_video = "/media/thorpham/PROJECT/AIC_2020_Challenge_Track-1/thor/data/videos/cam_01.mp4"
vid_writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))

vs = cv2.VideoCapture(test_video)
framID = 0
trackings = {}
while True:
    ret, frame = vs.read()
    imshow = frame.copy()
    if ret:
        framID += 1
        cv2.polylines(imshow,[np.array(ROI)],True,(0,0,255),thickness=4)
        start = time.time()
        detections = detector.detect(frame)
        detect_timestamp = time.time()

        detected_boxes = []
        for det in detections:
            box_xyxy = det[:4]
            detected_boxes.append(box_xyxy)
        detected_boxes = np.array(detected_boxes)
        tracks = tracker.update(detected_boxes)
        track_timestamp = time.time()
        get_track(tracks,framID)
#         print(trackings)
        ##split tracklets
        tracks_end_in_roi, tracks_start_in_roi, tracks_too_short = check_tracks_with_roi(trackings, mask)
        trackids = sorted([k for k in trackings.keys()])
        for trackid in trackids:
            if len(trackings[trackid]['tracklet']) < min_length:
                continue
            track_traj = trackings[trackid]['tracklet']
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
                        
                        
        # track_h = trackings[trackid]['bbox'][0][4] - trackings[trackid]['bbox'][0][2]
        # if abs(track_traj[-1][1] - track_traj[0][1]) < track_h:
        #     continue
#         mv_idx = min_idx.split('_')[1]
        #get last frameid in roi
        bboxes = trackings[trackid]['bbox']
        bboxes.sort(key=lambda x: x[0])

        dst_frame = bboxes[0][0]
        last_bbox = bboxes[-1]
        if check_bbox_overlap_with_roi(last_bbox, mask) == True:
            draw_one_boxes(imshow,last_bbox[1:5],str(trackid), (255,0,255), thickness=3)
            print("count ID: "  + str(trackid))
            dst_frame = last_bbox[0]
        else:
            for i in range(len(bboxes) - 2, 0, -1):
                bbox = bboxes[i]
                if check_bbox_overlap_with_roi(bbox, mask) == True:
                    dst_frame = bbox[0]
                    # draw_one_boxes(imshow,last_bbox[1:5],str(trackid), (255,0,255), thickness=3)
                    break
                else:
                    continue
#         print("frame {}: detection time: {} s, trackin time: {} s".format(tracker.frame_count, detect_timestamp - start,
#                                                                           track_timestamp - detect_timestamp))

        # frame = draw_boxes(frame, detected_boxes, color=[0, 255, 0])
        frame = draw_boxes(imshow, tracks[:, :4], tracks[:, 4].astype(int), color=[0, 255, 255])
        frame = draw_arrow(imshow,tracklets_1,(255,0,0))
        frame = draw_arrow(imshow,tracklets_2[::-1],(255,255,0))
        # print(tracks)
        vid_writer.write(imshow)

        cv2.imshow("track", imshow)

        # fix frame rate at 30 fps
        stop_time = time.time()
        wait_time = int(33.33 - (stop_time - start)*1000)
#         cv2.waitKey(max(wait_time, 1))
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
vs.release()