import os
import numpy as np
from hausdorff_dist import hausdorff_distance
from new_counting_utils import check_center_inside_with_roi, check_tracks_with_roi, is_same_direction, \
    get_nearest_movement, get_config_mask, get_config_cam_load_tracking, check_bbox_overlap_with_roi


def write_result(path, data):
    with open(path, 'w+') as f:
        f.write(data)

count  = 0


def count_video(data_root, video_name, save_root):
    '''
    data_root: the path which contains (1)the track_results(2)masks(3)AIC20_track1(4)cam-configs
    video_name: the video to count
    save_root: the root path to save counting results
    format for name cam and config:
    line 1: id_cam video_name video_config
    '''
    global count
    roi_movement  = get_config_mask(data_root, video_name)
    base_movements, mask, tracks = get_config_cam_load_tracking(data_root, video_name)
    # get shape mask image
    h, w, c = mask.shape

    # start counting
    dist_thr = 800
    angle_thr = 60
    # print("Old tracks ",len(tracks))
    # tracks = filter_tracks(tracks,mask)
    if video_name in ["cam_01","cam_02","cam_03"]:
        dist_thr = 1200
        angle_thr = 70
    if video_name in ["cam_09", "cam_12"]:
        dist_thr = 1000
        angle_thr = 80


    count += len(tracks)
    print(count)
    trackids = sorted([k for k in tracks.keys()])
    # print("New track ",len(tracks))
    # min_length = 10
    min_length = 6
    results = []
    for trackid in trackids:

        # check length of tracking id
        track_center = tracks[trackid]['tracklet']
        if len(track_center) < min_length:
            continue
        # calc hausdorff dist with tipical trajs, assign the movement with the min dist
        all_dists_dict = {k: float('inf') for k in base_movements}
        for direction_name, paths_point in base_movements.items():
            for path_point in paths_point:
                tmp_dist = hausdorff_distance(np.array(track_center), np.array(path_point), distance='euclidean')
                if tmp_dist < all_dists_dict[direction_name]:
                    all_dists_dict[direction_name] = tmp_dist

        # check direction
        # return [(name, path), ]
        all_dists = sorted(all_dists_dict.items(), key=lambda k: k[1])
        min_idx, min_dist = None, dist_thr
        for i in range(0, len(all_dists)):
            m_id = all_dists[i][0]
            m_dist = all_dists[i][1]
            if m_dist >= dist_thr: #if min dist > dist_thr, will not assign to any movement
                break
            else:
                if is_same_direction(track_center, base_movements[m_id][0], angle_thr): #check direction
                    min_idx = m_id
                    min_dist = m_dist
                    break #if match, end
                else:
                    continue
        # adjust movement id according ''heuristic"
        if video_name == "cam_10":
            cx, cy = track_center[0] # start point
            ex , ey = track_center[-1] # end point
            # print('start', cx, cy, 'end', ex , ey,'='*50)
            # print("min_idx origin :",min_idx)
            cx, cy, ex, ey = cx/w, cy/h, ex/w, ey/h
            if cx > 0.5 and cy > 0.7:
                cand_move = ["movement_5","movement_7","movement_2"]
            elif cx < 0.5 and cy < 0.4:
                cand_move = ["movement_3","movement_10","movement_1"]
            elif cx <0.3 and cy > 0.5 :
                cand_move = ["movement_6","movement_4","movement_8"]
            elif ex < 0.3 and ey < 0.6 :
                cand_move = ["movement_3","movement_5","movement_11"]
            elif ex < 0.5 and ey > 0.7 :
                cand_move = ["movement_6","movement_1","movement_12"]
            elif ex>0.5 and ey <0.45 :
                cand_move = ["movement_4","movement_2","movement_9"]

            if min_idx not in cand_move:
                min_idx = get_nearest_movement(track_center, cand_move, base_movements, 30)

        if video_name in ["cam_14", "cam_15"]:
            cx, cy = track_center[0] # start point
            ex , ey = track_center[-1] # end point
            # print('start', cx, cy, 'end', ex , ey,'='*50)
            # print("min_idx origin :",min_idx)


            cx, cy, ex, ey = cx/w, cy/h, ex/w, ey/h
            if cx < 0.48 and 0.22 < cy < 0.46:
                cand_move = ["movement_1","movement_6"]
            elif cx>0.38 and cy>0.47:
                cand_move = ["movement_2", "movement_4", "movement_5"]
            elif ex<0.5 and ey>0.47:
                cand_move = ["movement_1","movement_4"]
            elif 0.4<ex<0.72 and ey<0.33:
                cand_move = ["movement_2","movement_3"]
            if min_idx not in cand_move:
                min_idx = get_nearest_movement(track_center, cand_move, base_movements, 30)

        if video_name in ["cam_16", "cam_17"]:
            cx, cy = track_center[0] # start point
            ex , ey = track_center[-1] # end point

            # print('start', cx, cy, 'end', ex , ey,'='*50)
            # print("min_idx origin :",min_idx)

            cx, cy, ex, ey = cx/w, cy/h, ex/w, ey/h
            if cx > 0.57 and 0.32 < cy < 0.63:
                min_idx = "movement_3"
            elif ex>0.28 and ey<0.35:
                min_idx = "movement_2"

        if video_name  == "cam_21":
            cx, cy = track_center[0] # start point
            ex , ey = track_center[-1] # end point

            cx, cy, ex, ey = cx/w, cy/h, ex/w, ey/h
            if cy > 0.65:
                cand_move = ["movement_2","movement_6"]
                if min_idx not in cand_move:
                    min_idx = get_nearest_movement(track_center, cand_move, base_movements, 30)

        # movement_id is already defined here


        if min_idx == None:
            # print("-"*20)
            continue
        # cx, cy = track_center[0] # start point
        # ex , ey = track_center[-1] # end point
        # cx, cy, ex, ey = cx/w, cy/h, ex/w, ey/h
        # print("min_idx config : ",min_idx)
        # image =cv2.imread("/home/thorpham/AI-Challenge/counting/zones-movement_paths/cam_10.png")
        # cv2.arrowedLine(image,(int(cx*w), int(cy*h)), (int(ex*w), int(ey*h)),(255,255,255),4)
        # cv2.putText(image, f'{min_idx[-1]}', (int(ex*w), int(ey*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        # cv2.imshow("img",image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # save counting results
        direction_idx = min_idx.split('_')[1]
        # print("---",direction_idx)

        # get last frameid in roi
        bboxes = tracks[trackid]['bbox']
        # sort by frame id
        bboxes.sort(key=lambda x: x[0])

        # frame id for last bbox in ROI
        dst_frame = bboxes[0][0]
        last_bbox = bboxes[-1]

        flag = True


        # find the last bbox of tracking_points that has center inside `ROI` defined by THOR
        # TODO change `has center` by `overlapped`
        # input bbox, direction_id, mask by `ROI` defined by THOR
        if video_name in ["cam_09", "cam_12"]:
            roi_mask = roi_movement["movement_1"]
            if check_bbox_overlap_with_roi(last_bbox, roi_mask):
                dst_frame = last_bbox[0]
                flag = False
            else :
                for i in range(len(bboxes) - 2, 0, -1):
                    bbox = bboxes[i]
                    if check_bbox_overlap_with_roi(bbox, roi_mask):
                        dst_frame = bbox[0]
                        flag = False
                        break
                    else:
                        continue
        if video_name in [
            "cam_01","cam_02","cam_03","cam_04","cam_05","cam_06","cam_07","cam_08","cam_11","cam_13",
            "cam_18","cam_19","cam_23","cam_24","cam_25"
        ]:
            if min_idx in ["movement_1", 'movement_2']:
            # TODO luckily roi_movement_1 == movement_id
                # cx = int((last_bbox[1] + last_bbox[3]) / 2)
                # cy = int((last_bbox[2] + last_bbox[4]) / 2)
                roi_mask = roi_movement[min_idx]
                # if check_center_inside_with_roi((cx, cy), roi_movement["movement_1"]) == True:
                if check_bbox_overlap_with_roi(last_bbox, roi_mask):
                    dst_frame = last_bbox[0]
                    flag = False
                else :
                    for i in range(len(bboxes) - 2, 0, -1):
                        bbox = bboxes[i]
                        if check_bbox_overlap_with_roi(bbox, roi_mask):
                            dst_frame = bbox[0]
                            flag = False
                            break
                        else:
                            continue
        if video_name in ["cam_10"]:
            if min_idx in ["movement_4","movement_2","movement_9"]:
                roi_mask = roi_movement['movement_3']
            elif min_idx in ["movement_3","movement_5","movement_11"]:
                roi_mask = roi_movement['movement_2']
            elif min_idx in ["movement_6","movement_12","movement_1"]:
                roi_mask = roi_movement['movement_1']
            else:
                roi_mask = roi_movement['movement_4']

            if check_bbox_overlap_with_roi(last_bbox, roi_mask):
                dst_frame = last_bbox[0]
                flag = False
            else :
                for i in range(len(bboxes) - 2, 0, -1):
                    bbox = bboxes[i]
                    if check_bbox_overlap_with_roi(bbox, roi_mask):
                        dst_frame = bbox[0]
                        flag = False
                        break
                    else:
                        continue
        if video_name in ["cam_14","cam_15"]:
            if min_idx in ["movement_2","movement_3"]:
                roi_mask = roi_movement['movement_3']
            elif min_idx in ["movement_1","movement_4"]:
                roi_mask = roi_movement['movement_1']
            else:
                roi_mask = roi_movement['movement_2']
            if check_bbox_overlap_with_roi(last_bbox, roi_mask):
                dst_frame = last_bbox[0]
                flag = False
            else:
                for i in range(len(bboxes) - 2, 0, -1):
                    bbox = bboxes[i]
                    if check_bbox_overlap_with_roi(bbox, roi_mask):
                        dst_frame = bbox[0]
                        flag = False
                        break
                    else:
                        continue
        if video_name in ["cam_16","cam_17"]:
            if min_idx in ["movement_1","movement_3"]:
                roi_mask = roi_movement['movement_1']
            else:
                roi_mask = roi_movement['movement_2']

            if check_bbox_overlap_with_roi(last_bbox, roi_mask):
                dst_frame = last_bbox[0]
                flag = False
            else :
                for i in range(len(bboxes) - 2, 0, -1):
                    bbox = bboxes[i]
                    if check_bbox_overlap_with_roi(bbox, roi_mask):
                        dst_frame = bbox[0]
                        flag = False
                        break
                    else:
                        continue
        if video_name in ["cam_20"]:
            if min_idx in ["movement_5","movement_2"]:
                roi_mask = roi_movement['movement_3']
            elif min_idx in ["movement_6","movement_1"]:
                roi_mask = roi_movement['movement_1']
            else: # if min_idx in ["movement_5","movement_2"]:
                roi_mask = roi_movement['movement_2']

            if check_bbox_overlap_with_roi(last_bbox, roi_mask):
                dst_frame = last_bbox[0]
                flag = False
            else:
                for i in range(len(bboxes) - 2, 0, -1):
                    bbox = bboxes[i]
                    if check_bbox_overlap_with_roi(bbox, roi_mask):
                        dst_frame = bbox[0]
                        flag = False
                        break
                    else:
                        continue
        if video_name in ["cam_21","cam_22"]:
            if min_idx in ["movement_4","movement_2"]:
                roi_mask = roi_movement['movement_2']
            elif  min_idx in ["movement_1","movement_5"]:
                roi_mask = roi_movement['movement_3']
            else:
                roi_mask = roi_movement['movement_1']

            if check_bbox_overlap_with_roi(last_bbox, roi_mask):
                dst_frame = last_bbox[0]
                flag = False
            else:
                for i in range(len(bboxes) - 2, 0, -1):
                    bbox = bboxes[i]
                    if check_bbox_overlap_with_roi(bbox, roi_mask):
                        dst_frame = bbox[0]
                        flag = False
                        break
                    else:
                        continue

        if flag :
            continue

            #         continue
        # if check_only_bbox_overlap_with_roi(last_bbox,mask) :
        #     dst_frame = last_bbox[0]
        # cx1, cy1 = track_center[0] # start point
        # ex1 , ey1 = track_center[-1] # end point
        # cx1, cy1, ex1, ey1 = cx1/w, cy1/h, ex1/w, ey1/h
        # print("min_idx config : ",min_idx)
        # image =cv2.imread("/home/thorpham/AI-Challenge/counting/zones-movement_paths/cam_10.png")
        # # cv2.circle(image,end_point,6,(255,0,255))
        # x1, y1, x2, y2  = list(map(int, [last_bbox[1], last_bbox[2], last_bbox[3], last_bbox[4]]))
        # cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,255), 2)
        # cv2.arrowedLine(image,(int(cx1*w), int(cy1*h)), (int(ex1*w), int(ey1*h)),(255,255,255),4)
        # cv2.putText(image, f'{min_idx[-1]}', (int(ex1*w), int(ey1*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        # cv2.circle(roi_movement[min_idx],(cx,cy),3,(0,255,0))
        # cv2.imshow("img",image)
        # cv2.imshow("img1",roi_movement[min_idx])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


  
        track_classes = [k[5] for k in bboxes]
        track_class_id = max(track_classes, key=track_classes.count)
        #
        # if track_type == 'car':
        #     cls_id = 1
        # else:
        #     cls_id = 2
        video_idx = tracks[trackid]['video_idx']
        # for test
        # bb = list(map(str, [last_bbox[1], last_bbox[2], last_bbox[3], last_bbox[4]]))
        # results.append([video_idx, str(dst_frame), direction_idx, str(track_class_id), *bb])
        # for test
        center = []
        _ = [center.extend(x) for x in track_center[1:]]
        center = list(map(str, center))

        bb = list(map(str, [last_bbox[1], last_bbox[2], last_bbox[3], last_bbox[4]]))
        results.append([video_idx, str(dst_frame), direction_idx, str(track_class_id), *bb, *center, str(trackid)])
        # # for submit
        # results.append([video_idx, str(dst_frame), direction_idx, str(int(track_class_id) + 1)])

    # save
    results.sort(key=lambda x: (int(x[1]), int(x[2])))
    writed_results = '\n'.join([' '.join(line) for line in results])
    write_result(os.path.join('counting_result', video_name + '.txt'), writed_results)
    print('vehicle counting done.')


if __name__ == '__main__':
    import glob
    # data_root = '../'
    data_root = ''
    # save_root = './vehicle_counting_results'
    save_root = ''
    # video_list = os.listdir(os.path.join(data_root, 'imageset'))
    # cam01, cam02, ...
    # video_list = [sys.argv[1]]
    video_list = sorted(glob.glob('track_results/*'))
    print(len(video_list))
    for video_path in video_list:

        video_name = video_path.split('/')[-1].split('.')[0]
        print('start to counting video %s ... ' % video_name)

        count_video(data_root, video_name, save_root)
