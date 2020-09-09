import cv2
import numpy as np

cam = "25"

counting_path = f'counting_result/cam_{cam}.txt'
# counting_path = f'D:\\project\\AI_challenge\\AI-Challenge\counting\\track_results\\cam_{cam}.txt'
video_path = f'/home/thorpham/AI-Challenge/videos/cam_{cam}.mp4'

import cv2
import numpy as np

# cam = '05'

# counting_path = f'counting_result\\cam_{cam}.txt'
# # counting_path = f'D:\\project\\AI_challenge\\AI-Challenge\counting\\track_results\\cam_{cam}.txt'
# video_path = f'D:\\project\\AI_challenge\\videos\\cam_{cam}.mp4'

color = {0: (255,0,0), 1:(0,255,0), 2:(255,255,0), 3:(0,0,255)}
class_dict = {0: 'moto', 1: 'car', 2: 'bus', 3: 'truck'}
with open(counting_path, 'r') as f:
    data = [line.strip('\n').split() for line in f]

cam = cv2.VideoCapture(video_path)
frame_id = 0
data_id = 0
ret, frame = cam.read()
while ret:
    # print(int(data[data_id][1]))
    while int(data[data_id][1]) == frame_id:
        item = data.pop(0)



        # test counting
        cls = int(float(item[3]))
        dir = int(float(item[2]))
        x1, y1, x2, y2 = item[4:8]
        center = item[8:-1]
        center = np.array(list(map(float, center))).reshape(-1, 2)
        center = center.astype(np.uint32)
        idtrack = item[-1]
        for i in range(len(center)-1):
            cv2.line(frame, tuple(center[i]), tuple(center[i+1]), (0,0,255), 2)
            cv2.circle(frame, tuple(center[i]), 3, (255, 255, 255))
        print(idtrack)

        # test tracking
        # id_object = item[2]
        # cls = int(float(item[7]))
        # x1, y1, x2, y2 = item[3:7]

        x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
        # px1, py1, px2, py2 = int(float(px1)), int(float(py1)), int(float(px2)), int(float(py2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color[cls], 2)
        cv2.putText(frame, f'dir:{dir} - cls: {class_dict[cls]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.putText(frame, f'id:{id_object} - cls: {class_dict[cls]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # print("mv:", p)
        # print("class:", c)
        # data_id += 1
        cv2.imshow("img", frame)
        cv2.waitKey(500)

        if len(data) == 0:
            break

    cv2.imshow("img", frame)
    cv2.waitKey(200)
    ret, frame = cam.read()
    frame_id += 1