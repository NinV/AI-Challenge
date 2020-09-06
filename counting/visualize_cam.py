import random
import cv2
import numpy as np

# config param
cam = '12'
counting_path = f'/home/thorpham/AI-Challenge/counting/track_results/cam_{cam}.txt'
video_path = f'/home/thorpham/AI-Challenge/videos/cam_{cam}.mp4'
mask_path = f'/home/thorpham/AI-Challenge/counting/mask/cam_{cam}.png'
# ------------------------------------------------------------------------


def check_center_inside_with_roi(center, mask):
    if np.sum(mask[center[1], center[0]]) > 100:
        return True
    else:
        return False


mask = cv2.imread(mask_path)

color = {0: (255,0,0), 1:(0,255,0), 2:(255,255,0), 3:(0,0,255)}
class_dict = {0: 'moto', 1: 'car', 2: 'bus', 3: 'truck'}
with open(counting_path, 'r') as f:
    data = [line.strip('\n').split() for line in f]

cam = cv2.VideoCapture(video_path)
frame_id = 0
data_id = 0
ret, frame = cam.read()
draw_img = np.zeros_like(frame)

result = {}
object_color = {}
while ret:
    # print(int(data[data_id][1]))
    while int(data[data_id][1]) == frame_id:
        item = data.pop(0)

        id_object = item[2]
        cls = int(float(item[7]))
        x1 = max(int(float(item[3])), 0)
        y1 = max(int(float(item[4])), 0)
        x2 = min(int(float(item[5])), 1280)
        y2 = min(int(float(item[6])), 720)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if check_center_inside_with_roi((cx, cy), mask):
            if id_object not in result:
                result[id_object] = [(cx, cy)]
                object_color[id_object] = (random.randint(10, 255), random.randint(10, 255), random.randint(10, 255))
            else:
                result[id_object].append((cx, cy))

                cv2.line(draw_img, result[id_object][-2], result[id_object][-1], object_color[id_object], 2)
                cv2.circle(draw_img, result[id_object][-1], 4, (0, 0, 255))
                # cv2.putText(draw_img, str(len(result[id_object])), result[id_object][-1], cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(draw_img, str(id_object), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 255, 0), 1)
        if len(data) == 0:
            break
    frame[draw_img>0] = draw_img[draw_img>0]
    cv2.imshow("press n to renew", frame)
    cv2.waitKey(1)
    if frame_id % 10 == 0:
        k = cv2.waitKey(0)
        if k == ord('n'):
            draw_img = np.zeros_like(frame)
        elif k == 27:
            break
    ret, frame = cam.read()
    frame_id += 1

cam.release()
cv2.destroyAllWindows()