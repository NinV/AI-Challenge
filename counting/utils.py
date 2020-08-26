import cv2
import numpy as np
import json
import os
import glob

def create_mask(config="test.json"):
    ''' Create mask ROI'''
    path = config
    with open(path,"r") as file :
        data = json.load(file)
    # get param image
    path_image = data["imagePath"]
    width = data["imageWidth"]
    height = data["imageHeight"]
    # get ROI
    zone = data["shapes"][0]["points"]
   
    ROI = [[int(x),int(y)] for x,y in zone]
    #  create mask  image
    mask_mat = np.zeros((height,width,3),np.uint8)

    cv2.fillPoly(mask_mat, [np.array(ROI)], (255, 255, 255))
    cv2.imwrite(os.path.join("mask",os.path.basename(path_image)),mask_mat)

def is_outside_roi(bbox, mask):
    is_outside = True

    x_tl = bbox[0] if bbox[0] > 0 else 0
    y_tl = bbox[1] if bbox[1] > 0 else 0
    x_br = bbox[2] if bbox[2] < mask.shape[1] else mask.shape[1] - 1
    y_br = bbox[3] if bbox[3] < mask.shape[0] else mask.shape[0] - 1
    #print(x_tl, y_tl, x_br, y_br)
    vertexs = [[x_tl, y_tl], [x_tl, y_br], [x_br, y_tl], [x_br, y_br]]
    for v in vertexs:
        (g, b, r) = mask[v[1], v[0]]
        if (g, b, r) > (128, 128, 128):
            is_outside = False
            return is_outside
    return is_outside









    
if __name__ == "__main__":
    file = glob.glob("zones-movement_paths/*.json")
    for i in file:
        test = create_mask(i)