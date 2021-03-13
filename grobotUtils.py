
import cv2
import numpy as np
import math

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    # Colored labels dictionary
    color_dict = {
        'Tin can' : [0, 255, 255], 'Bottle': [238, 123, 158]
    }
    
    for label, confidence, bbox in detections:
        x, y, w, h = (bbox[0],
            bbox[1],
            bbox[2],
            bbox[3])
        name_tag = label
        for name_key, color_val in color_dict.items():
            if name_key == name_tag:
                color = color_val 
                xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                cv2.rectangle(img, pt1, pt2, color, 1)
                cv2.putText(img,
                            name_tag +
                            " [" + confidence + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)
    return img

def center(points):
    x = np.float32(
        (points[0][0] +
         points[1][0] +
         points[2][0] +
         points[3][0]) /
        4.0)
    y = np.float32(
        (points[0][1] +
         points[1][1] +
         points[2][1] +
         points[3][1]) /
        4.0)
    return np.array([np.float32(x), np.float32(y)], np.float32)

def find_closest(dets,b):
    centers = [[int(det[2][0]+det[2][2]/2), int(det[2][1] + det[2][3]/2)] for det in dets]
    min_distance = None
    curr_det_id = 0
    det_id = 0 
    for center in centers:
        if min_distance == None:
            min_distance = math.sqrt( (center[0] - ( b[0][0] + (b[1][0]-b[0][0]) /2 ))**2 + (center[1] - (b[0][1] + (b[1][1]-b[1][1]) /2))**2 )
            det_id=curr_det_id
            curr_det_id+=1
            continue
        curr_distance = math.sqrt( (center[0] - ( b[0][0] + (b[1][0]-b[0][0]) /2 ))**2 + (center[1] - (b[0][1] + (b[1][1]-b[1][1]) /2))**2 )
        if curr_distance < min_distance:
            min_distance = curr_distance
            det_id=curr_det_id
        curr_det_id+=1
    #print(dets[det_id])
    return dets[det_id]