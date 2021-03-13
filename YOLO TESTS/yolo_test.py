from ctypes import *                                               
import math
import random
import os
import cv2
import numpy as np
import time
from sort import *
import sys
from Darknet.darknet.build.darknet.x64 import darknet
import argparse
from AreaBorder import AreaBorder
from ActionController import ActionController
#from kalmantest import center


kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

measurement = np.array((2, 1), np.float32)
prediction = np.zeros((2, 1), np.float32)

s_lower = 60
s_upper = 255
v_lower = 32
v_upper = 255

frame_width = 640                                  # Returns the width and height of capture video
frame_height = 480
max_sector_offset = 30
left_border = AreaBorder('vertical', frame_width/2 - max_sector_offset, 1)
right_border = AreaBorder('vertical', frame_width/2 + max_sector_offset, 1)

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

boxes = []
tracking = False

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

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
    print(det_id)
    return dets[det_id]


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


netMain = None
metaMain = None
altNames = None

def YOLO():
   
    global metaMain, netMain, altNames, boxes
    configPath = "yolov4-obj.cfg"                                 # Path to cfg
    weightPath = "yolov4-obj_last.weights"                                 # Path to weights
    metaPath = "obj.data"                                    # Path to meta data
    mot_tracker = Sort() 
    network, class_names, class_colors = darknet.load_network(configPath,  metaPath, weightPath, batch_size=1)

    if not os.path.exists(configPath):                              # Checks whether file exists otherwise return ValueError
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:                                             # Checks the metaMain, NetMain and altNames. Loads it in script
        netMain = darknet.load_net_custom(configPath.encode( 
            "ascii"), weightPath.encode("ascii"), 0, 1)             # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture(0)                                    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height, 3) # Create image according darknet for compatibility of network
    while True:                                                      # Load the input frame and write output frame.
        prev_time = time.time()
        ret, frame_read = cap.read()                                 # Capture frame and return true if frame present
        # For Assertion Failed Error in OpenCV
        if not ret:                                                  # Check if frame present otherwise he break the while loop
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width, frame_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                # Copy that frame bytes to darknet_image

        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)    # Detection occurs at this line and return detections, for customize we can change the threshold.                                                                                   
        #print(detections)
        #print(boxes)
        if len(detections):
            for det in detections:
                if len(boxes)<1: #if not found or closer
                    boxes.append([int(det[2][0]), int(det[2][1])])
                    boxes.append([int(det[2][0]+det[2][1]), int(det[2][1]+ det[2][3])])
                else:
                    
                    last_track = find_closest(detections, boxes)
                    boxes = []
                    boxes.append([int(last_track[2][0]), int(last_track[2][1])])
                    boxes.append([int(last_track[2][0]+last_track[2][1]), int(last_track[2][1]+ last_track[2][3])])
                    print(boxes)

        if (len(boxes) > 1) and (boxes[0][1] < boxes[1][1]) and (
                boxes[0][0] < boxes[1][0]):
            
            crop = frame_read[boxes[0][1]:boxes[1][1],
                         boxes[0][0]:boxes[1][0]].copy()

            h, w, c = crop.shape   # size of template
            if (h > 0) and (w > 0):
                tracking = True

                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

                # select all Hue (0-> 180) and Sat. values but eliminate values
                # with very low saturation or value (due to lack of useful
                # colour information)

                mask = cv2.inRange(
                    hsv_crop, np.array(
                        (0., float(s_lower), float(v_lower))), np.array(
                        (180., float(s_upper), float(v_upper))))

                # construct a histogram of hue and saturation values and
                # normalize it

                crop_hist = cv2.calcHist(
                    [hsv_crop], [
                        0, 1], mask, [
                        180, 255], [
                        0, 180, 0, 255])
                cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)

                # set intial position of object

                track_window = (
                    boxes[0][0],
                    boxes[0][1],
                    boxes[1][0] -
                    boxes[0][0],
                    boxes[1][1] -
                    boxes[0][1])

                

            # reset list of boxes
            #boxes = []
        if (tracking):

            # convert incoming image to HSV
            img_hsv = cv2.cvtColor(frame_read, cv2.COLOR_BGR2HSV)

            # back projection of histogram based on Hue and Saturation only

            img_bproject = cv2.calcBackProject(
                [img_hsv], [
                    0, 1], crop_hist, [
                    0, 180, 0, 255], 1)

            # apply camshift to predict new location (observation)
            ret, track_window = cv2.CamShift(
                img_bproject, track_window, term_crit)

            # draw observation on image - in BLUE
            x, y, w, h = track_window
            frame_resized = cv2.rectangle(
                frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # extract centre of this observation as points

            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            # (cx, cy), radius = cv2.minEnclosingCircle(pts)

            # use to correct kalman filter
            kalman.correct(center(pts))

            # get new kalman filter prediction
            prediction = kalman.predict()
            # draw predicton on image - in GREEN
            frame_resized = cv2.rectangle(frame_resized,
                                  (int(prediction[0] - (0.5 * w)),
                                   int(prediction[1] - (0.5 * h))),
                                  (int(prediction[0] + (0.5 * w)),
                                   int(prediction[1] + (0.5 * h))),
                                  (0,255,0),2)

        
        if(prediction[0]<left_border.max_offset):
            print("LEFT")
        elif (prediction[0]>right_border.max_offset):
            print("RIGHT")
        else:
            print("FORWARD")
        
        
        image = cvDrawBoxes(detections, frame_resized)               # Call the function cvDrawBoxes() for colored bounding box per class
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.line(image, (left_border.max_offset, 0), (left_border.max_offset, frame_height), (0, 255, 0), thickness=2)
        cv2.line(image, (right_border.max_offset, 0), (right_border.max_offset, frame_height), (255, 0, 0), thickness=2)
        #cv2.line(image, (x_pos, y), (x_pos, y + h), (0, 0, 255), thickness=2)
        #print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)                                    # Display Image window
        cv2.waitKey(3)
        #out.write(image)                                             # Write that frame into output video
    cap.release()                                                    # For releasing cap and out. 
    #out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":  
    YOLO()                                                           # Calls the main function YOLO()