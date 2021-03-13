from ctypes import *                                               # Import libraries
import math
import random
import os
import cv2
import numpy as np
import time
from imutils import build_montages
from datetime import datetime
import imagezmq
import argparse
import imutils
from config import *
from AreaBorder import AreaBorder
from ActionController import ActionController
from grobotUtils import *
from Darknet.darknet.build.darknet.x64 import darknet
import scipy.misc
from PIL import Image as im 

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

boxes = []
tracking = False

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

netMain = None
metaMain = None
altNames = None

def YOLO():
    pickup_timer = 0
    global metaMain, netMain, altNames, boxes, tracking, prediction
    configPath = "yolov4-obj.cfg"                                 # Path to cfg
    weightPath = "yolov4-grobot.weights"                                 # Path to weights
    metaPath = "obj.data"                                    # Path to meta data

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

    
    imageHub = imagezmq.ImageHub()
    frameDict = {}
    lastActive = {}
    pickedUp = {}
    lastActiveCheck = datetime.now()

    ESTIMATED_NUM_PIS = 4
    ACTIVE_CHECK_PERIOD = 10
    ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

    mW = 2
    mH = 2
    
    

    cv2.namedWindow("Preview")
    print("Starting the YOLO loop...")
    frame_width = 640
    frame_height = 480
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(640, 480, 3) # Create image according darknet for compatibility of network
    while True:                                                      # Load the input frame and write output frame.
        prev_time = time.time()
        pickup_timer+=1
        (rpiName, frame_read) = imageHub.recv_image()

        


        left_border = AreaBorder('vertical', frame_width/2 - max_sector_offset, 1)
        right_border = AreaBorder('vertical', frame_width/2 + max_sector_offset, 1)
        # if a device is not in the last active dictionary then it means
        # that its a newly connected device
        if rpiName not in lastActive.keys():
            print("receiving data from {}...".format(rpiName))
        
        # record the last active time for the device from which we just
        lastActive[rpiName] = datetime.now()

        if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
            for (rpiName, ts) in list(lastActive.items()):
                # remove the RPi from the last active and frame
                # dictionaries if the device hasn't been active recently
                if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                    print("lost connection to {}".format(rpiName))
                    lastActive.pop(rpiName)
                    frameDict.pop(rpiName)
            lastActiveCheck = datetime.now()

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width, frame_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                # Copy that frame bytes to darknet_image
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)    # Detection occurs at this line and return detections, for customize we can change the threshold.                                                                                   
        custom_detections = []
        for det in detections:
            if robots[rpiName]["picked_up"] < robots[rpiName]["capacity"]:
                if det[0] in robots[rpiName]["targets"]:
                    custom_detections.append(det)
            else:
                if det[0] == robots[rpiName]["dispose_at"]:
                    custom_detections.append(det)
        custom_detections = tuple(custom_detections)

        if len(custom_detections):
            for det in custom_detections:
                if det[0] in robots[rpiName]["targets"]:
                    print(det[0])
                    if len(boxes)<1: #if not found or closer
                        boxes.append([int(det[2][0]), int(det[2][1])])
                        boxes.append([int(det[2][0]+det[2][1]), int(det[2][1]+ det[2][3])])
                    else:
                        
                        last_track = find_closest(custom_detections, boxes)
                        xmin, ymin, xmax, ymax = convertBack(last_track[2][0],last_track[2][1],last_track[2][2],last_track[2][3])
                        boxes = []
                        boxes.append([int(xmin), int(ymin)])
                        boxes.append([int(xmax), int(ymax)])

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

                # construct a histogram of hue and saturation values and normalize it
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
        
        if tracking:

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

            # draw observation on image
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

        
        if len(boxes)>0:     
            msg = ""   
            if prediction[0]<left_border.max_offset:
                msg = "LEFT "+str(abs(int(frame_width/2 - prediction[0])))
            elif (prediction[0]>right_border.max_offset):
                msg = "RIGHT " + str(abs(int(frame_width/2 - prediction[0])))
            else:
                
                if boxes[0][1] + (boxes[1][1] - boxes[0][1])/2 >(frame_height//4)*3:
                    msg = "PICKUP 0"
                    if pickup_timer>50:
                        robots[rpiName]["picked_up"]+=1
                    pickup_timer=0
                else:
                    msg = "FORWARD " + str(abs(int(frame_height - prediction[1])))
        else:
            msg = "IDLE 0"

        #print(msg)
        imageHub.send_reply(msg.encode('ascii'))
        
        
        image = cvDrawBoxes(detections, frame_resized)               # Call the function cvDrawBoxes() for colored bounding box per class
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        cv2.line(image, (left_border.max_offset, 0), (left_border.max_offset, frame_height), (0, 255, 0), thickness=2)
        cv2.line(image, (right_border.max_offset, 0), (right_border.max_offset, frame_height), (255, 0, 0), thickness=2)
        frameDict[rpiName] = image
        # build a montage using images in the frame dictionary
        (h, w) = image.shape[:2]
        montages = build_montages(frameDict.values(), (w, h), (mW, mH))
        
        # display the montage(s) on the screen
        cv2.putText(image, rpiName, (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #cv2.imshow('Preview', image)
        for (i, montage) in enumerate(montages):
            cv2.imshow("Preview",
                image)

        print(1/(time.time()-prev_time))

        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'): break

if __name__ == "__main__":  
    YOLO()                                                           # Calls the main function YOLO()