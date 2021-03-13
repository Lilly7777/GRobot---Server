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



def YOLO():
   

    imageHub = imagezmq.ImageHub()
    frameDict = {}

    lastActive = {}
    lastActiveCheck = datetime.now()

    ESTIMATED_NUM_PIS = 4
    ACTIVE_CHECK_PERIOD = 10
    ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

    mW = 2
    mH = 2
    frame_width = 640                                  # Returns the width and height of capture video
    frame_height = 480
    max_sector_offset = 30

    cv2.namedWindow("Result")

    #cap = cv2.VideoCapture(1)                                      # Uncomment to use Webcam
    #cap = cv2.VideoCapture("japan.mp4")                             # Local Stored video detection - Set input video
    
    # Set out for video writer

    print("Starting the YOLO loop...")

    target = None
    missedFrames = 0
    # Create an image we reuse for each detect
    while True:                                                      # Load the input frame and write output frame.
        prev_time = time.time()
        (rpiName, frame_read) = imageHub.recv_image()
        #imageHub.send_reply(b'DUNNO')
        
        # if a device is not in the last active dictionary then it means
        # that its a newly connected device
        if rpiName not in lastActive.keys():
            print("[INFO] receiving data from {}...".format(rpiName))
        # record the last active time for the device from which we just
        # received a frame
        lastActive[rpiName] = datetime.now()


        
    
	
        
        
        # if current time *minus* last time when the active device check
        # was made is greater than the threshold set then do a check
        if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
            # loop over all previously active devices
            for (rpiName, ts) in list(lastActive.items()):
                # remove the RPi from the last active and frame
                # dictionaries if the device hasn't been active recently
                if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                    print("[INFO] lost connection to {}".format(rpiName))
                    lastActive.pop(rpiName)
                    frameDict.pop(rpiName)
            # set the last active check time as current time
            lastActiveCheck = datetime.now()
    

        print(target)
        frameDict[rpiName] = frame_read
        # build a montage using images in the frame dictionary
        (h, w) = frame_read.shape[:2]
        montages = build_montages(frameDict.values(), (w, h), (mW, mH))
        
        # display the montage(s) on the screen
        cv2.putText(frame_read, rpiName, (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (i, montage) in enumerate(montages):
            cv2.imshow("Result",
                frame_read)

        print(1/(time.time()-prev_time))

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
                                                                   # For releasing cap and out. 

if __name__ == "__main__":  
    YOLO()                                                           # Calls the main function YOLO()