# import the necessary packages
from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
        help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())

sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
        args["server_ip"]))

rpiName = socket.gethostname()

vs = VideoStream(src=-1).start()
time.sleep(2.0)

while True:

        frame = vs.read()
        frame=imutils.resize(frame,width=640,height=200)
        answer = sender.send_image(rpiName, frame).decode("ascii")
        print(answer)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

vs.stream.strea.release()