################################################################
cameraNo = 0                       # CAMERA NUMBE
frameWidth= 640                     # DISPLAY WIDTH
frameHeight = 480                  # DISPLAY HEIGHT

max_sector_offset = 60
waste_rover_ip = "http://192.168.1.8:8081/"


def empty(a):
    pass


robots = {
    "raspberrypi-wasteRover": {
        "targets": ["Tin can"],
        "capacity": 5
    },
    
}

#################################################################