import requests

class ActionController:

    def __init__(self):
        pass
    
    def take_action(self, objects, left_border, right_border):
        
        if len(objects) < 0:
            return
        
        for (x,y,w,h) in objects:
            x_pos = x + (w)/2
            y_pos = y + (h)/2
            
            if x_pos < left_border.curr_offset:
                self.turn_left()
            elif x_pos > right_border.curr_offset:
                self.turn_right()
            else:
                self.go_forward()


    def turn_left(self):
        print("TURN left")

    def turn_right(self):
        print("TURN right")

    def go_forward(self):
        print("GO FORWARD")

    def pick_up(self):
        print("PICK UP")