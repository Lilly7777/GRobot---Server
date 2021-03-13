class AreaBorder:
    orientation = ""
    max_offset = 0
    curr_offset = 0
    step = 0
    def __init__(self, orientation, max_offset, step):
        self.orientation = orientation
        self.max_offset = int(max_offset)
        self.curr_offset = max_offset
        self.step = step