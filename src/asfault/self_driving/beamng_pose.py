class BeamNGPose:
    def __init__(self, pos=None, rot=None):
        self.pos = pos if pos else (0, 0, 0)
        self.rot = rot if rot else (0, 0, 0)
