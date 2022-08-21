from casadi import *

class DiffDriveModel:
    def __init__(self):
        self.mc = 2.0 #cart mass
        self.mp = 0.2 #pole mass
        self.l = 0.5 #pole length
        self.ga = 9.81 #gravity constant

    def dynamics(self, x, u):
        x_pos = x[0] # x position[m]
        y_pos = x[1] # y position[m]
        th = x[2] # angle[rad]
        v = u[0] # velocity input[m/s]
        w = u[1] # angular velocity input[rad/s]

        dx = v * cos(th)
        dy = v * sin(th)
        dth = w

        return dx, dy, dth