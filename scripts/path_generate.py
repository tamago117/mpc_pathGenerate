import math
import time
import matplotlib.pyplot as plt
import numpy as np
from casadi import *

from MPC import MPC
from DiffDriveModel import DiffDriveModel

sim_time = 10.0
sampling_time = 0.02 # 100hz
sim_steps = math.floor(sim_time / sampling_time)

ob = np.array([[-1, -1],
                [3, 3]])

def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, robot_radius):  # pragma: no cover
    circle = plt.Circle((x, y), robot_radius, color="b")
    plt.gcf().gca().add_artist(circle)
    out_x, out_y = (np.array([x, y]) +
                    np.array([np.cos(yaw), np.sin(yaw)]) * robot_radius)
    plt.plot([x, out_x], [y, out_y], "-k")

def main():
    x = np.array([0.0, 0.0, 0.0])
    x_ref = np.array([3.0, 3.0, 0.0])   # target
    xs = []
    us = []
    diffDrive = DiffDriveModel()
    mpc = MPC()
    #mpc.init()

    for step in range(sim_steps):
        if step%(1/sampling_time) == 0:
            print('t=', step*sampling_time)

        current_time = time.time()
        u = mpc.solve(x, x_ref)
        path = mpc.get_path()
        print(str(math.floor(1/(time.time() - current_time))) + "hz")
        xs.append(x)
        us.append(u)
        xs1 = [x[0] for x in xs]
        xs2 = [x[1] for x in xs]
        x1 = x + sampling_time * np.array(diffDrive.dynamics(x, u))
        x = x1

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        #plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
        plt.plot(xs1, xs2)
        plt.plot(x[0], x[1], "xr")
        plt.plot(x_ref[0], x_ref[1], "xb")
        plot_arrow(x_ref[0], x_ref[1], x_ref[2])
        #plt.plot(ob[:, 0], ob[:, 1], "ok")
        plot_robot(x[0], x[1], x[2], 0.3)
        plot_arrow(x[0], x[1], x[2])
        plt.plot(path[0], path[1])
        plt.title(f"MPC path generate\n v: {u[0]:.2f} , w: {u[1]:.2f}")
        plt.axis("equal")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.grid(True)
        plt.pause(0.0001)

        if math.sqrt((x_ref[0]-x[0])**2+(x_ref[1]-x[1])**2)<0.05:
            break

if __name__ == '__main__':
    main()