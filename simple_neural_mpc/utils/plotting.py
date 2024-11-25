import numpy as np
from casadi import cos, sin
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from simple_neural_mpc.utils import wrap


def plot_wheeled_robot(axis: Axes, x: float, y: float, psi: float, num_wheels: int = 2):
    # Plot circular shape
    r = 0.2
    circle = plt.Circle(xy=(x, y), radius=r, facecolor="orange", alpha=0.5, lw=2)
    axis.add_patch(circle)

    # Draw two wheels as rectangles
    wheel_angle = wrap(psi - np.pi / 2)
    width = 0.05
    height = 0.15
    x_wheel_right = (
        x + cos(wheel_angle) * r - cos(psi) * r / 3 - cos(wheel_angle) * width
    )
    y_wheel_right = (
        y + sin(wheel_angle) * r - sin(psi) * r / 3 - sin(wheel_angle) * width
    )
    wheel_right = plt.Rectangle(
        (x_wheel_right, y_wheel_right),
        width=width,
        height=height,
        angle=np.rad2deg(wheel_angle),
        facecolor="black",
    )
    axis.add_patch(wheel_right)
    x_wheel_left = x - cos(psi) * r / 3 - cos(wheel_angle) * r
    y_wheel_left = y - sin(psi) * r / 3 - sin(wheel_angle) * r
    wheel_left = plt.Rectangle(
        (x_wheel_left, y_wheel_left),
        width=width,
        height=height,
        angle=np.rad2deg(wheel_angle),
        facecolor="black",
    )
    axis.add_patch(wheel_left)