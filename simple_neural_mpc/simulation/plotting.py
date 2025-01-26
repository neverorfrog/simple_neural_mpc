import numpy as np
from casadi import cos, sin
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def wrap(angle):
    """Wrap between -pi and pi"""
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle


def plot_wheeled_robot(axis: Axes, x: float, y: float, psi: float, num_wheels: int = 2):
    # Plot circular shape
    r = 0.2
    circle = plt.Circle(xy=(x, y), radius=r, facecolor="orange", alpha=0.5, lw=2)
    axis.add_patch(circle)
    
    # Draw forward direction tick
    offset = 0.1
    tick_length = 0.1
    x_start = x + offset * cos(psi)
    x_tick = x_start + tick_length * cos(psi)
    y_start = y + offset * sin(psi)
    y_tick = y_start + tick_length * sin(psi)
    axis.plot([x_start, x_tick], [y_start, y_tick], color="red", lw=2)    

    # Draw two wheels as rectangles
    wheel_angle = wrap(psi - np.pi / 2)
    width = 0.05
    height = 0.15
    x_wheel_right = x + cos(wheel_angle) * r - cos(psi) * r / 3 - cos(wheel_angle) * width
    y_wheel_right = y + sin(wheel_angle) * r - sin(psi) * r / 3 - sin(wheel_angle) * width
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
