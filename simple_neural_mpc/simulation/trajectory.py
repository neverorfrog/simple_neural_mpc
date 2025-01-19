from abc import ABC, abstractmethod

import numpy as np


class Trajectory(ABC):

    @abstractmethod
    def update(self, t: float):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                p_d,     position, m
                pd_d,    velocity, m/s
                pdd_d,   acceleration, m/s**2
        """
        pass


class Zero(Trajectory):
    def __init__(self):
        """
        Constructor for the zero trajectory.
        """
        pass

    def update(self, t):
        return {
            "p": np.full((2, len(t)), 0),
            "pd": np.full((2, len(t)), 0),
            "pdd": np.full((2, len(t)), 0),
            "psi": np.full((len(t)), 0),
            "psid": np.full((len(t)), 0),
        }


class Dritto(Trajectory):
    def __init__(self):
        """
        Constructor for the dritto trajectory.
        """
        pass

    def update(self, t):

        p = np.full((2, len(t)), t * 0.1)
        pd = np.full((2, len(t)), t * 0.01)
        pdd = np.full((2, len(t)), t * 0.001)
        p[1, :] *= 0
        pd[1, :] *= 0
        pdd[1, :] *= 0

        return {
            "p": p,
            "pd": pd,
            "pdd": pdd,
            "psi": np.full((len(t)), 0),
            "psid": np.full((len(t)), 0),
        }


class Circle(Trajectory):
    def __init__(self, T=6, center=np.array([0, 0]), radius=1, freq=0.2):
        """
        This is the constructor for the circle trajectory

        Inputs:
            center, the center of the circle (m)
            radius, the radius of the circle (m)
            freq, the frequency with which a circle is completed (Hz)
        """
        self.center = center
        self.cx, self.cy = center[0], center[1]
        self.radius = radius
        self.freq = freq
        self.omega = 2 * np.pi * self.freq
        self.T = T

    def update(self, t):
        p = np.array(
            [
                self.cx + self.radius * np.cos(self.omega * t),
                self.cy + self.radius * np.sin(self.omega * t),
            ]
        )
        pd = np.array(
            [
                -self.radius * self.omega * np.sin(self.omega * t),
                self.radius * self.omega * np.cos(self.omega * t),
            ]
        )
        pdd = np.array(
            [
                -self.radius * ((self.omega) ** 2) * np.cos(self.omega * t),
                -self.radius * ((self.omega) ** 2) * np.sin(self.omega * t),
            ]
        )

        psi = np.arctan2(pd[1], pd[0])
        psid = (pd[0] * pdd[1] - pd[1] * pdd[0]) / (pd[0] ** 2 + pd[1] ** 2)

        return {"p": p, "pd": pd, "pdd": pdd, "psi": psi, "psid": psid}


class Ellipse(Trajectory):
    def __init__(self, T=6, center=np.array([0, 0]), a=2, b=1, freq=0.2):
        """
        Constructor for the elliptical trajectory.

        Inputs:
            center, the center of the ellipse (m)
            a, semi-major axis length (m)
            b, semi-minor axis length (m)
            freq, the frequency with which an ellipse is completed (Hz)
        """
        self.center = center
        self.cx, self.cy = center[0], center[1]
        self.a = a
        self.b = b
        self.freq = freq
        self.omega = 2 * np.pi * self.freq
        self.T = T

    def update(self, t):
        p = np.array(
            [
                self.cx + self.a * np.cos(self.omega * t),
                self.cy + self.b * np.sin(self.omega * t),
            ]
        )
        pd = np.array(
            [
                -self.a * self.omega * np.sin(self.omega * t),
                self.b * self.omega * np.cos(self.omega * t),
            ]
        )
        pdd = np.array(
            [
                -self.a * (self.omega) ** 2 * np.cos(self.omega * t),
                -self.b * (self.omega) ** 2 * np.sin(self.omega * t),
            ]
        )

        psi = np.arctan2(pd[1], pd[0])
        psid = (pd[0] * pdd[1] - pd[1] * pdd[0]) / (pd[0] ** 2 + pd[1] ** 2)

        return {"p": p, "pd": pd, "pdd": pdd, "psi": psi, "psid": psid}


class Spiral(Trajectory):
    def __init__(self, T=6, center=np.array([0, 0]), a=0.5, b=0.2, freq=0.2):
        """
        Constructor for the Archimedean spiral trajectory.

        Inputs:
            center, the center of the spiral (m)
            a, the initial distance from the center (m)
            b, the distance increment (m/radian)
            freq, the frequency of rotation (Hz)
        """
        self.center = center
        self.cx, self.cy = center[0], center[1]
        self.a = a
        self.b = b
        self.freq = freq
        self.omega = 2 * np.pi * self.freq
        self.T = T

    def update(self, t):
        r = self.a + self.b * t  # Radial distance increases with time
        p = np.array(
            [
                self.cx + r * np.cos(self.omega * t),
                self.cy + r * np.sin(self.omega * t),
            ]
        )
        pd = np.array(
            [
                -r * self.omega * np.sin(self.omega * t)
                + self.b * np.cos(self.omega * t),
                r * self.omega * np.cos(self.omega * t) + self.b * np.sin(self.omega * t),
            ]
        )
        pdd = np.array(
            [
                -r * (self.omega) ** 2 * np.cos(self.omega * t)
                - 2 * self.b * self.omega * np.sin(self.omega * t),
                -r * (self.omega) ** 2 * np.sin(self.omega * t)
                + 2 * self.b * self.omega * np.cos(self.omega * t),
            ]
        )

        psi = np.arctan2(pd[1], pd[0])
        psid = (pd[0] * pdd[1] - pd[1] * pdd[0]) / (pd[0] ** 2 + pd[1] ** 2)

        return {"p": p, "pd": pd, "pdd": pdd, "psi": psi, "psid": psid}


class Eight(Trajectory):
    def __init__(self, T=6, center=np.array([0, 0]), a=1, freq=0.2):
        """
        Constructor for the figure-eight (lemniscate) trajectory.

        Inputs:
            center, the center of the figure-eight (m)
            a, the size of the loops (m)
            freq, the frequency of the motion (Hz)
        """
        self.center = center
        self.cx, self.cy = center[0], center[1]
        self.a = a  # This controls the size of the '8'
        self.freq = freq
        self.omega = 2 * np.pi * self.freq
        self.T = T

    def update(self, t):
        # Lemniscate of Bernoulli equation:
        # x(t) = a * cos(omega * t)
        # y(t) = a * sin(omega * t) * cos(omega * t)

        x = self.cx + self.a * np.cos(self.omega * t)
        y = self.cy + self.a * np.sin(self.omega * t) * np.cos(self.omega * t)

        p = np.array([x, y])

        # Velocities (first derivatives):
        pd = np.array(
            [
                -self.a * self.omega * np.sin(self.omega * t),  # Velocity in x-direction
                self.a
                * self.omega
                * (
                    np.cos(self.omega * t) ** 2 - np.sin(self.omega * t) ** 2
                ),  # Velocity in y-direction
            ]
        )

        # Accelerations (second derivatives):
        pdd = np.array(
            [
                -self.a
                * (self.omega) ** 2
                * np.cos(self.omega * t),  # Acceleration in x-direction
                -2
                * self.a
                * (self.omega) ** 2
                * np.sin(self.omega * t)
                * np.cos(self.omega * t),  # Acceleration in y-direction
            ]
        )

        # Orientation and angular velocity:
        psi = np.arctan2(pd[1], pd[0])  # Heading angle
        psid = (pd[0] * pdd[1] - pd[1] * pdd[0]) / (
            pd[0] ** 2 + pd[1] ** 2
        )  # Angular velocity

        return {"p": p, "pd": pd, "pdd": pdd, "psi": psi, "psid": psid}
