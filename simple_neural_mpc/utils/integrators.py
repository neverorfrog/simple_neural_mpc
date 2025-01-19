from abc import ABC

import casadi as ca


class Integrator(ABC):
    def step(self, state, action, h):
        return self.discrete_ode(state, action, h)

    @property
    def discrete_ode(self):
        return self._discrete_ode  # redefined by subclass


class Euler(Integrator):

    def __init__(self, state: ca.MX, action: ca.MX, f: ca.MX, h: ca.MX):
        f = ca.Function("f", [state, action], [f])
        k = f(state, action)
        x_next = state + h * k
        self._discrete_ode = ca.Function(
            "f_discrete", [state, action, h], [x_next]
        ).expand()


class RK4(Integrator):

    def __init__(self, state: ca.MX, action: ca.MX, f: ca.MX, h: ca.MX):
        f = ca.Function("f", [state, action], [f])
        k_1 = f(state, action)
        k_2 = f(state + 0.5 * h * k_1, action)
        k_3 = f(state + 0.5 * h * k_2, action)
        k_4 = f(state + h * k_3, action)
        state_next = state + h * (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        self._discrete_ode = ca.Function(
            "f_discrete", [state, action, h], [state_next]
        ).expand()


class RK2(Integrator):

    def __init__(self, state: ca.MX, action: ca.MX, f: ca.MX, h: ca.MX):
        f = ca.Function("f", [state, action], [f])
        k_1 = f(state, action)
        k_2 = f(state + 0.5 * h * k_1, action)
        state_next = state + h * k_2
        self._discrete_ode = ca.Function(
            "f_discrete", [state, action, h], [state_next]
        ).expand()