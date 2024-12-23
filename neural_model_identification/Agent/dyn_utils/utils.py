import torch 
import numpy as np



def euler_integration(x, x_dot, delta_t):
    return x + x_dot * delta_t

def normalize_state(state, x_min, x_max):
    """
    Normalize state
    """
    state = (((state - x_min) / (x_max - x_min)) - 0.5) * 2

    return state


def denormalize_state(state, x_min, x_max):
    """
    Denormalize state
    """
    state = ((state / 2) + 0.5) * (x_max - x_min) + x_min
    return state

