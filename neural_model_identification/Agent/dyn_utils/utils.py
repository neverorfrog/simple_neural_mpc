import torch 
import numpy as np



def euler_integration(x, x_dot, delta_t):
    return x + x_dot * delta_t
