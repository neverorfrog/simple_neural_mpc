



# different type of models and propagation:

1) imitation setting:
    - pure mlp(x, u) = dot x
    - rely on multi-step integration for obtain the solution

2) PINN: 
    - mlp(x(t), u(t)) = x(t+1) --> we obtain the exact integration, no need it for manually integrate the output
    - trained with a loss that ensure that the derivative of the net is equals to the kinematic/dynamic model