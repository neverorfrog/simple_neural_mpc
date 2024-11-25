# OCP for Unicycle Trajectory Tracking

## State Variables and Control Inputs
Let:

$ \mathbf{x} $ be the state vector with dimension $ n_x $  
$ \mathbf{u} $ be the control input vector with dimension $ n_u $  

## Cost Function
The cost function is defined as a Linear Least Squares problem. The stage cost and terminal cost are given by:

### Mapping State and Control Variables

$ \mathbf{V}_x $ maps the state vector $ \mathbf{x}_k $ to the cost function.
$ \mathbf{V}_u $ maps the control input vector $ \mathbf{u}_k $ to the cost function.
Forming the Output Vector $ \mathbf{y}_k $:

The output vector $ \mathbf{y}_k $ is formed by combining the contributions of the state and control variables using $ \mathbf{V}_x $ and $ \mathbf{V}_u $: 
$$ \mathbf{y}_k = \mathbf{V}_x \mathbf{x}_k + \mathbf{V}_u \mathbf{u}_k $$

### Stage Cost

The cost at each time step $ k $ is calculated using the output vector $ \mathbf{y}k $, the reference vector $ \mathbf{y}{\text{ref}} $, and the weight matrix $ \mathbf{W} $: 
$$ J_k = \left( \mathbf{y}k - \mathbf{y}{\text{ref}} \right)^T \mathbf{W} \left( \mathbf{y}k - \mathbf{y}{\text{ref}} \right) $$

$ \mathbf{y}_{\text{ref}} $ is the reference trajectory

### Terminal Cost

The terminal cost is calculated using the terminal state $ \mathbf{x}_N $, the terminal reference vector $ \mathbf{y}{\text{ref},e} $, and the terminal weight matrix $ \mathbf{W}_e $: 
$$ J_T = \left( \mathbf{x}_N - \mathbf{y}{\text{ref},e} \right)^T \mathbf{W}_e \left( \mathbf{x}_N - \mathbf{y}{\text{ref},e} \right) $$

## Constraints
The control input constraints are defined as: 
$$ \mathbf{u}_{\text{min}} \leq \mathbf{u}k \leq \mathbf{u}{\text{max}} $$ where:

$$ \mathbf{u}{\text{min}} = -F{\text{max}} $$
$$ \mathbf{u}{\text{max}} = +F{\text{max}} $$
The initial state constraint is: $ \mathbf{x}(0) = \mathbf{X}_0 $

## Summary
The optimal control problem can be summarized as:

Minimize: 
$$ J = \sum_{k=0}^{N-1} \left( \mathbf{y}k - \mathbf{y}{\text{ref}} \right)^T \mathbf{W} \left( \mathbf{y}k - \mathbf{y}{\text{ref}} \right) + \left( \mathbf{x}N - \mathbf{y}{\text{ref},e} \right)^T \mathbf{W}_e \left( \mathbf{x}N - \mathbf{y}{\text{ref},e} \right) $$

Subject to: 
$$ \mathbf{u}_{\text{min}} \leq \mathbf{u}k \leq \mathbf{u}{\text{max}} \quad \text{for} ; k = 0, \ldots, N-1 $$ 
$$ \mathbf{x}(0) = \mathbf{X}_0 $$

This formulation defines the OCP that the Acados solver will use to find the optimal control inputs for the robot.