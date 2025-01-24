# Simple Neural MPC

This is a demo on applying neural networks on a mpc controllers for a differential drive robot that has to do trajectory tracking.

WORK IN PROGRESS


To use the code, install the environment with:  

```
pip install -e .
```

or with pixi:

```
pixi run build-acados && pixi install
```

Install l4casadi (CPU Version):
```
git clone https://github.com/Tim-Salzmann/l4casadi.git

pip install -r requirements_build.txt

pip install . --no-build-isolation
```

Build acados:
```
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init

cd external/acados && mkdir -p build && cd build && cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OSQP=ON .. && make -j8

pip install -e external/acados/interfaces/acados_template
```