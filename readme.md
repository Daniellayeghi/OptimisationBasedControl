## Overview

This repo is primarily based on implementation of trajectory optimisation algorithms. The two main algorithms implemented atm are iLQR and MPPI.
The **app** directory has the source files for both algorithms running on environments such as finger-spinner, cartpole, acrobot, double integrator in the context of an MPC loop.

## Dependencies
To build the following dependencies are required:
- Eigen3
- Gtest (Cloned and built by FetchContent)
- MuJoCo 2.0

## Instruction
- Clone the repo.
- Use the ```find_mjkey``` script to set MuJoCo key to variable ```MUJ_KEY_PATH```. Otherwise set it yourself.
- Copy MuJoCo binaries to directory ``libraries``.
- Make sure glfw is installed so cmake can link to it.
- Build
~~~
cd OptimisationBasedControl
source find_mjkey
mkdir libraries && cp /path/to/mujoco/bin/*.so* libraries/
mkdir build-release && cd build-release
cmake -DMUJ_KEY_PATH=$MUJ_KEY_PATH -DCMAKE_BUILD_TYPE=Release ..
make -j nThreads
~~~
## Tasks

This repo is the not very clean implementation of our ICRA 2022 paper:

[Optimal Control via Inference and Numerical Optimization](https://arxiv.org/pdf/2109.11361.pdf)


<p align="center">
  <img src="./gifs/Optimal%20Control%20via%20Combined%20Inference%20and%20Numerical%20Optimization(1).gif" alt="animated" />
</p>


