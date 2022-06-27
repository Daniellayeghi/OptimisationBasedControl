## Overview
This repo is the not very clean implementation of our ICRA 2022 paper:

[Optimal Control via Inference and Numerical Optimization](https://arxiv.org/pdf/2109.11361.pdf)


<p align="center">
  <img src="./gifs/Optimal%20Control%20via%20Combined%20Inference%20and%20Numerical%20Optimization(1).gif" alt="animated" />
</p>

## Dependencies
To build the following dependencies are required:
- Eigen3
- Gtest (Cloned and built by FetchContent)
- MuJoCo >= 2.2 
- GCC >= 8 compiler for both Mujoco and OpenMP

## Instruction [WIP]
- Clone the repo.
- Make sure glfw and Mujoco are installed so cmake can link to it.
- Build
~~~
cd OptimisationBasedControl
mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j nThreads
~~~
