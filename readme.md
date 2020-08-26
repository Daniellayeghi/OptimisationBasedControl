# Overview

This repo is primarily based on implementation of trajectory optimisation algorithms. The two main algorithms implemented atm are iLQR and MPPI.
The **app** directory has the source files for both algorithms running on environments such as finger-spinner, cartpole, acrobot, double integrator in the context of an MPC loop.

<<<<<<< HEAD
## Dependencies
To build the following dependencies are required:
- Eigen3
- Gtest (Cloned and built by FetchContent)
- MuJoCo 2.0

## Instruction
- Clone the repo.
- Use the ```find_mjkey``` script to set MuJoCo key to variable ```DMUJ_KEY_PATH```. Otherwise set it yourself.
- Copy MuJoCo binaries to directory ``libraries``.
- Make sure glfw is installed so cmake can linked to it.
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
The gifs below show some of the iLQR and MPPI workingo on different environments
=======

# Tasks
The gifs below show iLQR and MPPI working on different environments
- iLQR (Finger Spinner)

![Alt Text](gifs/finger_ilqr.gif) 

- MPPI (Finger Spinner)

![Alt Text](gifs/mppi_finger.gif)

- iLQR (Acrobot)

![Alt Text](gifs/acrobot_ilqr.gif)

- iLQR (Cartpole)

![Alt Text](gifs/cp_ilqr.gif)

- MPPI (Cartpole)

![Alt Text](gifs/cp_mppi.gif)


