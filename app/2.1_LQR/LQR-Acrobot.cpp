//------------------------------------------//
//  This file is a modified version of      //
//  basics.cpp, which was distributed as    //
//  part of MuJoCo,  Written by Emo Todorov //
//  Copyright (C) 2017 Roboti LLC           //
//  Modifications by Atabak Dehban          //
//------------------------------------------//


#include <iostream>

#include "mujoco.h"
#include "cstdio"
#include "cstdlib"
#include "cstring"
#include "glfw3.h"
#include "../../src/controller/controller.h"
#include "../../src/controller/cost_function.h"
// for sleep timers
#include <chrono>
#include <thread>

// local variables include

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjModel* m_cp = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;


double running(Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
               Eigen::Matrix<double, Eigen::Dynamic, 1> &u)
{
    return (x.transpose() * x + u.transpose() * u)(0, 0);
}


double terminal (Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
                 Eigen::Matrix<double, Eigen::Dynamic, 1> &u)
{
    return 1000.0;
}


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}


// main function
int main(int argc, const char** argv)
{
    // activate software
    mj_activate(MUJ_KEY_PATH);

    // load and compile model
    char error[1000] = "Could not load binary model";

    // check command-line arguments
    if( argc<2 ) {
        m = mj_loadXML("../../../models/Acrobot.xml", 0, error, 1000);
        m_cp = mj_loadXML("../../../models/Acrobot.xml", 0, error, 1000);
    }else {
        if (strlen(argv[1]) > 4 && !strcmp(argv[1] + strlen(argv[1]) - 4, ".mjb")) {
            m = mj_loadModel(argv[1], 0);
            m_cp = mj_loadModel(argv[1], 0);
        }
        else {
            m = mj_loadXML(argv[1], 0, error, 1000);
            m_cp = mj_loadXML(argv[1], 0, error, 1000);
        }
    }
    if( !m or !m_cp) {
        mju_error_s("Load model error: %s", error);
    }

    std::cout << m << "      " << m_cp << std::endl;
    // make data
    d = mj_makeData(m);
    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);                // space for 2000 objects
    mjr_makeContext(m, &con, mjFONTSCALE_150);   // model-specific context

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    FiniteDifference fd(m_cp);
    CostFunction cost_func;
    MyController control(m, d, fd, cost_func);
    ILQR ilqr(fd, cost_func, m, 100);
    MyController::set_instance(&control);

    // install control callback
    mjcb_control = MyController::callback_wrapper;

    // initial position
//    d->qpos[0] = M_PI/10.0;
    d->qpos[0] = M_PI_2/2;
    d->qpos[1] = 0.0;
    d->qvel[0] = 0.0;
    d->qvel[1] = 0.0;

    std::cout << m->nv << "\n";
    // use the first while condition if you want to simulate for a period.
    while( !glfwWindowShouldClose(window))
    {
        //  advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while( d->time - simstart < 1.0/60.0 )
        {
            mjcb_control = MyController::callback_wrapper;
            mj_step(m, d);
            mjcb_control = MyController::dummy_controller;
            cost_func.derivatives(d);
            ilqr.backward_pass();
            std::cout << "Full derivatives" << "\n" << fd.f_x(d) << "\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif
    return 1;
}
