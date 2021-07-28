
#include "mujoco.h"
#include "cstring"
#include "glfw3.h"
#include "../../src/controller/controller.h"
#include "../../src/utilities/buffer_utils.h"

// for sleep timers
#include <chrono>
#include <thread>
#include<iostream>


using namespace std;
using namespace std::chrono;
// local variables include

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
bool end_sim      = false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
//        mj_resetData(m, d);
//        mj_forward(m, d);
        end_sim = true;
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
        m = mj_loadXML("../../../models/doubleintegrator.xml", 0, error, 1000);

    }else {
        if (strlen(argv[1]) > 4 && !strcmp(argv[1] + strlen(argv[1]) - 4, ".mjb")) {
            m = mj_loadModel(argv[1], 0);
        }
        else {
            m = mj_loadXML(argv[1], 0, error, 1000);
        }
    }
    if( !m ) {
        mju_error_s("Load model error: %s", error);
    }

    m->opt.timestep = 0.01;
    // make data
    d = mj_makeData(m);

    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // Assert against model params (literals)
    using namespace SimulationParameters;
    assert(m->nv == n_jvel);
    assert(m->nq == n_jpos);
    assert(m->nu == n_ctrl);

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

    // setup cost params
    StateVector x_desired; x_desired << 2, 0;
    CtrlVector u_desired; u_desired << 0;

    StateMatrix x_terminal_gain; x_terminal_gain.setIdentity();
    for(auto element = 0; element < n_jpos; ++element)
    {
        x_terminal_gain(element + n_jpos,element + n_jpos) = 0.01;
    }
    x_terminal_gain *= 25000;

    StateMatrix x_gain; x_gain.setIdentity();
    for(auto element = 0; element < n_jpos; ++element)
    {
        x_gain(element + n_jpos,element + n_jpos) = 0.01;
    }
    x_gain *= 0;

    CtrlMatrix u_gain;
    u_gain.setIdentity();
    u_gain *= 0;

    CtrlMatrix du_gain;
    du_gain.setIdentity();
    du_gain *= 1;

    CtrlVector u_control_1;
    StateVector x_state_1;

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // initial position
    d->qpos[0] = 0; d->qvel[0] = 0;

    CtrlMatrix R;
    StateMatrix Q;

    FiniteDifference<n_jpos + n_jvel, n_ctrl> fd(m);
    CostFunction<n_jpos + n_jvel, n_ctrl> cost_func(x_desired, u_desired, x_gain, u_gain, du_gain, x_terminal_gain, m);
    ILQRParams params {1e-6, 1.6, 1.6, 0, 75, 5};
    ILQR<n_jpos + n_jvel, n_ctrl> ilqr(fd, cost_func, params, m, d, nullptr);

    // install control callback
    MyController<ILQR<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl> control(m, d, ilqr);
    MyController<ILQR<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::set_instance(&control);
    mjcb_control = MyController<ILQR<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::dummy_controller;


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
            std::cout << "Error: " << x_desired(0,0) - d->qpos[0] << std::endl;
            mjcb_control = MyController<ILQR<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::dummy_controller;
            ilqr.control(d);
            mjcb_control = MyController<ILQR<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::callback_wrapper;
            mj_step(m, d);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
