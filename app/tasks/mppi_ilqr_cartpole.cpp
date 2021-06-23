
#include "mujoco.h"
#include "cstdio"
#include "cstdlib"
#include "cstring"
#include "glfw3.h"
#include "random"
#include "../../src/controller/controller.h"
#include "../../src/utilities/buffer_utils.h"
#include "../../src/utilities/buffer.h"
#include "../../src/controller/mppi_ddp.h"

// for sleep timers
#include <chrono>
#include <thread>

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
bool save_data    = false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_ENTER)
    {
        save_data = true;
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
    mj_activate(MUJ_KEY_PATH);

    // load and compile model
    char error[1000] = "Could not load binary model";

    // check command-line arguments
    if( argc<2 ) {
        m = mj_loadXML("../../../models/cartpole.xml", 0, error, 1000);

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
    StateVector x_desired; x_desired << 0, 0, 0, 0;
    CtrlVector u_desired; u_desired << 0;

    StateMatrix x_terminal_gain; x_terminal_gain.setIdentity();
    for(auto element = 0; element < n_jpos; ++element)
    {
        x_terminal_gain(element + n_jpos,element + n_jpos) = 0.01;
    }
    x_terminal_gain *= 10000;
    x_terminal_gain(0,0) *= 2;

    StateMatrix x_gain; x_gain.setIdentity();
    for(auto element = 0; element < n_jpos; ++element)
    {
        x_gain(element + n_jpos,element + n_jpos) = 0.01;
    }
    x_gain *= 0;

   CtrlMatrix u_gain;
    u_gain.setIdentity();
    u_gain *= 0.0001;

    CtrlVector u_control_1;
    StateVector x_state_1;

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);


    CtrlMatrix ddp_var; ddp_var.setIdentity();
    CtrlMatrix ctrl_var; ctrl_var.setIdentity();
    CtrlVector ctrl_mean; ctrl_mean.setZero();
    for(auto elem = 0; elem < n_ctrl; ++elem)
    {
        ctrl_var.diagonal()[elem] = 1;
        ddp_var.diagonal()[elem] = 0.001;
    }

    StateMatrix t_state_reg; t_state_reg.setIdentity();
    for(auto elem = 0; elem < n_jpos; ++elem)
    {
        t_state_reg.diagonal()[elem + n_jvel] = 100;
        t_state_reg.diagonal()[elem] = 10000;
    }

    StateMatrix r_state_reg; r_state_reg.setIdentity();
    for(auto elem = 0; elem < n_jpos; ++elem)
    {
        r_state_reg.diagonal()[elem + n_jvel] = 0;
        r_state_reg.diagonal()[elem] = 0;
    }


    CtrlMatrix control_reg; control_reg.setIdentity();
    for(auto elem = 0; elem < n_ctrl; ++elem)
    {
        control_reg.diagonal()[elem] = 0;
    }

    const auto running_cost = [&](const StateVector &state_vector, const CtrlVector &ctrl_vector, const mjData* data=nullptr, const mjModel *model=nullptr){
        StateVector state_error  = x_desired - state_vector;
        CtrlVector ctrl_error = u_desired - ctrl_vector;

        return (state_error.transpose() * r_state_reg * state_error + ctrl_error.transpose() * control_reg * ctrl_error)
                (0, 0);
    };

    const auto terminal_cost = [&](const StateVector &state_vector) {
        StateVector state_error = x_desired - state_vector;

        return (state_error.transpose() * t_state_reg * state_error)(0, 0);
    };


    MPPIDDPParams<n_ctrl> params {10, 75, 0.0001, 1, 1, ctrl_mean, ddp_var, ctrl_var};
    QRCostDDP<n_jpos + n_jvel, n_ctrl> qrcost(1, params, running_cost, terminal_cost);

    MPPIDDP<n_jpos + n_jvel, n_ctrl> pi(m, qrcost, params);

    // initial position
    d->qpos[0] = 0; d->qpos[1] = M_PI; d->qvel[0] = 0; d->qvel[1] = 0;

    CtrlMatrix R;
    StateMatrix Q;

    FiniteDifference<n_jpos + n_jvel, n_ctrl> fd(m);
    CostFunction<n_jpos + n_jvel, n_ctrl> cost_func(x_desired, u_desired, x_gain, u_gain, x_terminal_gain, m);
    ILQR<n_jpos + n_jvel, n_ctrl> ilqr(fd, cost_func, m, 75, 1, d, nullptr);

    // install control callback
    MyController<MPPIDDP<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl> control(m, d, pi);
    MyController<MPPIDDP<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::set_instance(&control);
    mjcb_control = MyController<MPPIDDP<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::dummy_controller;

    DummyBuffer d_buff;

/* ============================================CSV Output Files=======================================================*/
    std::vector<CtrlVector> pi_ctrl;
    std::vector<CtrlVector> ddp_ctrl;
    pi_ctrl.reserve(1000);
    ddp_ctrl.reserve(1000);

    std::string path = "/home/daniel/Repos/OptimisationBasedControl/data/";
    std::fstream pi_ctrl_file(path + ("pi_ctrl.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream ddp_ctrl_file(path + ("ddp_ctrl.csv"), std::fstream::out | std::fstream::trunc);

    std::fstream cost_mpc(path + ("cartpole_pi_ddp_cost_mpc.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream ctrl_data(path + ("cartpole_pi_ddp_ctrl.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream pos_data(path + ("cartpole_pi_ddp_pos.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream vel_data(path + ("cartpole_pi_ddp_vel.csv"), std::fstream::out | std::fstream::trunc);

    // use the first while condition if you want to simulate for a period.
    while(!glfwWindowShouldClose(window))
    {
        //  advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while( d->time - simstart < 1.0/60.0 )
        {
            d_buff.fill_buffer(d);
            mjcb_control = MyController<MPPIDDP<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::dummy_controller;
            ilqr.control(d);
            pi.control(d, ilqr._u_traj_cp, ilqr._covariance);
            ilqr._u_traj = pi.m_control_cp;
            mjcb_control = MyController<MPPIDDP<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::callback_wrapper;
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

    if(save_data)
    {
        BufferUtilities::save_to_file(ddp_ctrl_file, ddp_ctrl);
        BufferUtilities::save_to_file(pi_ctrl_file, pi_ctrl);
        d_buff.save_buffer(pos_data, vel_data, ctrl_data);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "Saved!" << std::endl;
        save_data = false;
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
