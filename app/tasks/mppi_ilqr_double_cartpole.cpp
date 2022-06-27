

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include "cstdio"
#include "cstring"
#include "../../src/controller/controller.h"
#include "../../src/utilities/buffer_utils.h"
#include "../../src/utilities/buffer.h"
#include "../../src/controller/mppi_ddp.h"
#include "../../src/utilities/zmq_utils.h"

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
    if( act==GLFW_PRESS && key==GLFW_KEY_END)
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
        m = mj_loadXML("../../../models/double_cart_pole.xml", 0, error, 1000);

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
    StateVector x_desired; x_desired << 0, 0, 0, 0, 0, 0;
    CtrlVector u_desired; u_desired << 0;

    StateVector x_terminal_gain_vec; x_terminal_gain_vec << 700, 500, 250, 10, 200, 300;
    StateMatrix x_terminal_gain; x_terminal_gain = x_terminal_gain_vec.asDiagonal();


    StateVector x_gain_vec; x_gain_vec << 700, 500, 250, 0, 0, 0;
    StateMatrix x_gain; x_gain = x_gain_vec.asDiagonal();

    CtrlMatrix u_gain;
    u_gain.setIdentity();
    u_gain *= 20;

    CtrlMatrix du_gain;
    du_gain.setIdentity();
    du_gain *= 0;

    CtrlVector u_control_1;
    StateVector x_state_1;

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    CtrlVector ctrl_mean; ctrl_mean.setZero();
    CtrlMatrix ddp_var; ddp_var.setIdentity();
    CtrlMatrix ctrl_var; ctrl_var.setIdentity();
    for(auto elem = 0; elem < n_ctrl; ++elem)
    {
        ctrl_var.diagonal()[elem] = 1;
        ddp_var.diagonal()[elem] = 0.01;
    }

    StateMatrix t_state_reg; t_state_reg.setIdentity();
    for(auto elem = 0; elem < n_jpos; ++elem)
    {
        t_state_reg.diagonal()[elem + n_jvel] = 50000;
        t_state_reg.diagonal()[elem] = 100000;
    }
    t_state_reg.diagonal()[n_jvel] = 500;
    t_state_reg.diagonal()[0] = 100000;

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

    const auto terminal_cost = [&](const StateVector &state_vector, const mjData* data=nullptr, const mjModel *model=nullptr) {
        StateVector state_error = x_desired - state_vector;

        return (state_error.transpose() * t_state_reg * state_error)(0, 0);
    };

    FiniteDifference fd(m);
    CostFunction cost_func(x_desired, u_desired, x_gain, u_gain, du_gain, x_terminal_gain, m);
    ILQRParams ilqr_params {1e-6, 1.6, 1.6, 0, 180, 1};
    ILQR ilqr(fd, cost_func, ilqr_params, m, d, nullptr);

    MPPIDDPParams params {
        100, 100,0.0001, 0, 1, 1, 25,
        ctrl_mean, ddp_var, ctrl_var, {ilqr.m_u_traj_cp, ilqr._covariance}
    };
    QRCostDDP qrcost(params, running_cost, terminal_cost);
    MPPIDDP pi(m, qrcost, params);

    // initial position
    d->qpos[0] = 0; d->qpos[1] = M_PI; d->qpos[2] = 0; d->qvel[0] = 0; d->qvel[1] = 0; d->qvel[2] = 0;

    CtrlMatrix R;
    StateMatrix Q;

    // install control callback
    using ControlType = ILQR;
    MyController<ControlType, n_jpos + n_jvel, n_ctrl> control(m, d, ilqr);
    MyController<ControlType , n_jpos + n_jvel, n_ctrl>::set_instance(&control);
    mjcb_control = MyController<ControlType, n_jpos + n_jvel, n_ctrl>::dummy_controller;
    ilqr_params.iteration = 100;
    ilqr.control(d);
    ilqr_params.iteration = 1;

/* ============================================CSV Output Files=======================================================*/
    std::string path = "/home/daniel/Repos/OptimisationBasedControl/data/";
    printf ("Connecting to viewer server…\n");
    Buffer<RawTypeEig<CtrlVector>::type> ilqr_buffer{};
    Buffer<RawTypeEig<CtrlVector>::type> pi_buffer{};
    ZMQUBuffer<RawTypeEig<CtrlVector>::type> zmq_buffer(ZMQ_PUSH, "tcp://localhost:5555");
    zmq_buffer.push_buffer(&ilqr_buffer);
    zmq_buffer.push_buffer(&pi_buffer);
    std::fstream cost_mpc(path + ("cartpole_cost_mpc.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream ctrl_data(path + ("cartpole_ctrl.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream pos_data(path + ("cartpole_pos.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream vel_data(path + ("cartpole_vel.csv"), std::fstream::out | std::fstream::trunc);

    double cost;
    GenericBuffer<PosVector> pos_bt{d->qpos};   DataBuffer<GenericBuffer<PosVector>> pos_buff;
    GenericBuffer<VelVector> vel_bt{d->qvel};   DataBuffer<GenericBuffer<VelVector>> vel_buff;
    GenericBuffer<CtrlVector> ctrl_bt{d->ctrl}; DataBuffer<GenericBuffer<CtrlVector>> ctrl_buff;
    GenericBuffer<Eigen::Matrix<double, 1, 1>> cost_bt{&cost}; DataBuffer<GenericBuffer<Eigen::Matrix<double, 1, 1>>> cost_buff;

    pos_buff.add_buffer_and_file({&pos_bt, &pos_data});
    vel_buff.add_buffer_and_file({&vel_bt, &vel_data});
    ctrl_buff.add_buffer_and_file({&ctrl_bt, &ctrl_data});
    cost_buff.add_buffer_and_file({&cost_bt, &ctrl_data});

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
            mjcb_control = MyController<ControlType, n_jpos + n_jvel, n_ctrl>::dummy_controller;
            ilqr.control(d);
            ilqr_buffer.update(ilqr.cached_control.data(), true);
            pi_buffer.update(pi.cached_control.data(), false);
//            zmq_buffer.send_buffers();
            pos_buff.push_buffer(); vel_buff.push_buffer(); ctrl_buff.push_buffer(); cost_buff.push_buffer();
            mjcb_control = MyController<ControlType, n_jpos + n_jvel, n_ctrl>::callback_wrapper;
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
        pos_buff.save_buffer(); vel_buff.save_buffer(); ctrl_buff.save_buffer(); cost_buff.save_buffer();
        std::cout << "Saved!" << std::endl;
        save_data = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
