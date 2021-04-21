
#include "mujoco.h"
#include "cstring"
#include "glfw3.h"
#include "../../src/controller/controller.h"
#include "../../src/parameters/simulation_params.h"
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
bool button_left   = false;
bool button_middle = false;
bool button_right  = false;
bool save_data     = false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_HOME)
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
    // activate software
    mj_activate(MUJ_KEY_PATH);

    // load and compile model
    char error[1000] = "Could not load binary model";

    // check command-line arguments
    if( argc<2 ) {
        m = mj_loadXML("../../../models/finger.xml", 0, error, 1000);

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
    Eigen::Matrix<double, n_jpos + n_jvel, 1> x_desired; x_desired << 0, 0, 0, 0, 0, 0;
    Eigen::Matrix<double, n_ctrl, 1> u_desired; u_desired << 0, 0;

    Eigen::Matrix<double, n_jpos + n_jvel, n_jpos + n_jvel> x_terminal_gain; x_terminal_gain.setIdentity();
    for(auto element = 0; element < n_jpos; ++element)
    {
        x_terminal_gain(element + n_jpos,element + n_jpos) = 0.01;
    }
    x_terminal_gain *= 0;
    x_terminal_gain (2, 2) = 1500000000;
    x_terminal_gain (3, 3) = 50000 * 0.01;
    x_terminal_gain (4, 4) = 50000 * 0.01;
    x_terminal_gain (5, 5) = 5000000;

    Eigen::Matrix<double, n_jpos + n_jvel, n_jpos + n_jvel> x_gain; x_gain.setIdentity();
    for(auto element = 0; element < n_jpos; ++element)
    {
        x_gain(element + n_jpos,element + n_jpos) = 0.01;
    }

    x_gain *= 0;
    x_gain (3, 3) = 1 * 0.01;
    x_gain (4, 4) = 1 * 0.01;


    Eigen::Matrix<double, n_ctrl, n_ctrl> u_gain;
    u_gain.setIdentity();
    u_gain *= 100000;

    Eigen::Matrix<double, n_ctrl, 1> u_control_1;
    Eigen::Matrix<double, n_jpos + n_jvel, 1> x_state_1;

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // initial position
    d->qpos[0] = 0; d->qpos[1] = 0; d->qpos[2] = -.8;
    d->qvel[0] = 0; d->qvel[1] = 0; d->qvel[2] = 0;



    Eigen::Matrix<double, n_ctrl, n_ctrl> ddp_var; ddp_var.setIdentity();
    Eigen::Matrix<double, n_ctrl, n_ctrl> ctrl_var; ctrl_var.setIdentity();
    for(auto elem = 0; elem < n_ctrl; ++elem)
    {
        ctrl_var.diagonal()[elem] = 1;
        ddp_var.diagonal()[elem] = 0.0001;
    }

    Eigen::Matrix<double, n_jpos + n_jvel, 1> state_reg_vec;
    state_reg_vec << 0, 0, 500000, 5000, 5000, 5000;
    Eigen::Matrix<double,  n_jpos + n_jvel,  n_jpos + n_jvel> t_state_reg; t_state_reg = state_reg_vec.asDiagonal();


    Eigen::Matrix<double, n_ctrl, 1> control_reg_vec;
    state_reg_vec << 0, 0;
    Eigen::Matrix<double, n_ctrl, n_ctrl> control_reg; control_reg = control_reg_vec.asDiagonal();


    Eigen::Matrix<double, n_jpos + n_jvel, n_jpos + n_jvel> r_state_reg; r_state_reg.setIdentity();
    for(auto elem = 0; elem < n_jpos; ++elem)
    {
        r_state_reg.diagonal()[elem + n_jvel] = 200;
        r_state_reg.diagonal()[elem] = 0;
    }

    MPPIDDPParams<n_ctrl> params {400, 100, 1, 1, 1};
    QRCostDDP<n_jpos + n_jvel, n_ctrl> qrcost(
            ddp_var.inverse(), ctrl_var.inverse(), t_state_reg, r_state_reg, control_reg, x_desired, u_desired, params
            );
    MPPIDDP<n_jpos + n_jvel, n_ctrl> pi(m, qrcost, params);


    FiniteDifference<n_jpos + n_jvel, n_ctrl> fd(m);
    CostFunction<n_jpos + n_jvel, n_ctrl> cost_func(x_desired, u_desired, x_gain, u_gain, x_terminal_gain, m);
    ILQR<n_jpos + n_jvel, n_ctrl> ilqr(fd, cost_func, m, 100, 1, d, nullptr);

    // install control callback
    MyController<ILQR<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl> control(m, d, ilqr);
    MyController<ILQR<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::set_instance(&control);
    mjcb_control = MyController<ILQR<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::dummy_controller;

    DummyBuffer d_buff;

//    ilqr.control(d);

/* ============================================CSV Output Files=======================================================*/
    std::string path = "/home/daniel/Repos/OptimisationBasedControl/data/";

    std::fstream cost_mpc(path + ("finger_cost_mpc_pi_ddp_0.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream ctrl_data(path + ("finger_ctrl.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream pos_data(path + ("finger_pos.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream vel_data(path + ("finger_vel.csv"), std::fstream::out | std::fstream::trunc);
    std::vector<double> cost_buffer;

/* ==================================================Simulation=======================================================*/

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
            cost_buffer.emplace_back(pi.traj_cost);
            d_buff.fill_buffer(d);
            mjcb_control = MyController<ILQR<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::dummy_controller;
            ilqr.control(d);
            pi.control(d, ilqr._u_traj);
            ilqr._u_traj = pi.m_control;
//            const auto neg_cost = ilqr.exp_cost_reduction.front();
//            params.IMPORTANCE = 1.0/(1.0 + std::exp((neg_cost)*2.2))*2;
//            params.IMPORTANCE += (1/sqrt(M_PI*2*1e-4)*std::exp(-neg_cost*neg_cost/(2*1e-4)))/130;
//            params.IMPORTANCE = 1;
//            std::cout << "cost red: " << ilqr.exp_cost_reduction.front() << "\n";

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

        if(save_data)
        {
            BufferUtilities::save_to_file(cost_mpc, cost_buffer);
            d_buff.save_buffer(pos_data, vel_data, ctrl_data);
            std::cout << "Saved!" << std::endl;
            save_data = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
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
