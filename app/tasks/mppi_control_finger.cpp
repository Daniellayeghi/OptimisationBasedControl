

#include "mujoco.h"
#include "cstdio"
#include "cstdlib"
#include "cstring"
#include "glfw3.h"
#include "../../src/controller/controller.h"
#include "../../src/controller/simulation_params.h"
#include "../../src/utilities/buffer_utils.h"


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
bool end_sim       = false;
bool save_data     = false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if(act==GLFW_PRESS && key==GLFW_KEY_END)
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

    m->opt.timestep = 0.01;
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

    using namespace SimulationParameters;
    assert(m->nv == n_jvel);
    assert(m->nq == n_jpos);
    assert(m->nu == n_ctrl);

    Eigen::Matrix<double, n_ctrl, 1> u_control_1;
    Eigen::Matrix<double, n_jpos + n_jvel, 1> x_state_1;

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    Eigen::Matrix<double, n_ctrl, n_ctrl> R = Eigen::Matrix<double, n_ctrl, n_ctrl>::Identity() * 100;
    Eigen::Matrix<double, n_jpos + n_jvel, n_jpos + n_jvel> Q;

    FiniteDifference<n_jpos + n_jvel, n_ctrl> fd(m);

    MPPIParams params {400, 50, 0.99, 500};

    QRCost<n_jpos + n_jvel, n_ctrl> qrcost(R, Q, x_state_1, u_control_1);
    MPPI<n_jpos + n_jvel, n_ctrl> pi(m, qrcost, params);

    // install control callback
    MyController<MPPI<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl> control(m, d, pi);
    MyController<MPPI<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::set_instance(&control);
    mjcb_control = MyController<MPPI<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::callback_wrapper;

    // initial position
    d->qpos[0] = 0; d->qpos[1] = 0; d->qpos[2] = -0.8;
    d->qvel[0] = 0; d->qvel[1] = 0; d->qvel[2] = 0;
/* ============================================CSV Output Files=======================================================*/
    std::string path = "/home/daniel/Repos/OptimisationBasedControl/data/";

    std::fstream cost_mpc(path + ("finger_cost_mpc_mppi.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream ctrl_data(path + ("finger_ctrl_mppi.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream pos_data(path + ("finger_pos_mppi.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream vel_data(path + ("finger_vel_mppi.csv"), std::fstream::out | std::fstream::trunc);

    Eigen::Matrix<double, n_jpos, 1> pos;
    Eigen::Matrix<double, n_jvel, 1> vel;
    Eigen::Matrix<double, n_ctrl, 1> ctrl;

    std::vector<double> cost_buffer;
    std::vector<Eigen::Matrix<double, n_jpos, 1>> pos_buffer;
    std::vector<Eigen::Matrix<double, n_jvel, 1>> vel_buffer;
    std::vector<Eigen::Matrix<double, n_ctrl, 1>> ctrl_buffer;

/* ==================================================Simulation=======================================================*/
    auto start = high_resolution_clock::now();
    auto end   = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    int iteration = 1;
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
//            std::cout << pi.traj_cost << "\n";
            pos_buffer.emplace_back((pos << d->qpos[0], d->qpos[1], d->qpos[2]).finished());
            vel_buffer.emplace_back((vel << d->qvel[0], d->qvel[1], d->qvel[2]).finished());
            ctrl_buffer.emplace_back((ctrl << d->ctrl[0], d->ctrl[1]).finished());

            mjcb_control = MyController<MPPI<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::dummy_controller;
            start = high_resolution_clock::now();
            pi.control(d);
            end = high_resolution_clock::now();
            duration += duration_cast<milliseconds>(end - start).count();
            std::cout << duration/iteration << std::endl;
            mjcb_control = MyController<MPPI<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>::callback_wrapper;
            mj_step(m, d);
            ++iteration;
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
            BufferUtilities::save_to_file(pos_data, pos_buffer);
            BufferUtilities::save_to_file(vel_data, vel_buffer);
            BufferUtilities::save_to_file(ctrl_data, ctrl_buffer);

            std::cout << "Saved!" << std::endl;
            std::cout << "Duration: " << duration/iteration << "\n";
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
