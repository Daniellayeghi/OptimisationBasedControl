#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include "cstdio"
#include "cstring"
#include "../../src/controller/controller.h"
#include "../../src/utilities/buffer_utils.h"
#include "../../src/utilities/buffer.h"
#include "../../src/utilities/zmq_utils.h"
#include "../../src/controller/par_mppi_ddp.h"
#include "../../src/controller/mppi_ddp.h"
#include "../../src/utilities/mujoco_utils.h"
#include "../../src/utilities/math_utils.h"

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

    mj_activate(MUJ_KEY_PATH);

    // load and compile model
    char error[1000] = "Could not load binary model";

    std::string model_path = "../../../models/", name = "doubleintegrator";

    // check command-line arguments
    if( argc<2 ) {
        m = mj_loadXML((model_path + name + ".xml").c_str(), 0, error, 1000);

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
    printf("POS: %i, VEL: %i, CTRL: %i \n", m->nq, m->nv, m->nu);
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
    StateVector x_desired; x_desired << 0, 0;
    Eigen::Ref<PosVector> pos_des = x_desired.block(0,0, n_jpos, 1);
    Eigen::Ref<VelVector> vel_des = x_desired.block(n_jpos, 0, n_jvel, 1);
    CtrlVector u_desired; u_desired << 0;

    StateVector x_terminal_gain_vec; x_terminal_gain_vec << 1, 1;
    StateMatrix x_terminal_gain = x_terminal_gain_vec.asDiagonal();

    StateVector x_gain_vec; x_gain_vec << 1, 1;
    StateMatrix x_gain = x_gain_vec.asDiagonal();
    CtrlMatrix u_gain; u_gain << 1;
    CtrlMatrix du_gain; du_gain << 0;

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
        ctrl_var.diagonal()[elem] = 0.015;
        ddp_var.diagonal()[elem] = 0.001;
    }

    StateMatrix t_state_reg = x_terminal_gain;
    StateMatrix r_state_reg = x_gain;

    CtrlMatrix control_reg;
    control_reg = u_gain * 0;

    const auto running_cost =
            [](const StateVector& x_err, const CtrlVector& u_err, const StateMatrix& x_gain, const CtrlMatrix& u_gain, const mjData* d, const mjModel* m){
                return (x_err.transpose() * x_gain * x_err + u_err.transpose() * u_gain * u_err)
                        (0, 0);
            };

    const auto terminal_cost =
            [](const StateVector& x_err, const CtrlVector& u_err, const StateMatrix& x_gain, const CtrlMatrix& u_gain, const mjData* d, const mjModel* m){
                return (x_err.transpose() * x_gain * x_err)(0, 0);
            };

    using namespace MathUtils;
    const double lim = 10;
    PosVector init_pos = PosVector::Zero();
    PosVector goal_pos = PosVector::Zero();
    Eigen::Map<PosVector> pos_map(d->qpos);
    Eigen::Map<VelVector> vel_map(d->qvel);

    const auto seed = 3;
    for(auto goal = 1; goal < 2; ++goal)
    {
//        random_iid_data_const_bound<double, n_jpos>(goal_pos.data(), lim);
//        x_desired.block(0, 0, n_jpos, 1) = goal_pos;
        for (auto init = 1; init < 10; ++init)
        {
            // Init random pos
            MathUtils::Rand::random_iid_data_const_bound<double, n_jpos>(init_pos.data(), lim);
            std::copy(init_pos.data(), init_pos.data() + n_jpos, d->qpos);

            FiniteDifference fd(m);
            QRCst cost_func(x_desired, x_gain, x_terminal_gain, u_gain, nullptr);
            ILQRParams ilqr_params {1e-6, 1.6, 1.6, 0, 75, 1,  false};
            ILQR ilqr(fd, cost_func, ilqr_params, m, d, nullptr);

            // To show difference in sampling try 3 samples
            MPPIDDPParamsPar params{
                    200, 75, 0.1, 1, 1, 1, 1000, ctrl_mean,
                    ddp_var, ctrl_var, {ilqr.m_u_traj_cp, ilqr.m_covariance}, 1
            };

            MPPIDDPCstParams p{1, 0.1, ctrl_var.inverse()};
            PICost cst(x_desired, x_gain, x_terminal_gain, control_reg, running_cost, terminal_cost, p);
            MPPIDDPPar pi(m, cst, params);

            // install control callback
            using ControlType = MPPIDDPPar;
            MyController<ControlType, n_jpos + n_jvel, n_ctrl> control(m, d, pi);
            MyController<ControlType, n_jpos + n_jvel, n_ctrl>::set_instance(&control);

/* ============================================CSV Output Files=======================================================*/
            const std::string path = "/home/daniel/Repos/OptimisationBasedControl/data/";
            const auto goal_state = std::to_string(int(x_desired(0))) + std::to_string(int(x_desired(1)));
            const auto start_state = std::to_string(int(init_pos(0))) + std::to_string(0);

            const std::string mode = std::to_string(goal) + std::to_string(init) + "_start_" + start_state + "_" + goal_state + "_goal_";
            std::fstream state_file(path + mode + name + "_state_00"  ".csv", std::fstream::out | std::fstream::trunc);
            std::fstream value_file(path + mode + name + "_value_00"  ".csv", std::fstream::out | std::fstream::trunc);
            GenericBuffer<StateVector> state_bt{pi.m_state_value.first.front().data()};
            DataBuffer<GenericBuffer<StateVector>> state_buff;
            GenericBuffer<Eigen::Matrix<double, 1, 1>> value_bt{&pi.m_state_value.second.front()};
            DataBuffer<GenericBuffer<Eigen::Matrix<double, 1, 1>>> value_buff;

            state_buff.add_buffer_and_file({&state_bt, &state_file});
            value_buff.add_buffer_and_file({&value_bt, &value_file});

            /* Use REQ because we want to make sure we recieved all info*/
            printf("Connecting to viewer server…\n");
            ZMQUBuffer<RawTypeEig<CtrlVector>::type> zmq_buffer(ZMQ_PUSH, "tcp://localhost:5555");

            std::vector<BufferParams<scalar_type, char>> buffer_params{
                {ilqr.cached_control.data(),ilqr.cached_control.data() + n_ctrl, 'q'},
                {pi.cached_control.data(), pi.cached_control.data() + n_ctrl, 'i'}
            };

            SimpleBuffer<RawTypeEig<CtrlVector>::scalar, char> simp_buff(buffer_params);
/* ==================================================Simulation=======================================================*/
            // use the first while condition if you want to simulate for a period.
            while (!glfwWindowShouldClose(window)) {
                //  advance interactive simulation for 1/60 sec
                //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
                //  this loop will finish on time for the next frame to be rendered at 60 fps.
                //  Otherwise add a cpu timer and exit this loop when it is time to render.
                mjtNum simstart = d->time;
                while (d->time - simstart < 1.0 / 60.0) {
                    mjcb_control = MyController<ControlType, n_jpos + n_jvel, n_ctrl>::dummy_controller;
                    ilqr.control(d, false);
                    pi.control(d, false);
                    simp_buff.update_buffer();
                    ilqr.m_u_traj = pi.m_u_traj;
                    zmq_buffer.send_buffer(simp_buff.get_buffer(), simp_buff.get_buffer_size());
                    state_buff.push_buffer();
                    value_buff.push_buffer();
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

                if (save_data or ((pos_des - pos_map).norm() < 1e-1 and (vel_des - vel_map).norm() < 1e-1)) {
                    state_buff.save_buffer();
                    value_buff.save_buffer();
                    std::cout << "Saved!" << std::endl;
                    save_data = false;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    break;
                }

            }
        }
    }
    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

//     terminate ipc
//    zmq_close (requester);
//    zmq_ctx_destroy (context);

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif
    return 1;
}

