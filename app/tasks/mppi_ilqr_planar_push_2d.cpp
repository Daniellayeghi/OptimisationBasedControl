
#include "mujoco.h"
#include "cstring"
#include "glfw3.h"
#include "../../src/controller/controller.h"
#include "../../src/utilities/buffer_utils.h"
#include "../../src/utilities/buffer.h"
#include "../../src/utilities/mujoco_utils.h"
#include "../../src/utilities/zmq_utils.h"
#include "../../src/controller/mppi_ddp.h"
#include <chrono>
#include <thread>

// for sleep timers
#include <chrono>
#include <thread>
#include <random>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using namespace std;
using namespace std::chrono;
// local variables include
namespace {
// MuJoCo data structures
    mjModel *m = NULL;                  // MuJoCo model
    mjData *d = NULL;                   // MuJoCo data
    mjvCamera cam;                      // abstract camera
    mjvOption opt;                      // visualization options
    mjvScene scn;                       // abstract scene
    mjrContext con;                     // custom GPU context

// mouse interaction
    bool button_left = false;
    bool button_middle = false;
    bool button_right = false;
    bool save_data = false;
    double lastx = 0;
    double lasty = 0;

    std::random_device r;

// Choose a random mean between 1 and 6
    std::default_random_engine e1(r());
    std::uniform_real_distribution<double> uniform_dist(-10, 10);
    int mean = uniform_dist(e1);


    void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
    {
        // backspace: reset simulation
        if( act==GLFW_PRESS && key==GLFW_KEY_HOME)
        {
            save_data = true;
        }
    }

// mouse button callback
    void mouse_button(GLFWwindow *window, int button, int act, int mods) {
        // update button state
        button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
        button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
        button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

        // update mouse position
        glfwGetCursorPos(window, &lastx, &lasty);
    }


// mouse move callback
    void mouse_move(GLFWwindow *window, double xpos, double ypos) {
        // no buttons down: nothing to do
        if (!button_left && !button_middle && !button_right)
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
        bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                          glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

        // determine action based on mouse button
        mjtMouse action;
        if (button_right)
            action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        else if (button_left)
            action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        else
            action = mjMOUSE_ZOOM;

        // move camera
        mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
    }


// scroll callback
    void scroll(GLFWwindow *window, double xoffset, double yoffset) {
        // emulate vertical mouse motion = 5% of window height
        mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
    }
}


// main function
int main(int argc, const char** argv)
{
    mj_activate(MUJ_KEY_PATH);

    // load and compile model
    char error[1000] = "Could not load binary model";


    std::string model_path = "../../../models/contact_comp/", name = "point_mass_tendon";
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



    StateVector x_desired; x_desired << 0, 0 ,0, 0.25, -0.22, 0.022, 0, 0, 0, 0;
    CtrlVector u_desired; u_desired << 0, 0, 0;

    StateVector x_terminal_gain_vec;
    x_terminal_gain_vec << 0, 0, 0, 100000, 100000, 0, 0, 0, 500, 500;
    StateMatrix x_terminal_gain; x_terminal_gain = x_terminal_gain_vec.asDiagonal();


    StateVector x_gain_vec;
    x_gain_vec << 0, 0, 0, 100000, 100000, 0, 0, 0, 0, 0;
    StateMatrix x_gain; x_gain = x_gain_vec.asDiagonal();


    CtrlVector u_gain_vec; u_gain_vec << 0.0000001, 0.0000001, 0.0000001;
    CtrlMatrix u_gain;
    u_gain = u_gain_vec.asDiagonal();

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

    // initial position     StateVector initial_state; initial_state <<  0, 0, 0, 0.20536, 0.1585, 0.0223, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    StateVector initial_state; initial_state <<  0, 0 ,0, -0.15, 0.42, 0, 0, 0, 0, 0;
    CtrlVector ctrl_mean; ctrl_mean.setZero();
    CtrlMatrix ddp_var; ddp_var.setIdentity();
    CtrlMatrix ctrl_var; ctrl_var.setIdentity();
    for(auto elem = 0; elem < n_ctrl; ++elem)
    {
        ctrl_var.diagonal()[elem] = 0.25;
        ddp_var.diagonal()[elem] = 0.001;
    }

    StateMatrix t_state_reg; t_state_reg = x_terminal_gain;
    StateMatrix r_state_reg; r_state_reg = x_gain;

    CtrlVector control_reg_vec;
    control_reg_vec = u_gain_vec;
    CtrlMatrix control_reg = u_gain_vec.asDiagonal();

    const auto collision_cost = [](const mjData* data=nullptr, const mjModel *model=nullptr){
        std::array<int, 4> body_list {{0, 1, 2, 3}};

        if(data and model)
            for(auto i = 0; i < data->ncon; ++i)
            {
                // Contact with world body (0) is always there so ignore that.
                auto elem_1 = std::find(body_list.begin(), body_list.end(), model->geom_bodyid[data->contact[i].geom1]);
                auto elem_2 = std::find(body_list.begin(), body_list.end(), model->geom_bodyid[data->contact[i].geom2]);
                bool world_contact = elem_1 == body_list.begin() or elem_2 == body_list.begin();
                bool check_1 = elem_1 != body_list.end(), check_2 = elem_2 != body_list.end();
                if (check_1 != check_2 and not world_contact)
                    return true;
            }
        return false;
    };

    const auto running_cost = [&](const StateVector &state_vector, const CtrlVector &ctrl_vector, const mjData* data=nullptr, const mjModel *model=nullptr){
        StateVector state_error  = x_desired - state_vector;
        CtrlVector ctrl_error = u_desired - ctrl_vector;
        return (state_error.transpose() * r_state_reg * state_error + ctrl_error.transpose() * control_reg * ctrl_error)
        (0, 0) + (not collision_cost(data, model) * 10000) * (state_error.transpose() * r_state_reg * state_error)(0, 0);
    };

    const auto terminal_cost = [&](const StateVector &state_vector, const mjData* data=nullptr, const mjModel *model=nullptr) {
        StateVector state_error = x_desired - state_vector;

        return (state_error.transpose() * t_state_reg * state_error)(0, 0) + not collision_cost(data, model) * (state_error.transpose() * t_state_reg * state_error)(0, 0);
    };

    std::array<unsigned int, 5> seeds {{7, 8, 9, 10, 11}};
    for (const auto seed : seeds)
    {
        std::copy(initial_state.data(), initial_state.data()+n_jpos, d->qpos);
        std::copy(initial_state.data()+n_jpos, initial_state.data()+state_size, d->qvel);
        //10 samples work original params 1 with importance 1/0 damping at 3 without mean update and 0.005 timestep
        // 40 and 10 and 100 samples with 10 lmbda and 1 importance with mean/2 update timestep 0.005 and damping 3
        MPPIDDPParams params{30, 75, 0.1, 0, 1, 1, 1e-8, ctrl_mean, ddp_var, ctrl_var, seed};
        QRCostDDP<n_jpos + n_jvel, n_ctrl> qrcost(params, running_cost, terminal_cost);
        MPPIDDP<n_jpos + n_jvel, n_ctrl> pi(m, qrcost, params);

        FiniteDifference<n_jpos + n_jvel, n_ctrl> fd(m);
        CostFunction<n_jpos + n_jvel, n_ctrl> cost_func(x_desired, u_desired, x_gain, u_gain, du_gain, x_terminal_gain, m);
        ILQRParams ilqr_params{1e-6, 1.6, 1.6, 0, 75, 1};
        ILQR<n_jpos + n_jvel, n_ctrl> ilqr(fd, cost_func, ilqr_params, m, d, nullptr);

        // install control callback
        using ControlType = MPPIDDP<n_jpos + n_jvel, n_ctrl>;
        MyController<ControlType, n_jpos + n_jvel, n_ctrl> control(m, d, pi);
        MyController<ControlType, n_jpos + n_jvel, n_ctrl>::set_instance(&control);
        mjcb_control = MyController<ControlType, n_jpos + n_jvel, n_ctrl>::dummy_controller;

/* =============================================CSV Output Files=======================================================*/
        std::string path = "/home/daniel/Repos/OptimisationBasedControl/data/";

        const std::string mode = "vid_ddp";
        std::fstream cost_mpc(path + name + "_cost_mpc_" + mode + std::to_string(int(params.importance)) + std::to_string(seed) +  ".csv",
                              std::fstream::out | std::fstream::trunc);
        std::fstream ctrl_data(path + name + "_ctrl_" + mode + std::to_string(int(params.importance)) + std::to_string(seed) +  ".csv",
                               std::fstream::out | std::fstream::trunc);
        std::fstream pos_data(path + name + "_pos_" + mode + std::to_string(int(params.importance)) + std::to_string(seed) +  ".csv",
                              std::fstream::out | std::fstream::trunc);
        std::fstream vel_data(path + name + "_vel_" + mode + std::to_string(int(params.importance)) + std::to_string(seed) +  ".csv",
                              std::fstream::out | std::fstream::trunc);

        double cost;
        GenericBuffer<PosVector> pos_bt{d->qpos};   DataBuffer<GenericBuffer<PosVector>> pos_buff;
        GenericBuffer<VelVector> vel_bt{d->qvel};   DataBuffer<GenericBuffer<VelVector>> vel_buff;
        GenericBuffer<CtrlVector> ctrl_bt{d->ctrl}; DataBuffer<GenericBuffer<CtrlVector>> ctrl_buff;
        GenericBuffer<Eigen::Matrix<double, 1, 1>> cost_bt{&cost}; DataBuffer<GenericBuffer<Eigen::Matrix<double, 1, 1>>> cost_buff;

        pos_buff.add_buffer_and_file({&pos_bt, &pos_data});
        vel_buff.add_buffer_and_file({&vel_bt, &vel_data});
        ctrl_buff.add_buffer_and_file({&ctrl_bt, &ctrl_data});
        cost_buff.add_buffer_and_file({&cost_bt, &ctrl_data});

        printf("Connecting to viewer server…\n");
        Buffer<RawType<CtrlVector>::type> ilqr_buffer{};
        Buffer<RawType<CtrlVector>::type> pi_buffer{};
        ZMQUBuffer<RawType<CtrlVector>::type> zmq_buffer(ZMQ_PUSH, "tcp://localhost:5555");
        zmq_buffer.push_buffer(&ilqr_buffer);
        zmq_buffer.push_buffer(&pi_buffer);

        StateVector temp_state = StateVector::Zero();
        CtrlVector temp_ctrl = CtrlVector::Zero();
        mj_step(m, d);

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
                ilqr.control(d);
                pi.control(d, ilqr._u_traj_cp, ilqr._covariance);
                ilqr._u_traj = pi.m_control;
                ilqr_buffer.update(ilqr._cached_control.data(), true);
                pi_buffer.update(pi._cached_control.data(), false);
                zmq_buffer.send_buffers();
                MujocoUtils::fill_state_vector(d, temp_state, m);
                MujocoUtils::fill_ctrl_vector(d, temp_ctrl, m);
                cost = running_cost(temp_state, temp_ctrl, d , m);
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

            if (save_data) {
                pos_buff.save_buffer(); vel_buff.save_buffer(); ctrl_buff.save_buffer(); cost_buff.save_buffer();
                std::cout << "Saved!" << std::endl;
                save_data = false;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                break;
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

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif
    return 1;
}
