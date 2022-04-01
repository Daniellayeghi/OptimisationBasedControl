
#include "mujoco.h"
#include "cstring"
#include "glfw3.h"
#include "../../src/controller/controller.h"
#include "../../src/utilities/buffer.h"
#include "../../src/utilities/zmq_utils.h"
#include "../../src/controller/par_mppi_ddp.h"

// for sleep timers
#include<chrono>
#include<thread>
#include<iostream>


using namespace std;
using namespace std::chrono;
using namespace SimulationParameters;
constexpr const bool show_gui =  false;
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
    // activate software
    mj_activate(MUJ_KEY_PATH);

    char error[1000] = "Could not load binary model";


    std::string model_path = "../../../models/", name = "arm_hand";
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
    StateVector x_desired; x_desired << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.207, -.245, 0.03, 0.707, -0.707, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    CtrlVector u_desired; u_desired << 0, 0, 0, 0, 0, 0, 0, 0, 0;

    StateVector x_terminal_diag; x_terminal_diag << 0, 0, 0, 0, 0, 0, 0, 0, 0, 10000, 10000, 10, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, .010, 0, 0, 0;
    StateMatrix x_terminal_gain; x_terminal_gain = x_terminal_diag.asDiagonal();

    StateVector x_running_diag; x_running_diag << 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0, .1, .1, .1, 0, 0, 0;
    StateMatrix x_running_gain; x_running_gain = x_running_diag.asDiagonal();

    CtrlVector u_gain_vector; u_gain_vector <<  10, 10, 5, 5, 5, 5, 5, 5, 5;
    CtrlMatrix u_gain = u_gain_vector.asDiagonal() * .5;

    CtrlMatrix du_gain;
    du_gain.setIdentity();
    du_gain *= 0;

    StateVector x_initial; x_initial << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.207, 0, 0.09, 0.707, -0.707, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
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
        ctrl_var.diagonal()[elem] = 0.0325;
        ddp_var.diagonal()[elem] = 0.001;
    }

    StateMatrix t_state_reg = x_terminal_gain;
    StateMatrix r_state_reg = x_running_gain;
    CtrlMatrix control_reg = u_gain;

    const auto max_ctrl_auth = [](const mjData* data=nullptr, const mjModel *model=nullptr){
        std::array<int, 4> body_list {{0, 1, 2, 3}};

        auto iteration = 0;
        if(data and model)
            for(auto i = 0; i < data->ncon; ++i)
            {
                // Contact with world body (0) is always there so ignore that.
                auto elem_1 = std::find(body_list.begin(), body_list.end(), model->geom_bodyid[data->contact[i].geom1]);
                auto elem_2 = std::find(body_list.begin(), body_list.end(), model->geom_bodyid[data->contact[i].geom2]);
                bool world_contact = elem_1 == body_list.begin() or elem_2 == body_list.begin();
                bool check_1 = elem_1 != body_list.end(), check_2 = elem_2 != body_list.end();
                if (check_1 != check_2 and not world_contact)
                    ++iteration;
            }
        return (iteration == 0)?1:1.0/(iteration+1);
    };


    const auto collision_cost = [](const mjData* data=nullptr, const mjModel *model=nullptr){
        std::array<int, 2> body_list {{0, 6}};
        static PosVector x_desired; x_desired << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.207, -.245, 0.03, 0.707, -0.707, 0, 0;
        static const Eigen::Map<Eigen::Vector3d> m_pos(d->qpos+9);
        if(m_pos.isApprox(x_desired.block(9, 0, 3, 1), 0.1)) {return true;}

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

    const auto running_cost =
            [&](const StateVector &state_vector, const CtrlVector &ctrl_vector, const mjData* data=nullptr, const mjModel *model=nullptr){
        StateVector state_error  = x_desired - state_vector;
        CtrlVector ctrl_error = u_desired - ctrl_vector;
        auto error = (state_error.transpose() * r_state_reg * state_error + ctrl_error.transpose() * control_reg * ctrl_error).eval();
        auto col_cost = not collision_cost(data, model) * 1e5;
        return (error(0, 0) + col_cost);
    };

    const auto terminal_cost =
            [&](const StateVector &state_vector, const mjData* data=nullptr, const mjModel *model=nullptr) {
        StateVector state_error = x_desired - state_vector;
        return (state_error.transpose() * t_state_reg * state_error)(0, 0);
    };

    const auto importance_reg =[&](const mjData* data=nullptr, const mjModel *model=nullptr){

        return 1;
    };

    std::array<int, 1> seeds {{1}};
    for (const auto seed : seeds) {
        auto iteration = 0;
        std::copy(x_initial.data(), x_initial.data()+n_jpos, d->qpos);
        std::copy(x_initial.data()+n_jpos, x_initial.data()+state_size, d->qvel);

        // initial position
        FiniteDifference fd(m);
        CostFunction cost_func(x_desired, u_desired, x_running_gain, u_gain, du_gain, x_terminal_gain, m);
        ILQRParams ilqr_params{1e-6, 1.6, 1.6, 0, 75, 1, true};
        ILQR ilqr(fd, cost_func, ilqr_params, m, d, nullptr);

        MPPIDDPParamsPar params{
                800, 75, 0.25, 1, 1, 1, 675,
                ctrl_mean, ddp_var, ctrl_var, {ilqr.m_u_traj_cp, ilqr._covariance},
                1, importance_reg, true};

        QRCostDDPPar qrcost(params, running_cost, terminal_cost);
        MPPIDDPPar pi(m, qrcost, params);

        // install control callback
        using ControlType = MPPIDDPPar;
        MyController<ControlType, n_jpos + n_jvel, n_ctrl> control(m, d, pi, true);
        MyController<ControlType, n_jpos + n_jvel, n_ctrl>::set_instance(&control);
        mjcb_control = MyController<ControlType, n_jpos + n_jvel, n_ctrl>::dummy_controller;
/* =============================================CSV Output Files=======================================================*/
        std::string path = "/home/daniel/Repos/OptimisationBasedControl/data/";

        const std::string mode = path + "Test1" + name + std::to_string(int(params.importance)) + std::to_string(seed);
        std::fstream cost_mpc(mode + "_cost_mpc_" + ".csv",std::fstream::out | std::fstream::trunc);
        std::fstream ctrl_data(mode +"_ctrl_" + ".csv",std::fstream::out | std::fstream::trunc);
        std::fstream pos_data(mode +"_pos_" + ".csv",std::fstream::out | std::fstream::trunc);
        std::fstream vel_data(mode +"_vel_" + ".csv",std::fstream::out | std::fstream::trunc);

        double cost;
        GenericBuffer<PosVector> pos_bt{d->qpos};
        DummyBuffer<GenericBuffer<PosVector>> pos_buff;
        GenericBuffer<VelVector> vel_bt{d->qvel};
        DummyBuffer<GenericBuffer<VelVector>> vel_buff;
        GenericBuffer<CtrlVector> ctrl_bt{d->ctrl};
        DataBuffer<GenericBuffer<CtrlVector>> ctrl_buff;
        GenericBuffer<Eigen::Matrix<double, 1, 1>> cost_bt{&cost};
        DummyBuffer<GenericBuffer<Eigen::Matrix<double, 1, 1>>> cost_buff;

        pos_buff.add_buffer_and_file({&pos_bt, &pos_data});
        vel_buff.add_buffer_and_file({&vel_bt, &vel_data});
        ctrl_buff.add_buffer_and_file({&ctrl_bt, &ctrl_data});
        cost_buff.add_buffer_and_file({&cost_bt, &ctrl_data});

        /* Use REQ because we want to make sure we recieved all info*/
        printf("Connecting to viewer server…\n");
        ZMQUBuffer<RawTypeEig<CtrlVector>::type> zmq_buffer(ZMQ_PUSH, "tcp://localhost:5555");
        std::vector<BufferParams<scalar_type, char>> buffer_params{{ilqr.cached_control.data(), ilqr.cached_control.data() + n_ctrl, 'q'},
                                                                   {pi.cached_control.data(), pi.cached_control.data()+n_ctrl,       'i'}};
        SimpleBuffer<RawTypeEig<CtrlVector>::scalar, char> simp_buff(buffer_params);

        StateVector temp_state = StateVector::Zero();
        Eigen::Map<CtrlVector> mapped_ctrl = Eigen::Map<CtrlVector>(d->ctrl);
        mj_step(m, d);
/* ==================================================Simulation=======================================================*/

        // use the first while condition if you want to simulate for a period.
        while (not glfwWindowShouldClose(window)) {
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
                ilqr.m_u_traj = pi.m_u_traj;;
                zmq_buffer.send_buffer(simp_buff.get_buffer(), simp_buff.get_buffer_size());
                zmq_buffer.buffer_wait_for_res();
                pos_buff.push_buffer(); vel_buff.push_buffer(); ctrl_buff.push_buffer(); cost_buff.push_buffer();
                mjcb_control = MyController<ControlType, n_jpos + n_jvel, n_ctrl>::callback_wrapper;
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

            if (save_data or iteration  == 75) {
                pos_buff.save_buffer(); vel_buff.save_buffer();
                ctrl_buff.save_buffer(); cost_buff.save_buffer();
                save_data = false;
                std::cout << "Saved!" << std::endl;
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