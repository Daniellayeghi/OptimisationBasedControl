
#include "mujoco.h"
#include "cstring"
#include "glfw3.h"
#include "../../src/controller/controller.h"
#include "../../src/utilities/buffer_utils.h"
#include "../../src/utilities/buffer.h"

#include "../third_party/imgui/imgui.h"
#include "../third_party/imgui/examples/imgui_impl_glfw.h"
#include "../third_party/imgui/examples/imgui_impl_opengl3.h"

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


static void gui_reset(mjData *data, const mjModel *model)
{
    mj_resetData(model, data);
    data->qpos[0] = 0; data->qpos[1] = 0; data->qpos[2] = 0; data->qpos[3] = -1.0; data->qpos[4] = 0; data->qpos[5] = 0; data->qpos[6] = 0;
    data->qvel[0] = 0; data->qvel[1] = 0; data->qvel[2] = 0; data->qvel[3] = -0.0; data->qvel[4] = 0; data->qvel[5] = 0; data->qvel[6] = 0;
}


template<int square_size>
static void generate_input(char * input, int buff_size, Eigen::Matrix<double, square_size, square_size>& gain, int offset = 0)
{
    ImGui::InputText("Input", input, buff_size);
    if(ImGui::Button("SET"))
    {
        std::stringstream ss (input);
        int iteration = 0 + offset;
        for (double value; ss >> value;)
        {
            if (iteration < square_size)
                gain(iteration, iteration) = value;
            if (ss.peek() == ',')
                ss.ignore();
            if (ss.peek() == ']')
                break;
            ++iteration;
        }
    }
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
        m = mj_loadXML("../../../models/franka_sparse_env.xml", 0, error, 1000);

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
    StateVector x_desired; x_desired << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    CtrlVector u_desired; u_desired << 0, 0, 0, 0, 0, 0, 0, 0, 0;

    StateVector x_terminal_diag; x_terminal_diag << 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 0, 0, 0, 0;
    x_terminal_diag *= 100;
    x_terminal_diag.block<n_jvel, 1>(n_jpos, 0) *= m->opt.timestep;
    StateMatrix x_terminal_gain; x_terminal_gain = x_terminal_diag.asDiagonal();

    StateVector x_running_diag; x_running_diag << 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0,
                                                                                10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0;
    x_running_diag  *= 0;
    x_running_diag.block<n_jvel, 1>(n_jpos, 0) *= m->opt.timestep;
    StateMatrix x_running_gain; x_running_gain = x_running_diag.asDiagonal();

    CtrlMatrix u_gain;
    u_gain.setIdentity();
    u_gain *= 0.001;

    CtrlMatrix du_gain;
    du_gain.setIdentity();
    du_gain *= 0;

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // initial position
    d->qpos[0] = 0; d->qpos[1] = -M_PI/4; d->qpos[2] = 0; d->qpos[3] = -3*M_PI/4; d->qpos[4] = 0; d->qpos[5] = M_PI_2; d->qpos[6] = 0;
    d->qvel[0] = 0; d->qvel[1] = 0; d->qvel[2] = 0; d->qvel[3] = -0.0; d->qvel[4] = 0; d->qvel[5] = 0; d->qvel[6] = 0;

    FiniteDifference fd(m);
    CostFunction cost_func(x_desired, u_desired, x_running_gain, u_gain, du_gain, x_terminal_gain, m);
    ILQRParams params {1e-6, 1.6, 1.6, 0, 20, 1};
    ILQR ilqr(fd, cost_func, params, m, d, nullptr);
    // install control callback
    MyController<ILQR, n_jpos + n_jvel, n_ctrl> control(m, d, ilqr);
    MyController<ILQR, n_jpos + n_jvel, n_ctrl>::set_instance(&control);
    mjcb_control = MyController<ILQR, n_jpos + n_jvel, n_ctrl>::callback_wrapper;

/* ==================================================GUI Setup=======================================================*/

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(nullptr);
    ImGui::StyleColorsDark();

    enum QuadMat {Qp_r = 0, Qv_r, Qp_f, Qv_f, R_r};
    std::array<const char *, 5> gain_names {{"Qp_r", "Qv_r","Qp_f","Qv_f", "R_r"}};

    std::array<std::pair<QuadMat, int>, 5> matrix{{{Qp_r, n_jpos},{Qv_r, n_jvel},{Qp_f, n_jpos},{Qv_f, n_jvel},{R_r, n_ctrl}}};

    static int gain_selection = 0;
    constexpr const int buff_size = 5*n_jpos+n_jpos*2;
    char input[buff_size];

/* ============================================CSV Output Files=======================================================*/
    std::string path = "/home/daniel/Repos/OptimisationBasedControl/data/";

    std::fstream cost_mpc(path + ("franka_ilqr_cost_mpc.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream ctrl_data(path + ("franka_ilqr_ctrl.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream pos_data(path + ("franka_ilqr_pos.csv"), std::fstream::out | std::fstream::trunc);
    std::fstream vel_data(path + ("franka_ilqr_vel.csv"), std::fstream::out | std::fstream::trunc);

    double cost;
    GenericBuffer<PosVector> pos_bt{d->qpos};   DataBuffer<GenericBuffer<PosVector>> pos_buff;
    GenericBuffer<VelVector> vel_bt{d->qvel};   DataBuffer<GenericBuffer<VelVector>> vel_buff;
    GenericBuffer<CtrlVector> ctrl_bt{d->ctrl}; DataBuffer<GenericBuffer<CtrlVector>> ctrl_buff;
    GenericBuffer<Eigen::Matrix<double, 1, 1>> cost_bt{&cost}; DataBuffer<GenericBuffer<Eigen::Matrix<double, 1, 1>>> cost_buff;

    pos_buff.add_buffer_and_file({&pos_bt, &pos_data});
    vel_buff.add_buffer_and_file({&vel_bt, &vel_data});
    ctrl_buff.add_buffer_and_file({&ctrl_bt, &ctrl_data});
    cost_buff.add_buffer_and_file({&cost_bt, &ctrl_data});
/* ==================================================Simulation=======================================================*/

    // use the first while condition if you want to simulate for a period.
    while(not glfwWindowShouldClose(window))
    {
        //  advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.

//        if constexpr (show_gui)
//        {
//
//            ImGui_ImplOpenGL3_NewFrame();
//            ImGui_ImplGlfw_NewFrame();
//            ImGui::NewFrame();
//
//            // The same shit as below but with the opposite bool changed
//            for(unsigned int elem = 0; elem < matrix.size(); ++elem)
//            {
//                ImGui::RadioButton(gain_names[elem], &gain_selection, static_cast<int>(elem));
//                if(elem < matrix.size() - 1) ImGui::SameLine();
//            }
//
//            switch(gain_selection)
//            {
//                case Qp_f : generate_input(input, buff_size, cost_func._x_terminal_gain); break;
//                case Qv_f : generate_input(input, buff_size, cost_func._x_terminal_gain, n_jpos); break;
//                case Qp_r : generate_input(input, buff_size, cost_func._x_gain); break;
//                case Qv_r : generate_input(input, buff_size, cost_func._x_gain, n_jpos); break;
//                case R_r : generate_input(input, buff_size, cost_func._u_gain); break;
//                default: break;
//            }
//
//            if(ImGui::Button("Reset"))
//                gui_reset(d, m);
//        }

        mjtNum simstart = d->time;

        while( d->time - simstart < 1.0/60.0 )
        {
            mjcb_control = MyController<ILQR, n_jpos + n_jvel, n_ctrl>::dummy_controller;
            ilqr.control(d);
            pos_buff.push_buffer(); vel_buff.push_buffer(); ctrl_buff.push_buffer(); cost_buff.push_buffer();
            mjcb_control = MyController<ILQR, n_jpos + n_jvel, n_ctrl>::callback_wrapper;
            mj_step(m, d);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        // get framebuffer viewport

        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        if constexpr (show_gui)
        {
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

        if(save_data)
        {
            pos_buff.save_buffer(); vel_buff.save_buffer(); ctrl_buff.save_buffer(); cost_buff.save_buffer();
            save_data = false;
            std::cout << "Saved!" << std::endl;
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
    ImGui_ImplGlfw_Shutdown();
    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif
    return 1;
}
