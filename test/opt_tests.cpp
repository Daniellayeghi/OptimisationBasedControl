

#include <mujoco/mujoco.h>
#include <fstream>
#include <GLFW/glfw3.h>
#include <thread>

#include "gtest/gtest.h"
#include "../src/utilities/buffer_utils.h"
#include "../src/controller/ilqr.h"
#include "../src/controller/controller.h"
#include "../src/controller/par_mppi_ddp.h"

#include "Eigen/Core"


class SolverTests : public testing::Test {

public:
    void SetUp()
    {
        mj_activate(MUJ_KEY_PATH);
        char error[1000] = "Could not load binary model";
        m = mj_loadXML(
                "/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml",0, error, 1000
        );

        if (!m)
            mju_error_s("Load model error: %s", error);

        m->opt.timestep = 0.01;
        d = mj_makeData(m);

        if( !glfwInit() )
            mju_error("Could not initialize GLFW");

        window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        // initialize visualization data structures
        mjv_defaultCamera(&cam);
        mjv_defaultOption(&opt);
        mjv_defaultScene(&scn);
        mjr_defaultContext(&con);
        mjv_makeScene(m, &scn, 2000);                // space for 2000 objects
        mjr_makeContext(m, &con, mjFONTSCALE_150);   // model-specific context
    }

    void TearDown()
    {
        mjcb_control = MyController<ILQR, n_jpos + n_jvel, n_ctrl>::dummy_controller;
        mjv_freeScene(&scn);
        mjr_freeContext(&con);
        mj_deleteData(d);
        mj_deleteModel(m);
        mj_deactivate();
        glfwTerminate();

    }

    mjModel* m = nullptr;
    mjData* d   = nullptr;
    std::string save_dir = "/home/daniel/Repos/OptimisationBasedControl/test/data/";
    mjvCamera cam;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;
    GLFWwindow* window;
};



TEST_F(SolverTests, ILQR_solve_test)
{
    // setup cost params
    StateVector x_desired; x_desired << 0, 0, 0, 0;
    CtrlVector u_desired; u_desired << 0;

    StateVector x_terminal_gain_vec; x_terminal_gain_vec << 100000, 50000, 500, 500;
    StateMatrix x_terminal_gain; x_terminal_gain = x_terminal_gain_vec.asDiagonal();
    StateVector x_gain_vec; x_gain_vec << 2, 2, 0, 0;
    StateMatrix x_gain = x_gain_vec.asDiagonal();

    CtrlMatrix u_gain;
    u_gain.setIdentity();
    u_gain *= 0.01;

    CtrlMatrix du_gain;
    du_gain.setIdentity();
    du_gain *= 0;

    // initial position
    d->qpos[0] = 0; d->qpos[1] = M_PI; d->qvel[0] = 0; d->qvel[1] = 0;

    FiniteDifference fd(m);
    CostFunction cost_func(x_desired, u_desired, x_gain, u_gain, du_gain, x_terminal_gain, m);
    ILQRParams params {1e-6, 1.6, 1.6, 0, 75, 1,  false};
    ILQR ilqr(fd, cost_func, params, m, d, nullptr);

    // install control callback
    MyController<ILQR, n_jpos + n_jvel, n_ctrl> control(m, d, ilqr);
    MyController<ILQR, n_jpos + n_jvel, n_ctrl>::set_instance(&control);
    mjcb_control = MyController<ILQR, n_jpos + n_jvel, n_ctrl>::dummy_controller;

    Eigen::Map<PosVector> pos_map(d->qpos);
    Eigen::Map<CtrlVector> ctrl_map(d->ctrl);
    std::vector<CtrlVector> u_vec;

    {
        TimeBench timer("ILQR_solve_test");
        do {
            mjtNum simstart = d->time;
            while (d->time - simstart < 1.0 / 60.0) {
                mjcb_control = MyController<ILQR, n_jpos + n_jvel, n_ctrl>::dummy_controller;
                ilqr.control(d);
                mjcb_control = MyController<ILQR, n_jpos + n_jvel, n_ctrl>::callback_wrapper;
                mj_step(m, d);
                u_vec.emplace_back(ctrl_map);
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
        } while ((pos_map - x_desired.block<2, 1>(0, 0)).norm() > 1e-3);
    }

    std::vector<CtrlVector> u_des_vec;
    BufferUtilities::read_csv_file(save_dir + "datacartpole_ctrl_vec.csv", u_des_vec);

    for (int i = 0; i < u_des_vec.size(); ++i)
        ASSERT_TRUE((u_des_vec[i]-u_vec[i]).norm() < 1e-8);
}


TEST_F(SolverTests, MPPI_ILQR_solve_test)
{
    // setup cost params
    StateVector x_desired; x_desired << 0, 0, 0, 0;
    CtrlVector u_desired; u_desired << 0;

    StateVector x_terminal_gain_vec; x_terminal_gain_vec << 100000, 50000, 500, 500;
    StateMatrix x_terminal_gain; x_terminal_gain = x_terminal_gain_vec.asDiagonal();
    StateVector x_gain_vec; x_gain_vec << 2, 2, 0, 0;
    StateMatrix x_gain = x_gain_vec.asDiagonal();

    CtrlMatrix u_gain;
    u_gain.setIdentity();
    u_gain *= 0.01;

    CtrlMatrix du_gain;
    du_gain.setIdentity();
    du_gain *= 0;

    CtrlMatrix ddp_var; ddp_var.setIdentity();
    CtrlMatrix ctrl_var; ctrl_var.setIdentity();
    CtrlVector ctrl_mean; ctrl_mean.setZero();
    for(auto elem = 0; elem < n_ctrl; ++elem)
    {
        ctrl_var.diagonal()[elem] = 0.25;
        ddp_var.diagonal()[elem] = 0.001;
    }

    StateMatrix t_state_reg = x_terminal_gain;
    StateMatrix r_state_reg = x_gain;

    CtrlMatrix control_reg;
    control_reg = u_gain * 0;

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

    // initial position
    d->qpos[0] = 0; d->qpos[1] = M_PI; d->qvel[0] = 0; d->qvel[1] = 0;

    FiniteDifference fd(m);
    CostFunction cost_func(x_desired, u_desired, x_gain, u_gain, du_gain, x_terminal_gain, m);
    ILQRParams ilqr_params{1e-6, 1.6, 1.6, 0, 75, 1};
    ILQR ilqr(fd, cost_func, ilqr_params, m, d, nullptr);

    // To show difference in sampling try 3 samples
    MPPIDDPParamsPar params{
            200, 75, 0.1, 1, 1, 1, 1000,ctrl_mean,
            ddp_var, ctrl_var, {ilqr.m_u_traj_cp, ilqr._covariance}, 1
    };
    QRCostDDPPar qrcost(params, running_cost, terminal_cost);
    MPPIDDPPar pi(m, qrcost, params);

    // install control callback
    using ControlType = MPPIDDPPar;
    MyController<ControlType, n_jpos + n_jvel, n_ctrl> control(m, d, pi);
    MyController<ControlType, n_jpos + n_jvel, n_ctrl>::set_instance(&control);

    Eigen::Map<PosVector> pos_map(d->qpos);
    Eigen::Map<CtrlVector> ctrl_map(d->ctrl);
    std::vector<CtrlVector> u_vec;

    {
        TimeBench timer("ILQR_solve_test");
        do {
            mjtNum simstart = d->time;
            while (d->time - simstart < 1.0 / 60.0) {
                mjcb_control = MyController<ControlType, n_jpos + n_jvel, n_ctrl>::dummy_controller;
                ilqr.control(d);
                pi.control(d);
                ilqr.m_u_traj = pi.m_u_traj;
                mjcb_control = MyController<ControlType, n_jpos + n_jvel, n_ctrl>::callback_wrapper;
                mj_step(m, d);
                u_vec.emplace_back(ctrl_map);
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
        } while ((pos_map - x_desired.block<2, 1>(0, 0)).norm() > 1e-2);
    }

    std::vector<CtrlVector> u_des_vec;
    BufferUtilities::read_csv_file(save_dir + "datacartpole_ctrl_vec_mppi_ilqr.csv", u_des_vec);

    for (int i = 0; i < u_des_vec.size(); ++i)
        ASSERT_TRUE((u_des_vec[i]-u_vec[i]).norm() < 1e-8);
}
