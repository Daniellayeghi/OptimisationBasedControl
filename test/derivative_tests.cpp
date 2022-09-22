
#include <mujoco/mujoco.h>
#include <fstream>
#include <GLFW/glfw3.h>
#include <thread>

#include "gtest/gtest.h"
#include "../src/controller/ilqr.h"
#include "../src/controller/controller.h"
#include "../src/utilities/fast_derivatives.h"

#include "Eigen/Core"


class DerivativeTests : public testing::Test {

public:
    void SetUp()
    {
        mj_activate(MUJ_KEY_PATH);
        char error[1000] = "Could not load binary model";
        m = mj_loadXML(
                "/home/daniel/Repos/OptimisationBasedControl/models/2link.xml",0, error, 1000
        );

        if (!m)
            mju_error_s("Load model error: %s", error);

        m->opt.o_margin = 0.01;
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


TEST_F(DerivativeTests, CP_CTRL_Jac)
{
    d->qpos[0] = 0; d->qvel[0] = 0; d->qacc[0] = 0;
    d->qpos[1] = 0; d->qvel[1] = 0; d->qacc[1] = 0;    FiniteDifference fd(m);

    MjDerivativeParams params{1e-6, Wrt::Ctrl, Mode::Fwd, Order::First};
    MjDerivative deriv_mj(m, d, params);

    {
        TimeBench timer("Deriv Comp New");
        auto& res = deriv_mj.output();
        std::cout << res << std::endl;
    }

    d->qpos[0] = 0; d->qvel[0] = 0; d->qacc[0] = 0;
    d->qpos[1] = 0; d->qvel[1] = 0; d->qacc[1] = 0;    
    {
        TimeBench timer("Deriv Comp Original");
        fd.f_x_f_u(d);
        const auto res = fd.f_x();
        std::cout << res << std::endl;
    }
}


TEST_F(DerivativeTests, CP_CTRL_Hess)
{
    d->qpos[0] = 10; d->qvel[0] = 0; d->qacc[0] = 0;
    d->qpos[1] = 10; d->qvel[1] = 0; d->qacc[1] = 0;
    d->ctrl[0] = 0; d->ctrl[1] = 0;
    FiniteDifference fd(m);

    MjDerivativeParams params{1e-6, Wrt::Ctrl, Mode::Fwd, Order::Second};
    MjDerivative deriv_mj(m, d, params);

    {
        TimeBench timer("Deriv Comp New");
        auto& res = deriv_mj.output();
        std::cout << res << std::endl;
    }

    d->qpos[0] = 0; d->qvel[0] = 0; d->qacc[0] = 0;
    d->qpos[1] = 0; d->qvel[1] = 0; d->qacc[1] = 0;
    {
        TimeBench timer("Deriv Comp Original");
        fd.f_x_f_u(d);
        const auto res = fd.f_x();
        std::cout << res << std::endl;
    }
}


TEST_F(DerivativeTests, JOINT_ID) {
    d->qpos[0] = 0;
    d->qpos[1] = M_PI;
    d->qvel[0] = 0;
    d->qvel[1] = 0;

    // In general finite differences the perturbation happens with respect to all the states in order.
    // in this case states being, pos, vel, acc. so a call to finite diff w.r.t to some parameter implicitly traverses
    // the states in order.

    // In that sense the type perturbation is important depending on the joint type.
    // For mj_derivative:
    for(int i = 0; i < m->nv; ++i)
    {
        printf("Joint id: %i, Body id: %i, Start add qpos: %i Start add qvel %i \n",
               m->dof_jntid[i], m->dof_bodyid[i], m->jnt_qposadr[i], m->jnt_dofadr[i]
        );
    }
}