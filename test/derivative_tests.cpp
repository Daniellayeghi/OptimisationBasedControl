
#include <mujoco/mujoco.h>
#include <fstream>
#include <GLFW/glfw3.h>
#include <thread>

#include "gtest/gtest.h"
#include "../src/controller/ilqr.h"
#include "../src/controller/controller.h"
#include "../src/utilities/fast_derivatives.h"

#include "Eigen/Core"

constexpr const int nu = 1;
constexpr const int np = 1;
constexpr const int nv = 1;
constexpr const int ns = np + nv;
constexpr const int nf = np + nv + nu;

constexpr auto check_size = [](){
    if(full_size != nf)
        FAIL();
};

class DerivativeTests : public testing::Test {

public:
    void SetUp()
    {
        check_size();
        mj_activate(MUJ_KEY_PATH);
        char error[1000] = "Could not load binary model";
        m = mj_loadXML(
            "/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml",0, error, 1000
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
        check_size();
        mjv_freeScene(&scn);
        mjr_freeContext(&con);
        mj_deleteData(d);
        mj_deleteModel(m);
        mj_deactivate();
        glfwTerminate();

    }

    mjModel* m = nullptr;
    mjData* d   = nullptr;
    mjvCamera cam;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;
    GLFWwindow* window;
};


TEST_F(DerivativeTests, CP_CTRL_Jac)
{

    check_size();
    d->qpos[0] = 0.3; d->qvel[0] = 0.6; d->qacc[0] = 0;
//    d->qpos[1] = 0; d->qvel[1] = 0; d->qacc[1] = 0;

    FiniteDifference fd(m);

    MjDerivativeParams params{1e-6, Wrt::Ctrl, Mode::Fwd, Order::First};
    MjDerivative deriv_mj(m, d, params);

    {
        TimeBench timer("Deriv Comp New");
        auto& res = deriv_mj.output();
        std::cout << res << std::endl;
    }

    d->qpos[0] = 0; d->qvel[0] = 0; d->qacc[0] = 0;
//    d->qpos[1] = 0; d->qvel[1] = 0; d->qacc[1] = 0;
    {
        TimeBench timer("Deriv Comp Original");
        fd.f_x_f_u(d);
        const auto res = fd.f_u();
        std::cout << res << std::endl;
    }
}


TEST_F(DerivativeTests, CP_CTRL_Hess)
{
    check_size();
    d->qpos[0] = 10; d->qvel[0] = 0; d->qacc[0] = 0;
    d->ctrl[0] = 0;
    FiniteDifference fd(m);

    MjDerivativeParams params{1e-6, Wrt::Ctrl, Mode::Fwd, Order::Second};
    MjDerivative deriv_mj(m, d, params);

    {
        TimeBench timer("Deriv Comp New");
        auto& res = deriv_mj.output();
        std::cout << res << std::endl;
    }
}


TEST_F(DerivativeTests, JOINT_ID)
{
    check_size();
    d->qpos[0] = 0;
    d->qvel[0] = 0;

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