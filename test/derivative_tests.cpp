
#include <mujoco/mujoco.h>
#include <fstream>
#include <GLFW/glfw3.h>
#include <thread>

#include "gtest/gtest.h"
#include "../src/utilities/buffer_utils.h"
#include "../src/controller/ilqr.h"
#include "../src/controller/controller.h"
#include "../src/controller/par_mppi_ddp.h"
#include "../src/utilities/fast_derivatives.h"

#include "Eigen/Core"


class DerivativeTests : public testing::Test {

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


TEST_F(DerivativeTests, CP_CTRL_Deriv)
{
    d->qpos[0] = 0; d->qpos[1] = M_PI; d->qvel[0] = 0; d->qvel[1] = 0;

    MJDataEig eig_d(m);
    eig_d.set_state(PosVector::Zero(), VelVector::Zero());

    MJDerivativeParams<void(const mjModel*, mjData*), double> deriv_params{
        eig_d.m_ctrl, mj_step, 1e-6, state_size, n_ctrl
    };

    {
        TimeBench timer("Deriv Comp");
        MJDerivative<void(const mjModel*, mjData*), double> deriv_mj(deriv_params, m);
        std::cout << deriv_mj(eig_d) << std::endl;
    }
}