#include "gtest/gtest.h"
#include "../src/utilities/finite_diff.h"
#include "Eigen/Core"
#include "mujoco.h"

namespace
{
    constexpr const int num_jctrl = 2;
    constexpr const int num_jvel  = 2;
    constexpr const int num_jpos  = 2;
}


class SolverTests : public testing::Test {
public:
    void SetUp() override
    {
        mj_activate(MUJ_KEY_PATH);
        char error[1000] = "Could not load binary model";
        model = mj_loadXML
                (
                "/home/daniel/Repos/OptimisationBasedControl/models/Acrobot.xml",0, error, 1000
                );

        if (!model)
            mju_error_s("Load model error: %s", error);

        model->opt.timestep = 0.01;
        data = mj_makeData(model);

        assert(model->nv == num_jvel);
        assert(model->nq == num_jpos);
        assert(model->nu == num_jctrl);
    }

    void TearDown() override
    {
        mj_deleteData(data);
        mj_deleteModel(model);
        mj_deactivate();
    }

    mjModel* model = nullptr;
    mjData* data   = nullptr;
    float tolerance = 1e-6;
};



TEST_F(SolverTests, Finite_Difference_Jacobian_Stable_Equilibrium)
{
    data->qpos[0] = -1.57; data->qpos[1] = 0;
    data->qvel[0] = 0.0;   data->qvel[1] = 0.0;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_u();

    Eigen::Matrix<double, num_jpos, num_jctrl> result_ref;

    result_ref << 86.96336955, -206.13998602,
                 -206.1399860,  611.16300185;

    ASSERT_TRUE(result_ref.isApprox(result, tolerance));
}


TEST_F(SolverTests, Finite_Difference_Jacobian_Unstable_Equilibrium)
{
    data->qpos[0] = 1.57; data->qpos[1] = 0;
    data->qvel[0] = 0.0;   data->qvel[1] = 0.0;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_x();

    Eigen::Matrix<double, num_jpos, num_jpos + num_jvel> result_ref;

     result_ref << 65.94859829, -72.647543,   -0.8696337,  2.06139986,
                  -81.6380463,   246.89345808, 2.06139986, -6.11163002;

    ASSERT_TRUE(result_ref.isApprox(result, tolerance));
}


TEST_F(SolverTests, Finite_Difference_Jacobian_Unstable_Equilibrium_With_Velocity)
{
    data->qpos[0] = 1.57; data->qpos[1] = 0;
    data->qvel[0] = 1.0;   data->qvel[1] = 0.0;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_x();

    Eigen::Matrix<double, num_jpos, num_jpos + num_jvel> result_ref;

    result_ref << 65.9485982, -70.34187923,  -0.8696337,   2.06139986,
                 -81.6380463,  240.05763549,  2.0613998,  -6.11163002;

    ASSERT_TRUE(result_ref.isApprox(result, tolerance));
}


TEST_F(SolverTests, Finite_Difference_Ctrl_Jacobian_Unstable_Equilibrium)
{
    data->qpos[0] = -1.57; data->qpos[1] = 0;
    data->qvel[0] = 0.0;  data->qvel[1] = 0.0;
    data->ctrl[0] = 0.1;  data->ctrl[1] = 0.3;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_u();

    Eigen::Matrix<double, num_jpos, num_jctrl> result_ref;

    result_ref << 86.96336956, -206.13998601,
                 -206.13998603, 611.16300182;

    ASSERT_TRUE(result_ref.isApprox(result, tolerance));
}


TEST_F(SolverTests, Finite_Difference_Ctrl_Jacobian_Stable_Equilibrium)
{
    // NOTE: the gradient df/du at pos(1.57, 0) and pos(-1.57) is the same as mass matrix is constant.
    // This value is equal to M^-1 as the system is control affine.
    data->qpos[0] = 1.57; data->qpos[1] = 0;
    data->qvel[0] = 0.0;  data->qvel[1] = 0.0;
    data->ctrl[0] = 0;  data->ctrl[1] = 0.3;


    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_u();

    Eigen::Matrix<double, num_jpos, num_jctrl> result_ref;

    result_ref << 86.96336955, -206.13998601,
                 -206.13998601, 611.16300185;

    ASSERT_TRUE(result_ref.isApprox(result, tolerance));
}


TEST_F(SolverTests, Finite_Difference_Ctrl_Jacobian)
{
    // df/du is equal to M^-1 as the system is control affine. i.e. xd = f(x) + g(x)u
    data->qpos[0] = 1;    data->qpos[1] = 1.3;
    data->qvel[0] = 0.0;  data->qvel[1] = 0.0;
    data->ctrl[0] = 0.5;  data->ctrl[1] = 0.3;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_u();

    Eigen::Matrix<double, num_jpos, num_jctrl> result_ref;

    result_ref <<38.86456575, -53.11179446,
                 -53.11179447, 195.10586921;

    ASSERT_TRUE(result_ref.isApprox(result, tolerance));
}
