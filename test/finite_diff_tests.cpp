
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
    void SetUp()
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
    }

    void TearDown()
    {
        mj_deleteData(data);
        mj_deleteModel(model);
        mj_deactivate();
    }

    mjModel* model = nullptr;
    mjData* data   = nullptr;
};



TEST_F(SolverTests, Finite_Difference_Jacobian_Stable_Equilibrium)
{
    data->qpos[0] = -1.57; data->qpos[1] = 0;
    data->qvel[0] = 0.0;   data->qvel[1] = 0.0;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_x();
    std::cout << result << std::endl;
              Eigen::Matrix<double, 4, 4> result_ref;
    result_ref <<  9.93514456e-01,  6.98680871e-03,  9.95772860e-03, 9.95909855e-05,
                    7.85670324e-03,  9.76116416e-01,  9.95909855e-05, 9.70447566e-03,
                    -6.48554438e-01,  6.98680871e-01,  9.95772860e-01, 9.95909855e-03,
                    7.85670324e-01, -2.38835840e+00,  9.95909855e-03, 9.70447566e-01;

    ASSERT_TRUE(result_ref.isApprox(result, .05));
}


TEST_F(SolverTests, Finite_Difference_Jacobian_Unstable_Equilibrium)
{
    data->qpos[0] = 1.57; data->qpos[1] = 0;
    data->qvel[0] = 0.0;   data->qvel[1] = 0.0;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_x();

    Eigen::Matrix<double, 4, 4> result_ref;

    result_ref << 1.00648560e+00, -6.98678782e-03,  9.95772860e-03,  9.95909855e-05,
            -7.85676581e-03,  1.02388352e+00,  9.95909855e-05,  9.70447566e-03,
            6.48559603e-01, -6.98678782e-01,  9.95772860e-01,  9.95909855e-03,
            -7.85676581e-01,  2.38835235e+00,  9.95909855e-03,  9.70447566e-01;

    ASSERT_TRUE(result_ref.isApprox(result, .05));
}


TEST_F(SolverTests, Finite_Difference_Jacobian_Unstable_Equilibrium_With_Velocity)
{
    data->qpos[0] = 1.57; data->qpos[1] = 0;
    data->qvel[0] = 1.0;   data->qvel[1] = 0.0;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_x();

    Eigen::Matrix<double, 4, 4> result_ref;

    result_ref << 1.00648560e+00, -6.76350083e-03,  9.95772860e-03,  9.95909855e-05,
            -7.85676581e-03,  1.02322098e+00,  9.95909855e-05,  9.70447566e-03,
            6.48559603e-01, -6.76350083e-01,  9.95772860e-01,  9.95909855e-03,
            -7.85676581e-01,  2.32209833e+00,  9.95909855e-03,  9.70447566e-01;

    ASSERT_TRUE(result_ref.isApprox(result, .05));
}


TEST_F(SolverTests, Finite_Difference_Ctrl_Jacobian_Unstable_Equilibrium)
{
    data->qpos[0] = -1.57; data->qpos[1] = 0;
    data->qvel[0] = 0.0;  data->qvel[1] = 0.0;
    data->ctrl[0] = 0.1;  data->ctrl[1] = 0.3;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_u();

    Eigen::Matrix<double, 4, 2> result_ref;

    result_ref << 0.00822762, -0.01926687,
            -0.01926687,  0.05722194,
            0.82276195, -1.92668744,
            -1.92668744,  5.72219396;

    ASSERT_TRUE(result_ref.isApprox(result, .00001));
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

    Eigen::Matrix<double, 4, 2> result_ref;

    result_ref << 0.00822762, -0.01926687,
            -0.01926687,  0.05722194,
            0.82276195, -1.92668744,
            -1.92668744,  5.72219396;

    ASSERT_TRUE(result_ref.isApprox(result, .00001));
}


TEST_F(SolverTests, Finite_Difference_Ctrl_Jacobian)
{
    // df/du is equal to M^-1 as the system is control affine.
    data->qpos[0] = 1; data->qpos[1] = 1.3;
    data->qvel[0] = 0.0;  data->qvel[1] = 0.0;
    data->ctrl[0] = 0.5;  data->ctrl[1] = 0.3;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_u();

    Eigen::Matrix<double, 4, 2> result_ref;

    result_ref << 0.00384395, -0.00518951,
                    -0.00518951,  0.01911017,
                    0.38439548, -0.51895131,
                    -0.51895131,  1.91101738;

    ASSERT_TRUE(result_ref.isApprox(result, .00001));
}



TEST_F(SolverTests, Finite_Difference_Ctrl_Jacobian_Stable_Equilibrium_NULL)
{
    // df/du is equal to M^-1 as the system is control affine.
    data->qpos[0] = M_PI_2; data->qpos[1] = 0;
    data->qvel[0] = 0.0;    data->qvel[1] = 0;
    data->ctrl[0] = 0.5;    data->ctrl[1] = 0;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_u();

    std::cout << result << std::endl;

    Eigen::Matrix<double, 4, 2> result_ref;

    result_ref << 0.00384395, -0.00518951,
            -0.00518951,  0.01911017,
            0.38439548, -0.51895131,
            -0.51895131,  1.91101738;

    ASSERT_TRUE(result_ref.isApprox(result, .00001));
}


TEST_F(SolverTests, Finite_Difference_Ctrl_Jacobian_Random)
{
    // df/du is equal to M^-1 as the system is control affine.
    data->qpos[0] = -M_PI_2;   data->qpos[1] = 0;
    data->qvel[0] = 0.0;       data->qvel[1] = 0.0;
    data->ctrl[0] = -0.90672;  data->ctrl[1] = 0.3419;

    FiniteDifference<num_jpos+num_jvel, num_jctrl> fd(model);
    fd.f_x_f_u(data);
    auto result = fd.f_u();

    Eigen::Matrix<double, 4, 2> result_ref;

    result_ref <<   0.204829, -0.228752,
                   -0.457504,  0.684114,
                    20.4829,  -22.8752,
                    -45.7504,   68.4114;

    ASSERT_TRUE(result_ref.isApprox(result, .00001));
}