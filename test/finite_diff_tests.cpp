#include "gtest/gtest.h"
#include "../src/utilities/finite_diff.h"
#include "Eigen/Core"
#include "mujoco.h"

class SolverTests : public testing::Test {

public:
    void SetUp()
    {
        mj_activate(MUJ_KEY_PATH);
        char error[1000] = "Could not load binary model";
        model = mj_loadXML("/home/daniel/Repos/OptimisationBasedControl/models/Acrobot.xml", 0, error, 1000);

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

    FiniteDifference fd(model);
    auto result = fd.f_x(data);

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

    FiniteDifference fd(model);
    auto result = fd.f_x(data);
    std::cout << result << std::endl;

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

    FiniteDifference fd(model);
    auto result = fd.f_x(data);
    std::cout << result << std::endl;

    Eigen::Matrix<double, 4, 4> result_ref;

    result_ref << 1.00648560e+00, -6.76350083e-03,  9.95772860e-03,  9.95909855e-05,
                 -7.85676581e-03,  1.02322098e+00,  9.95909855e-05,  9.70447566e-03,
                  6.48559603e-01, -6.76350083e-01,  9.95772860e-01,  9.95909855e-03,
                 -7.85676581e-01,  2.32209833e+00,  9.95909855e-03,  9.70447566e-01;

    ASSERT_TRUE(result_ref.isApprox(result, .05));
}