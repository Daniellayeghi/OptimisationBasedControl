
#ifndef OPTCONTROL_MUJOCO_COST_FUNCTION_H
#define OPTCONTROL_MUJOCO_COST_FUNCTION_H

#include <functional>
#include <mujoco.h>
#include "eigen3/Eigen/Core"
using namespace Eigen;
#include "../utilities/internal_types.h"

using namespace InternalTypes;

class CostFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // TODO: Pass cost as functor
    CostFunction(const mjData* d,
                 const Mat4x1& x_desired,
                 const Mat2x1& u_desired,
                 const Mat4x4& x_gain,
                 const Mat2x2& u_gain,
                 const Mat4x4& x_terminal_gain);

    Eigen::Matrix<mjtNum, 4, 1> L_x();
    Eigen::Matrix<mjtNum, 2, 1> L_u();
    Eigen::Matrix<mjtNum, 4, 4> L_xx();
    Eigen::Matrix<mjtNum, 2, 2> L_uu();
    Eigen::Matrix<mjtNum, 2, 4> L_ux();
    Eigen::Matrix<mjtNum, 4, 1> Lf_x();
    Eigen::Matrix<mjtNum, 4, 4> Lf_xx();

    mjtNum running_cost();
    mjtNum terminal_cost();

    template<int x_rows, int u_rows, int cols>
    mjtNum trajectory_running_cost(const std::vector<Eigen::Matrix<mjtNum, x_rows, cols>> & x_trajectory,
                                   const std::vector<Eigen::Matrix<mjtNum, u_rows, cols>> & u_trajectory);

private:
    void update_errors();
    template<int x_rows, int u_rows, int cols>
    void update_errors(const Eigen::Matrix<mjtNum, x_rows, cols> &state,
                       const Eigen::Matrix<mjtNum, u_rows, cols> &ctrl);

    Eigen::Matrix<double, 2, 2> _u_gain;
    Eigen::Matrix<double, 4, 4> _x_gain;
    Eigen::Matrix<double, 4, 4> _x_terminal_gain;
    Eigen::Matrix<double, 4, 1> _x_error;
    Eigen::Matrix<double, 2, 1> _u_error;
    Eigen::Matrix<double, 2, 1> _u_desired;
    Eigen::Matrix<double, 4, 1> _x_desired;

public:
    const mjData* _d;
};

#endif //OPTCONTROL_MUJOCO_COST_FUNCTION_H
