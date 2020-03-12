
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

#if 0
    void derivatives(const mjData* d);
    Eigen::Ref<Block<Eigen::Matrix<double, 8, 1>, 4, 1>> L_x();
    Eigen::Ref<Block<Eigen::Matrix<double, 8, 1>, 2, 1>> L_u();
    Eigen::Ref<Block<Eigen::Matrix<double, 8, 8, 0, 8, 8>, 4, 4>> L_xx();
    Eigen::Ref<Block<Eigen::Matrix<double, 8, 8, 0, 8, 8>, 2, 2>> L_uu();
    Eigen::Ref<Block<Eigen::Matrix<double, 8, 8, 0, 8, 8>, 2, 4>> L_ux();
#endif

private:
    void update_errors();
    mjtNum running_cost();
    mjtNum terminal_cost();

    Eigen::Matrix<double, 2, 2> _u_gain;
    Eigen::Matrix<double, 4, 4> _x_gain;
    Eigen::Matrix<double, 4, 4> _x_terminal_gain;
    Eigen::Matrix<double, 4, 1> _x_error;
    Eigen::Matrix<double, 2, 1> _u_error;
    Eigen::Matrix<double, 2, 1> _u_desired;
    Eigen::Matrix<double, 4, 1> _x_desired;

    const mjData* _d;
#if 0
    Eigen::Matrix<double, 8, 1> _gradient;
    Eigen::Matrix<double, 8, 8> _hessian;
    AutoDiffTypes::AVector4x1 _Ax;
    AutoDiffTypes::AVector4x1 _Au;
    AutoDiffTypes::outer_active_scalar4x1 _Ac;
#endif
};

#endif //OPTCONTROL_MUJOCO_COST_FUNCTION_H
