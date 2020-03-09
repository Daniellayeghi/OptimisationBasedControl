
#ifndef OPTCONTROL_MUJOCO_COST_FUNCTION_H
#define OPTCONTROL_MUJOCO_COST_FUNCTION_H

#include <functional>
#include <mujoco.h>
#include "eigen3/Eigen/Core"
using namespace Eigen;
#include "../utilities/internal_types.h"

class CostFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // TODO: Pass cost as functor
    explicit CostFunction();
    void derivatives(mjData* d);

//    VectorXd Lf_x();
//    VectorXd Lf_xx();
    Eigen::Ref<Block<Eigen::Matrix<double, 12, 1>, 6, 1>> L_x();
    Eigen::Ref<Block<Eigen::Matrix<double, 12, 1>, 2, 1>> L_u();
    Eigen::Ref<Block<Eigen::Matrix<double, 12, 12, 0, 12, 12>, 6, 6>> L_xx();
    Eigen::Ref<Block<Eigen::Matrix<double, 12, 12, 0, 12, 12>, 2, 2>> L_uu();
    Eigen::Ref<Block<Eigen::Matrix<double, 12, 12, 0, 12, 12>, 2, 4>> L_ux();


private:
    Eigen::Matrix<double, 6, 1> _u;
    Eigen::Matrix<double, 6, 1> _x;
    Eigen::Matrix<double, 12, 1> _gradient;
    Eigen::Matrix<double, 12, 12> _hessian;

    AutoDiffTypes::AVector4x1 _Ax;
    AutoDiffTypes::AVector4x1 _Au;
    AutoDiffTypes::outer_active_scalar4x1 _Ac;
};

#endif //OPTCONTROL_MUJOCO_COST_FUNCTION_H
