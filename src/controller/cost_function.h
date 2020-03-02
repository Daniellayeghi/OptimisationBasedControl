
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
    explicit CostFunction(mjData *state);

//    double Lf();
    VectorXd Lf_x();
//    VectorXd Lf_xx();
//    Vector4d L_x();
//    Vector2d L_u();
//    VectorXd L_xx();
//    VectorXd L_ux();

private:
    Eigen::Matrix<double, Eigen::Dynamic, 1> _u;
    Eigen::Matrix<double, Eigen::Dynamic, 1> _x;
    Eigen::Matrix<double, Eigen::Dynamic, 1> _gradient;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _hessian;

    AutoDiffTypes::AVector _Ax;
    AutoDiffTypes::AVector _Au;
    AutoDiffTypes::outer_active_scalar _Ac;

    mjData* _state;
};

#endif //OPTCONTROL_MUJOCO_COST_FUNCTION_H
