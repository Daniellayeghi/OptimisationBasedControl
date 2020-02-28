
#ifndef OPTCONTROL_MUJOCO_COST_FUNCTION_H
#define OPTCONTROL_MUJOCO_COST_FUNCTION_H

#include <functional>
#include <mujoco.h>
#include "eigen3/Eigen/Core"
using namespace Eigen;
#include "autodiff/forward.hpp"
#include "autodiff/forward/eigen.hpp"
using namespace autodiff;

class CostFunction
{
public:
    using OBJ_FUNC_PTR = dual(*)(Vector4dual &x, Vector2dual &u);
    explicit CostFunction(OBJ_FUNC_PTR running, OBJ_FUNC_PTR terminal, mjData *state);
    void fill_data();

    dual Lf();
    VectorXd Lf_x();
    VectorXd Lf_xx();
    Vector4d L_x();
    Vector2d L_u();
    VectorXd L_xx();
    VectorXd L_ux();

private:
    void fill_control();
    void fill_state();

    OBJ_FUNC_PTR _running_cost;
    OBJ_FUNC_PTR _terminal_cost;
    Vector2dual _u;
    Vector4dual _x;
    mjData* _state;
};

#endif //OPTCONTROL_MUJOCO_COST_FUNCTION_H
