#include "cost_function.h"

CostFunction::CostFunction(OBJ_FUNC_PTR running, OBJ_FUNC_PTR terminal, mjData *state)
{
    _running_cost  = running;
    _terminal_cost = terminal;
    _state         = state;
}


void CostFunction::fill_state()
{
    _x(0, 0) = _state->qpos[0];
    _x(1, 0) = _state->qpos[1];
    _x(2, 0) = _state->qvel[0];
    _x(3, 0) = _state->qvel[1];
}


void CostFunction::fill_control()
{
    _u(0, 0) = _state->ctrl[0];
    _u(1, 0) = _state->ctrl[1];
}


VectorXd CostFunction::L_x()
{
    dual running_cost;
    return gradient(_running_cost, wrt(_x), at(_x, _u), running_cost);
}


VectorXd CostFunction::L_u()
{
    dual running_cost;
    return gradient(_running_cost, wrt(_u), at(_x, _u), running_cost);
}


VectorXd CostFunction::L_xx()
{
    dual running_cost;
    return gradient(_running_cost, wrt<2>(_x), at(_x, _u), running_cost);
}


VectorXd CostFunction::L_ux()
{
    dual running_cost;
    return gradient(_running_cost, wrt(_u, _x), at(_x, _u), running_cost);
}


dual CostFunction::Lf()
{
    return _terminal_cost(_x, _u);
}


VectorXd CostFunction::Lf_x()
{
    dual running_cost;
    return gradient(_terminal_cost, wrt(_x), at(_x, _u), running_cost);
}


VectorXd CostFunction::Lf_xx()
{
    dual running_cost;
    return gradient(_terminal_cost, wrt<2>(_x), at(_x, _u), running_cost);
}