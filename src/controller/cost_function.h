
#ifndef OPTCONTROL_MUJOCO_COST_FUNCTION_H
#define OPTCONTROL_MUJOCO_COST_FUNCTION_H

#include "mujoco.h"
#include "Eigen/Core"
#include "../utilities/internal_types.h"
#include "../parameters/simulation_params.h"
#include <functional>

using namespace InternalTypes;
using namespace SimulationParameters;
template<int state_size, int ctrl_size>
class CostFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // TODO: Pass cost as functor
    CostFunction(const StateVector& x_desired,
                 const CtrlVector& u_desired,
                 const StateMatrix& x_gain,
                 const CtrlMatrix& u_gain,
                 const StateMatrix& x_terminal_gain,
                 const mjModel* model);

    StateVector L_x(const mjData *d);
    CtrlVector L_u(const mjData *d);
    StateMatrix L_xx(const mjData *d);
    CtrlMatrix L_uu(const mjData *d);
    CtrlStateMatrix L_ux(const mjData *d);
    StateVector Lf_x(const mjData *d);
    StateMatrix Lf_xx();

    mjtNum running_cost(const mjData *d);
    mjtNum terminal_cost(const mjData *d);
    mjtNum trajectory_running_cost(std::vector<StateVector> & x_trajectory, std::vector<CtrlVector> & u_trajectory);

    const CtrlMatrix& _u_gain;
    const StateMatrix& _x_gain;
    const StateMatrix& _x_terminal_gain;

private:
    void update_errors(const mjData *d);
    void update_errors(StateVector &state, CtrlVector &ctrl);

    CtrlVector _u;
    StateVector _x;
    CtrlVector _u_error;
    StateVector _x_error;
    const CtrlVector& _u_desired;
    const StateVector& _x_desired;
    const mjModel* _m;
};

#endif //OPTCONTROL_MUJOCO_COST_FUNCTION_H
