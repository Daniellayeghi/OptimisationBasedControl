#ifndef OPTCONTROL_MUJOCO_ILQR_H
#define OPTCONTROL_MUJOCO_ILQR_H

#include <vector>
#include "mjdata.h"
#include "Eigen/Core"
#include "../utilities/internal_types.h"
#include "../utilities/finite_diff.h"
#include "cost_function.h"


class ILQR
{
public:
    ILQR(FiniteDifference& fd, CostFunction& cf, const mjModel * m, int simulation_time);
    void backward_pass(mjData* d);
private:
    void forward_simulate(const mjData* d);
    void calculate_derivatives();

    std::vector<double> _V;
    std::vector<InternalTypes::Mat6x1> _V_x;
    std::vector<InternalTypes::Mat6x6> _V_xx;
    std::vector<mjData> _simulated_state;

    InternalTypes::Mat6x1 desired_state;
    int _simulation_time;
    FiniteDifference& _fd;
    CostFunction& _cf;
    mjData sim_data;
    mjData* _d_cp;
    const mjModel* _m;
};


#endif //OPTCONTROL_MUJOCO_ILQR_H
