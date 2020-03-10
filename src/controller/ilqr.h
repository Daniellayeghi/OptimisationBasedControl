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
    ~ILQR();

    void backward_pass(mjData* d);

private:
    void forward_simulate(const mjData* d);

    std::vector<mjtNum> _V;
    std::vector<InternalTypes::Mat4x4> _V_xx;
    std::vector<InternalTypes::Mat4x1> _V_x;

    std::vector<InternalTypes::Mat4x4> _F_x;
    std::vector<InternalTypes::Mat4x2> _F_u;

    std::vector<InternalTypes::Mat4x1> _L_x;
    std::vector<InternalTypes::Mat2x1> _L_u;
    std::vector<InternalTypes::Mat4x4> _L_xx;
    std::vector<InternalTypes::Mat2x4> _L_ux;
    std::vector<InternalTypes::Mat2x2> _L_uu;

    std::vector<mjData> _simulated_state;

    InternalTypes::Mat6x1 desired_state;
    int  _simulation_time;
    FiniteDifference& _fd;
    CostFunction& _cf;
    mjData* _d_cp = nullptr;
    const mjModel* _m;
};


#endif //OPTCONTROL_MUJOCO_ILQR_H
