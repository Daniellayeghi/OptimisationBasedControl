#ifndef OPTCONTROL_MUJOCO_MPPI_H
#define OPTCONTROL_MUJOCO_MPPI_H

#include <vector>
#include"mujoco.h"
#include "../utilities/internal_types.h"

using namespace InternalTypes;
class MPPI
{
public:
    MPPI(const mjModel* m);
    void control(const mjData* d);
    InternalTypes::Mat2x1 _cached_control;

private:
    double q_cost(Mat4x1 state);
    double delta_q_cost(Mat4x1& state, Mat2x1& du, Mat2x1& u);
    Mat2x1 total_entropy(const std::vector<Mat2x1>& delta_control_samples,
                         const std::vector<double>& d_cost_to_go_samples) const;

    const int _k_samples  = 50;
    const int _sim_time   = 10;
    const float _variance = 10;
    const float _lambda   = 10;

    Mat2x2 _R_control_cost;
    Mat4x4 _Q_state_cost;
    std::vector<mjtNum> _cost;
    std::vector<Mat4x1> _state;
    std::vector<Mat2x1> _control;
    std::array<std::vector<double>, 10> _delta_cost_to_go;
    std::array<std::vector<Mat2x1>, 10> _delta_control;
    const mjModel* _m;
    mjData*  _d_cp    = nullptr;
};


#endif //OPTCONTROL_MUJOCO_MPPI_H
