#ifndef OPTCONTROL_MUJOCO_MPPI_H
#define OPTCONTROL_MUJOCO_MPPI_H

#include <vector>
#include"mujoco.h"
#include "../utilities/internal_types.h"

using namespace InternalTypes;
class MPPI
{
public:
    MPPI(mjModel* m);
    void control(const mjData* d);

private:
    double q_cost(Mat4x1& state);
    double delta_q_cost(Mat4x1& state, Mat2x1& du, Mat2x1& u);

    const int _k_samples  = 100;
    const int _sim_time   = 500;
    const float _variance = 10;
    const float _lambda   = 10;

    Mat2x2 _R_control_cost;
    Mat4x4 _Q_state_cost;
    std::vector<mjtNum> _cost;
    std::vector<Mat4x1> _state;
    std::vector<Mat2x1> _control;
    std::vector<Mat2x1> _avg_control;
    Eigen::Matrix<Mat2x1, 500, 100> _delta_control;

    const mjModel* _m = nullptr;
    mjData*  _d_cp    = nullptr;
};


#endif //OPTCONTROL_MUJOCO_MPPI_H
