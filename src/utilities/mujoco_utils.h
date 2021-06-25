#ifndef OPTCONTROL_MUJOCO_MUJOCO_UTILS_H
#define OPTCONTROL_MUJOCO_MUJOCO_UTILS_H

#include "Eigen/Core"
#include "../parameters/simulation_params.h"
#include <random>

namespace MujocoUtils
{
    inline void populate_obstacles(const int start_id,
                                   const int end_id,
                                   const std::array<double, 6> &bounds,
                                   const mjModel *model)
                                   {
        const constexpr int geo_dims = 3;
        using namespace std;

        array<double, geo_dims> random_pos{};
        auto random_iid_array = [](array<double, geo_dims> &result, const array<double, geo_dims * 2> &bounds) {
            random_device r;
            default_random_engine generator(r());
            for (auto dim = 0; dim < geo_dims; ++dim) {
                uniform_real_distribution<double> distribution(bounds[dim * 2], bounds[dim * 2 + 1]);
                result[dim] = distribution(generator);
            }
        };

        // Assumes that all obstacles are contiguous
        auto total_obs = end_id - start_id + 1;
        auto num_geoms = static_cast<int>(total_obs / geo_dims);
        auto bodies = 0;
        for (auto id = 0; id < num_geoms + 1; id++)
        {
            bodies = id * geo_dims;
            random_iid_array(random_pos, bounds);
            model->body_pos[start_id + bodies] = random_pos[0];
            model->body_pos[start_id + bodies + 1] = random_pos[1];
            model->body_pos[start_id + bodies + 2] = random_pos[2];
        }
    }


    inline CtrlVector clamp_control_r(CtrlVector &control, const mjtNum *limits)
    {
        CtrlVector clamped_ctrl;

        for (auto row = 0; row < control.rows(); ++row)
        {
            clamped_ctrl(row, 0) = std::clamp(control(row, 0), limits[row * 2], limits[row * 2 + 1]);
        }

        return clamped_ctrl;
    }


    inline void clamp_control(CtrlVector &control, const mjtNum *limits)
    {
        for (auto row = 0; row < control.rows(); ++row)
        {
            control(row, 0) = std::clamp(control(row, 0), limits[row * 2], limits[row * 2 + 1]);
        }
    }


    inline void set_control_data(const mjData *data, const CtrlVector &ctrl)
    {
        for (auto row = 0; row < ctrl.rows(); ++row)
        {
            data->ctrl[row] = ctrl(row, 0);
        }
    }


    template<typename T>
    void copy_data(const mjModel *model, const mjData *data, T *data_cp)
    {
        data_cp->time = data->time;
        mju_copy(data_cp->qpos, data->qpos, model->nq);
        mju_copy(data_cp->qvel, data->qvel, model->nv);
        mju_copy(data_cp->qacc, data->qacc, model->nv);
        mju_copy(data_cp->qfrc_applied, data->qfrc_applied, model->nv);
        mju_copy(data_cp->xfrc_applied, data->xfrc_applied, 6 * model->nbody);
        mju_copy(data_cp->ctrl, data->ctrl, model->nu);
    }


    inline void fill_state_vector(const mjData *data, StateVector &state)
    {
        for (auto row = 0; row < state.rows() / 2; ++row)
        {
            state(row, 0) = data->qpos[row];
            state(row + state.rows() / 2, 0) = data->qvel[row];
        }
    }


    inline void fill_ctrl_vector(const mjData *data, CtrlVector &ctrl)
    {
//        std::copy(data->ctrl, data->ctrl + ctrl.size(), ctrl.data());
        for(unsigned int row = 0; row < ctrl.rows(); ++row)
        {
            ctrl(row, 0) = data->ctrl[row];
        }
    }


    inline double wrap_to_max(double x, double max)
    {
        return fmod(max + fmod(x, max), max);
    }


    inline double wrap_to_min_max(double x, double min, double max)
    {
        if (x == max) return max;
        return min + wrap_to_max(x - min, max - min);
    }


    inline double wrap_to_2pi(double x)
    {
//        Comment wrapping for cartpole uncomment for acrobot
//        auto angle = (2*M_PI - x) < 1e-7 ?  0 :  x - 2 * M_PI * floor( x / (2 * M_PI));
//        return angle;
        return x;
    }
}

#endif //OPTCONTROL_MUJOCO_MUJOCO_UTILS_H
