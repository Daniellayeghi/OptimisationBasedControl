#ifndef OPTCONTROL_MUJOCO_MUJOCO_UTILS_H
#define OPTCONTROL_MUJOCO_MUJOCO_UTILS_H

#include "Eigen/Core"
#include "../parameters/simulation_params.h"
#include "./math_utils.h"
#include <random>

namespace MujocoUtils
{
    template<typename T_1, typename T_2>
    void copy_data(const mjModel *model, const T_1 *data_src, T_2 *data_cp)
    {
        data_cp->time = data_src->time;
        mju_copy(data_cp->qpos, data_src->qpos, model->nq);
        mju_copy(data_cp->qvel, data_src->qvel, model->nv);
        mju_copy(data_cp->qacc, data_src->qacc, model->nv);
        mju_copy(data_cp->qfrc_applied, data_src->qfrc_applied, model->nv);
        mju_copy(data_cp->xfrc_applied, data_src->xfrc_applied, 6 * model->nbody);
        mju_copy(data_cp->ctrl, data_src->ctrl, model->nu);
    }


    template<typename T_1, typename T_2>
    void copy_data(const mjModel *model, T_1 *data_src, T_2 *data_cp)
    {
        data_cp->time = data_src->time;
        mju_copy(data_cp->qpos, data_src->qpos, model->nq);
        mju_copy(data_cp->qvel, data_src->qvel, model->nv);
        mju_copy(data_cp->qacc, data_src->qacc, model->nv);
        mju_copy(data_cp->qfrc_applied, data_src->qfrc_applied, model->nv);
        mju_copy(data_cp->xfrc_applied, data_src->xfrc_applied, 6 * model->nbody);
        mju_copy(data_cp->ctrl, data_src->ctrl, model->nu);
    }


    template<typename T>
    inline void populate_obstacles(const int start_id,
                                   const int end_id,
                                   const std::vector<T>& poses,
                                   const mjModel *model)
                                   {
        using namespace std;
        constexpr const int geo_dims = 3;

        // Assumes that all obstacles are contiguous
        auto total_obs = end_id - start_id + 1;
        auto num_geoms = static_cast<int>(total_obs / geo_dims);
        auto bodies = 0;
        for (auto id = 0; id < num_geoms + 1; id++)
        {
            bodies = id * 3;
            const auto &pos = poses[id];
            model->body_pos[start_id + bodies]     = pos(0, 0);
            model->body_pos[start_id + bodies + 1] = pos(1, 0);
            model->body_pos[start_id + bodies + 2] = pos(2, 0);
        }
    }


    inline CtrlVector clamp_control_r(CtrlVector &control, const mjtNum *limits)
    {
        CtrlVector clamped_ctrl;

        for (auto row = 0; row < control.rows(); ++row)
            clamped_ctrl(row, 0) = std::clamp(control(row, 0), limits[row * 2], limits[row * 2 + 1]);

        return clamped_ctrl;
    }


    inline void clamp_control(CtrlVector &control, const mjtNum *limits)
    {
        for (auto row = 0; row < control.rows(); ++row)
            control(row, 0) = std::clamp(control(row, 0), limits[row * 2], limits[row * 2 + 1]);
    }


    inline void set_control_data(const mjData *data, const CtrlVector &ctrl, const mjModel* m)
    {
        std::copy(ctrl.data(), ctrl.data()+m->nu, data->ctrl);
    }


    inline void fill_state_vector(const mjData *data, StateVector &state, const mjModel* m)
    {
        std::copy(data->qpos, data->qpos+m->nq, state.data());
        std::copy(data->qvel, data->qvel+m->nv, state.data()+SimulationParameters::n_jpos);
    }


    inline void fill_ctrl_vector(const mjData *data, CtrlVector &ctrl, const mjModel* m)
    {
        std::copy(data->ctrl, data->ctrl+m->nu, ctrl.data());
    }


    inline void apply_ctrl_update_state(const CtrlVector& ctrl, StateVector& state, mjData* d, const mjModel* m, mjfGeneric ctrl_cb = nullptr)
    {
        if (ctrl_cb)
            mjcb_control = ctrl_cb;
        set_control_data(d, ctrl, m);
        mj_step(m, d);
        fill_state_vector(d, state, m);
    }


    inline void rollout_dynamics(const std::vector<CtrlVector>& ctrls, std::vector<StateVector>& states,
                                 mjData *d, const mjModel *m, mjfGeneric ctrl_cb = nullptr)
    {
        if (ctrl_cb)
            mjcb_control = ctrl_cb;

        fill_state_vector(d, states.front(), m);
        for(auto iteration = 0; iteration < ctrls.size(); ++iteration)
        {
            set_control_data(d, ctrls[iteration], m);
            mj_step(m, d);
            fill_state_vector(d, states[iteration+ 1], m);
        }
    }
}

#endif //OPTCONTROL_MUJOCO_MUJOCO_UTILS_H
