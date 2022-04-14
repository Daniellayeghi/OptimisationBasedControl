#ifndef OPTCONTROL_MUJOCO_MUJOCO_UTILS_H
#define OPTCONTROL_MUJOCO_MUJOCO_UTILS_H

#include "Eigen/Core"
#include "../parameters/simulation_params.h"
#include <random>

namespace MujocoUtils
{

    struct DummyMjData
    {
        explicit DummyMjData(const mjModel* m)
        {
            m_qpos.assign(m->nq, 0);
            m_qvel.assign(m->nv, 0);
            m_qacc.assign(m->nv, 0);
            m_ctrl.assign(m->nu, 0);
            m_qfrc_applied.assign(m->nv, 0);
            m_xfrc_applied.assign(m->nbody, 0);

            // Set ptrs
            qpos = m_qpos.data();
            qvel = m_qvel.data();
            qacc = m_qacc.data();
            ctrl = m_ctrl.data();
            qfrc_applied = m_qfrc_applied.data();
            xfrc_applied = m_xfrc_applied.data();
        }

    public:
        SimulationParameters::scalar_type time = 0;
        scalar_type* qpos;
        scalar_type* qvel;
        scalar_type* qacc;
        scalar_type* ctrl;
        scalar_type* qfrc_applied;
        scalar_type* xfrc_applied;

    private:
        std::vector<SimulationParameters::scalar_type> m_qpos;
        std::vector<SimulationParameters::scalar_type> m_qvel;
        std::vector<SimulationParameters::scalar_type> m_qacc;
        std::vector<SimulationParameters::scalar_type> m_ctrl;
        std::vector<SimulationParameters::scalar_type> m_qfrc_applied;
        std::vector<SimulationParameters::scalar_type> m_xfrc_applied;
    };


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
