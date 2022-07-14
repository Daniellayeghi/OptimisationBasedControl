#include <omp.h>
#include "par_mppi_ddp.h"

using namespace SimulationParameters;
using namespace MujocoUtils;
constexpr const int nthreads = n_threads;

static void (*s_callback_ctrl)(const mjModel *, mjData *);
static auto step = [](const mjModel* m, mjData* d, mjfGeneric cbc){mjcb_control = cbc;  mj_step(m, d);};

MPPIDDPPar::MPPIDDPPar(const mjModel* m, PICost& cost, MPPIDDPParamsPar& params):
        m_padded_cst(params.m_k_samples, std::vector<double>(8)),
        m_sample_ctrl_traj(params.m_k_samples),
        m_dist_gens(nthreads,{params.pi_ctrl_mean, params.ctrl_variance,
                              params.m_sim_time, true,   params.m_seed}),
        m_m(m),
        m_params(params),
        m_cost_func(cost)
{
#pragma omp parallel for default(none) shared(m_params, m_sample_ctrl_traj) num_threads(nthreads)
    for(auto sample = 0; sample < m_params.m_k_samples; ++sample)
        m_sample_ctrl_traj[sample].resize(1, m_params.m_sim_time*n_ctrl);

    for(auto thread = 0; thread < nthreads; ++thread)
    {
        m_thread_mjdata.emplace_back(mj_makeData(m_m));
        m_dist_gens[thread].randN.seed(m_params.m_seed+thread);
    }

    cached_control = CtrlVector::Zero();
    m_u_traj.assign(m_params.m_sim_time, CtrlVector::Zero());
    m_u_traj_cp.assign(m_params.m_sim_time, CtrlVector::Zero());
    m_x_traj.assign(m_params.m_sim_time + 1, StateVector::Zero());
    m_u_traj_new.assign(m_params.m_sim_time+1, CtrlVector ::Zero());
    m_ddp_cov_inv_vec.assign(m_params.m_sim_time, CtrlMatrix::Identity());
    m_state_value.first.assign(m_params.m_sim_time+1, StateVector::Zero());
    m_state_value.second.assign(m_params.m_sim_time+1, 0);

    auto m_carry_over = m_params.m_k_samples % n_threads;
    m_per_thread_sample = (m_params.m_k_samples - m_carry_over)/ n_threads;

    if (m_params.m_grav_comp)
        s_callback_ctrl = [](const mjModel* m, mjData *d){
            //mju_copy(d->qfrc_applied, d->qfrc_bias, n_ctrl);
        };
    else
        s_callback_ctrl = [](const mjModel* m, mjData *d) {};
}


void MPPIDDPPar::compute_cov_from_hess(const std::vector<CtrlMatrix> &ddp_variance)
{
#pragma omp  parallel for default(none) shared(m_ddp_cov_inv_vec, ddp_variance, m_params) num_threads(nthreads)
    for (auto elem = 0; elem < m_params.m_sim_time; ++elem)
        m_ddp_cov_inv_vec[elem] = (ddp_variance[elem] / m_params.ddp_cov_reg).llt().solve(CtrlMatrix::Identity());
}


//TODO: remove critical need one rng per thread.
void MPPIDDPPar::fill_ctrl_samples()
{
#pragma omp  parallel default(none) shared(m_dist_gens, m_sample_ctrl_traj, m_params, m_per_thread_sample) num_threads(nthreads)
    {
        int id = omp_get_thread_num();
        int adjust = 0;
        if (id == nthreads-1) {adjust = m_params.m_k_samples % n_threads;}
        const auto limit = (id + 1) * m_per_thread_sample + adjust;
        for (int sample = id * m_per_thread_sample; sample < limit; ++sample) {
            m_dist_gens[id].samples_fill(m_sample_ctrl_traj[sample]);
        }
    }
}


void MPPIDDPPar::convert_costs_to_is_weight()
{
    auto norm_const = compute_normalisation_constant();
#pragma omp  parallel for default(none) shared(m_padded_cst, m_params, norm_const) num_threads(nthreads)
    for(auto sample = 0; sample < m_params.m_k_samples; ++sample)
    {
        m_padded_cst[sample][0] =  m_padded_cst[sample][0] / norm_const;
    }
}


void MPPIDDPPar::exponentiate_costs(double min_cost)
{
#pragma omp  parallel for default(none) shared(m_padded_cst, m_params, min_cost) num_threads(nthreads)
    for(auto sample = 0; sample < m_params.m_k_samples; ++sample)
    {
        auto cost_diff = m_padded_cst[sample][0] - min_cost;
        m_padded_cst[sample][0] = std::exp(-(1 / m_params.m_lambda) * (cost_diff));
    }
}


void MPPIDDPPar::weight_samples_ctrl_traj()
{
    convert_costs_to_is_weight();
#pragma omp  parallel for default(none) shared(m_padded_cst, m_params, m_sample_ctrl_traj) num_threads(nthreads)
    for(auto sample = 0; sample < m_params.m_k_samples; ++sample)
    {
        m_sample_ctrl_traj[sample] =  m_sample_ctrl_traj[sample] * m_padded_cst[sample][0];
    }
}


double MPPIDDPPar::compute_normalisation_constant()
{
    auto min_cost = GenericUtils::parallel_min(m_padded_cst);

    exponentiate_costs(min_cost.val);
    double normalise_const = 0;
#pragma omp  parallel for reduction(+:normalise_const) default(none) shared(m_padded_cst, m_params) num_threads(nthreads)
    for(auto sample = 0; sample < m_params.m_k_samples; ++sample)
    {
        normalise_const += m_padded_cst[sample][0];
    }

    return normalise_const;
}


void MPPIDDPPar::perturb_ctrl_traj()
{
//#pragma omp  declare reduction(+:CtrlVector: omp_out=omp_out+omp_in)\
//initializer(omp_priv=CtrlVector::Zero(omp_orig.rows(), omp_orig.cols()))

#pragma omp parallel for default(none) collapse(2) shared(m_params, m_u_traj, m_sample_ctrl_traj) num_threads(1)
    for (auto time = 0; time < m_params.m_sim_time; ++time) {
        for (auto sample = 0; sample < m_params.m_k_samples; ++sample) {
            Eigen::Ref<const CtrlVector> pert_sample(
                    m_sample_ctrl_traj[sample].block(0, time * n_ctrl, 1, n_ctrl).eval().transpose()
            );
            m_u_traj[time] += pert_sample;
        }
    }
}



void MPPIDDPPar::rollout_trajectories(const mjData* d)
{
#pragma omp  parallel default(none) shared(m_thread_mjdata, d, m_cost_func, m_sample_ctrl_traj, m_params, m_u_traj, m_m, m_per_thread_sample) num_threads(nthreads)
    {
        int id = omp_get_thread_num();
        int adjust = 0;
        if (id == nthreads-1) adjust = m_params.m_k_samples % n_threads;
        const auto iter_limit = (id + 1) * m_per_thread_sample + adjust;
        for (int sample = id * m_per_thread_sample; sample < iter_limit; ++sample) {
            m_ThreadData t_d;
            fill_state_vector(d, t_d.current, m_m);
            copy_data(m_m, d, m_thread_mjdata[id]);
            const auto &ctrl_traj = m_sample_ctrl_traj[sample];
            auto &mjdata = m_thread_mjdata[id];
            for (int time = 0; time < m_params.m_sim_time - 1; ++time)
            {
                // Set sampled perturbation
                Eigen::Ref<const CtrlVector> pert_sample(
                        ctrl_traj.block(0, time*n_ctrl, 1, n_ctrl).eval().transpose()
                );

                t_d.instant_ctrl = m_u_traj[time] + pert_sample;
                // Forward simulate controls and compute running cost
                if(m_params.m_grav_comp)
                    mju_copy(mjdata->qfrc_applied, mjdata->qfrc_bias, n_ctrl);

                MujocoUtils::apply_ctrl_update_state(t_d.instant_ctrl, t_d.next, mjdata, m_m);

                m_padded_cst[sample][0] += m_cost_func.pi_ddp_cost(
                        t_d.next, m_u_traj[time], pert_sample,
                        m_params.m_ddp_args.first[time], m_ddp_cov_inv_vec[time], mjdata, m_m);

            }

            // Set final pert sample
            Eigen::Ref<const CtrlVector> final_sample(ctrl_traj.block(
                    0, (m_params.m_sim_time - 1) * n_ctrl, 1, n_ctrl
            ).eval().transpose());
            // Apply final sample
            t_d.instant_ctrl = m_u_traj.back() + final_sample;

            if(m_params.m_grav_comp)
                mju_copy(mjdata->qfrc_applied, mjdata->qfrc_bias, n_ctrl);
            MujocoUtils::apply_ctrl_update_state(t_d.instant_ctrl, t_d.next, mjdata, m_m);

            // Compute terminal cost
            m_padded_cst[sample][0] += m_cost_func.terminal_cost(mjdata, m_m);
        }
    }
}


void MPPIDDPPar::control(const mjData* d, const bool skip)
{
    // TODO: compute the previous trajectory cost here with the new state then compare to the new one
    if (not skip)
    {
        compute_cov_from_hess(m_params.m_ddp_args.second);
        for (auto iteration = 0; iteration < m_params.iteration; ++iteration)
        {
            std::fill(m_padded_cst.begin(), m_padded_cst.end(), std::vector<double>(8, 0));
            fill_ctrl_samples();
            rollout_trajectories(d);
            weight_samples_ctrl_traj();
            perturb_ctrl_traj();
            copy_data(m_m, d, m_thread_mjdata.front());
            rollout_dynamics(m_u_traj, m_x_traj, m_thread_mjdata.front(), m_m);
            m_u_traj_cp = m_u_traj;
        }
    }

    cached_control = m_u_traj.front();
    m_cost_func.compute_state_value(m_x_traj, m_u_traj, m_state_value, m_thread_mjdata.front(), m_m);
    std::rotate(m_u_traj.begin(), m_u_traj.begin() + 1, m_u_traj.end());
    m_u_traj.back() = CtrlVector::Zero();
}
