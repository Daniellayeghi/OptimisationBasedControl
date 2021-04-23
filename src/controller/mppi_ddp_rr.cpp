//#include "mppi_ddp.h"
//
//#include <iostream>
//#include <numeric>
//#include "MPPI.h"
//#include "../utilities/eigen_norm_dist.h"
//
//using namespace SimulationParameters;
//
//namespace
//{
//    template <typename T>
//    void copy_data(const mjModel * model, const mjData *data, T *data_cp)
//    {
//        data_cp->time = data->time;
//        mju_copy(data_cp->qpos, data->qpos, model->nq);
//        mju_copy(data_cp->qvel, data->qvel, model->nv);
//        mju_copy(data_cp->qacc, data->qacc, model->nv);
//        mju_copy(data_cp->qfrc_applied, data->qfrc_applied, model->nv);
//        mju_copy(data_cp->xfrc_applied, data->xfrc_applied, 6*model->nbody);
//        mju_copy(data_cp->ctrl, data->ctrl, model->nu);
//    }
//
//
//    template<int state_size>
//    inline void fill_state_vector(mjData* data, Eigen::Matrix<double, state_size, 1>& state)
//    {
//        for(auto row = 0; row < state.rows()/2; ++row)
//        {
////            state(row, 0) = BasicMath::wrap_to_min_max(data->qpos[row],-M_PI, M_PI);
//            state(row, 0) = data->qpos[row];
//            state(row+state.rows()/2, 0) = data->qvel[row];
//        }
//    }
//
//
//    template<int ctrl_size>
//    void set_control_data(mjData* data, const Eigen::Matrix<double, ctrl_size, 1>& ctrl)
//    {
//        for(auto row = 0; row < ctrl.rows(); ++row)
//        {
//            data->ctrl[row] = ctrl(row, 0);
//        }
//    }
//
//
//    // TODO: CLamp like ilqr and put clamping in some form of util
//    template<int ctrl_size>
//    void clamp_control(Eigen::Matrix<mjtNum, ctrl_size, 1>& control, mjtNum max_bound, mjtNum min_bound)
//    {
//        for (auto row = 0; row < control.rows(); ++row)
//        {
//            control(row, 0) = std::clamp(control(row, 0), min_bound, max_bound);
//        }
//    }
//}
//
//
//template<int state_size, int ctrl_size>
//MPPIDDP<state_size, ctrl_size>::MPPIDDP(const mjModel* m,
//                                        const QRCostDDP<state_size, ctrl_size>& cost,
//                                        const MPPIDDPParams<ctrl_size>& params)
//        :
//        m_params(params),
//        m_cost_func(cost),
//        m_m(m),
//        m_normX_cholesk(Eigen::Matrix<double,n_ctrl, 1>::Zero(), params.ctrl_variance, true)
//
//{
//    m_d_cp = mj_makeData(m_m);
//
//    _cached_control = MPPIDDP<state_size, ctrl_size>::ctrl_vector::Zero();
//
//    m_state.assign(m_params.m_sim_time, Eigen::Matrix<double, state_size, 1>::Zero());
//    m_control.assign(m_params.m_sim_time, Eigen::Matrix<double, ctrl_size, 1>::Zero());
//    m_delta_control.assign(m_params.m_sim_time,std::vector<Eigen::Matrix<double, ctrl_size, 1>>(
//            m_params.m_k_samples,Eigen::Matrix<double, ctrl_size, 1>::Random()
//    ));
//
//    m_delta_cost_to_go.assign(m_params.m_k_samples,0);
//}
//
//
//template<int state_size, int ctrl_size>
//typename MPPIDDP<state_size, ctrl_size>::ctrl_vector
//MPPIDDP<state_size, ctrl_size>::total_entropy(const std::vector<MPPIDDP<state_size, ctrl_size>::ctrl_vector>& delta_control_samples,
//                                              const double min_cost) const
//{
//    MPPIDDP<state_size, ctrl_size>::ctrl_vector numerator = MPPIDDP<state_size, ctrl_size>::ctrl_vector::Zero();
//    double denomenator =  0;
//    for (auto& sample_cost: m_delta_cost_to_go)
//    {
//        auto cost_diff = sample_cost - min_cost;
//        denomenator += (std::exp(-(1 / m_params.m_lambda) * (cost_diff)));
//    }
//
//    for (unsigned long col = 0; col < m_delta_cost_to_go.size(); ++col)
//    {
//        numerator += (
//                std::exp(-(1 / m_params.m_lambda) * (m_delta_cost_to_go[col] - min_cost)) * delta_control_samples[col]
//                );
//    }
//    return numerator/(
//            denomenator * std::pow(denomenator, - m_params.importance/(1+ m_params.importance)* m_params.m_scale)
//            );
//}
//
//
//template <int state_size, int ctrl_size>
//void MPPIDDP<state_size, ctrl_size>::MPPIDDP::compute_control_trajectory()
//{
//    auto min_cost = std::min_element(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end());
//
//    for (auto time = 0; time < m_params.m_sim_time; ++time)
//    {
//        m_control[time] += (total_entropy(m_delta_control[time], *min_cost));
//    }
//
//    _cached_control = m_control.front();
//
//    std::rotate(m_control.begin(), m_control.begin() + 1, m_control.end());
//    m_control.back() = Eigen::Matrix<double, ctrl_size, 1>::Zero();
//}
//
//
//template <int state_size, int ctrl_size>
//void MPPIDDP<state_size, ctrl_size>::MPPIDDP::adapt_variance()
//{
//    Eigen::Matrix<double, n_ctrl, n_ctrl> new_variance; new_variance.setZero();
//    for(const auto & delta_control_samples : m_delta_control)
//        for(const auto & delta_control_sample: delta_control_samples)
//        {
//
//        }
//}
//
//
////template<int state_size, int ctrl_size>
////void MPPIDDP<state_size, ctrl_size>::control(const mjData* d, std::vector<MPPIDDP<state_size, ctrl_size>::ctrl_vector>& ddp_ctrl)
////{
////    MPPIDDP<state_size, ctrl_size>::ctrl_vector instant_control;
////
////    fill_state_vector(m_d_cp, m_state.front());
////    std::fill(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0);
////
////    for(auto sample = 0; sample < m_params.m_k_samples; ++sample)
////    {
////        copy_data(m_m, d, m_d_cp);
////        for (auto time = 0; time < m_params.m_sim_time - 1; ++time)
////        {
////            // u += du -> du ~ N(0, variance)
////            m_delta_control[time][sample] = 0.99 * MPPIDDP<state_size, ctrl_size>::ctrl_vector::Random();
////            instant_control = m_control[time] + m_delta_control[time][sample];
////
////            // Forward simulate controls
////            set_control_data(m_d_cp, instant_control);
////            mj_step(m_m, m_d_cp);
////            fill_state_vector(m_d_cp, m_state[time + 1]);
////
////            // Compute cost-to-go of the controls
////            m_delta_cost_to_go[sample] = m_delta_cost_to_go[sample] + m_cost_func(m_state[time + 1],
////                                                                                  m_control[time],
////                                                                                  m_delta_control[time][sample],
////                                                                                  ddp_ctrl[time]);
////        }
////        m_delta_cost_to_go[sample] = m_delta_cost_to_go[sample] + m_cost_func.terminal_cost(m_state.back());
////        traj_cost += std::accumulate(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0.0)/m_delta_cost_to_go.size();
////        m_delta_control.back()[sample] = 0.99 * MPPIDDP<state_size, ctrl_size>::ctrl_vector::Random();
////    }
////    traj_cost /= m_params.m_k_samples;
////    compute_control_trajectory();
////    std::cout << "diff: " <<  ddp_ctrl.front() - _cached_control << std::endl;
////
////}
//
//template<int state_size, int ctrl_size>
//void MPPIDDP<state_size, ctrl_size>::control(const mjData* d, std::vector<MPPIDDP<state_size, ctrl_size>::ctrl_vector>& ddp_ctrl)
//{
//    MPPIDDP<state_size, ctrl_size>::ctrl_vector instant_control;
//
//    fill_state_vector(m_d_cp, m_state.front());
//    std::fill(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0);
//
//    for (auto sample = 0; sample < m_params.m_k_samples; ++sample)
//    {
//        copy_data(m_m, d, m_d_cp);
//        for(auto time = 0; time < m_params.m_sim_time - 1; ++time)
//        {
////             u += du -> du ~ N(0, variance)
//            m_delta_control[time][sample] = m_normX_cholesk.samples(1);
////            m_delta_control[time][sample] = m_params.m_variance * MPPIDDP<state_size, ctrl_size>::ctrl_vector::Random();
//            instant_control = m_control[time] + m_delta_control[time][sample];
//
//            // Forward simulate controls
//            set_control_data(m_d_cp, instant_control);
//            mj_step(m_m, m_d_cp);
//            fill_state_vector(m_d_cp, m_state[time + 1]);
//
//            // Compute cost-to-go of the controls
//            m_delta_cost_to_go[sample] = m_delta_cost_to_go[sample]
//                                         + m_cost_func(
//                    m_state[time + 1],
//                    m_control[time],
//                    m_delta_control[time][sample],
//                    ddp_ctrl[time]
//            );
//        }
//        m_delta_cost_to_go[sample] = m_delta_cost_to_go[sample] + m_cost_func.terminal_cost(m_state.back());
//        traj_cost += std::accumulate(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0.0)/m_delta_cost_to_go.size();
////        m_delta_control.back()[sample] = m_params.m_variance * MPPIDDP<state_size, ctrl_size>::ctrl_vector::Random();
//        m_delta_control.back()[sample] = m_normX_cholesk.samples(1);
//    }
//    traj_cost /= m_params.m_k_samples;
//    compute_control_trajectory();
////    std::cout << "diff: " <<  ddp_ctrl.front() - _cached_control << std::endl;
//
//}
//
//
////template<int state_size, int ctrl_size>
////void MPPIDDP<state_size, ctrl_size>::control(const mjData* d, std::vector<MPPIDDP<state_size, ctrl_size>::ctrl_vector>& ddp_ctrl)
////{
////    MPPIDDP<state_size, ctrl_size>::ctrl_vector instant_control;
////
////    fill_state_vector(m_samples_d_cp, m_state.front());
////    std::fill(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0);
////
////    for (auto sample = 0; sample < m_params.m_k_samples- 1; ++sample)
////    {
////        copy_data(m_m, d, m_samples_d_cp);
////        for(auto time = 0; time < m_params.m_sim_time - 1; ++time)
////        {
//////             u += du -> du ~ N(0, variance)
////            m_delta_control[time][sample] = m_normX_cholesk.samples(1);
//////            m_delta_control[time][sample] = m_params.m_variance * MPPIDDP<state_size, ctrl_size>::ctrl_vector::Random();
////            instant_control = m_control[time] + m_delta_control[time][sample];
////
////            // Forward simulate controls
////            set_control_data(m_samples_d_cp, instant_control);
////            mj_step(m_m, m_samples_d_cp);
////            fill_state_vector(m_samples_d_cp, m_state[time + 1]);
////
////            // Compute cost-to-go of the controls
////            m_delta_cost_to_go[sample] = m_delta_cost_to_go[sample]
////                    + m_cost_func(
////                            m_state[time + 1],
////                            m_control[time],
////                            m_delta_control[time][sample],
////                            ddp_ctrl[time]
////                            );
////        }
////        m_delta_cost_to_go[sample] = m_delta_cost_to_go[sample] + m_cost_func.terminal_cost(m_state.back());
////        traj_cost += std::accumulate(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0.0)/m_delta_cost_to_go.size();
//////        m_delta_control.back()[sample] = m_params.m_variance * MPPIDDP<state_size, ctrl_size>::ctrl_vector::Random();
////        m_delta_control.back()[sample] = m_normX_cholesk.samples(1);
////    }
////    traj_cost /= m_params.m_k_samples;
////    compute_control_trajectory();
//////    std::cout << "diff: " <<  ddp_ctrl.front() - _cached_control << std::endl;
////
////}
//
//
//template<int state_size, int ctrl_size>
//MPPIDDP<state_size, ctrl_size>::~MPPIDDP()
//{
//    mj_deleteData(m_d_cp);
//}
//
//template class MPPIDDP<n_jpos + n_jvel, n_ctrl>;
