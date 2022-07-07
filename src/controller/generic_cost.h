
#ifndef OPTCONTROL_MUJOCO_GENERIC_COST_H
#define OPTCONTROL_MUJOCO_GENERIC_COST_H

#include <mujoco/mujoco.h>

#include <utility>
#include "../parameters/simulation_params.h"
#include "../utilities/generic_algs.h"
#include "../utilities/mujoco_utils.h"

using namespace SimulationParameters;

template <typename T>
class BaseCost
{
protected:
    using RunningCostPtr = double (const StateVector&, const CtrlVector&, const StateMatrix&, const CtrlMatrix&, const mjData* d, const mjModel* m);

public:
    explicit BaseCost(StateVector x_goal, StateMatrix  x_gain, StateMatrix x_tgain, CtrlMatrix u_gain,  RunningCostPtr* r_cost) :
    m_r_cost(r_cost), m_x_goal(std::move(x_goal)), m_x_gain(std::move(x_gain)), m_x_tgain(std::move(x_tgain)), m_u_gain(std::move(u_gain))

    {};

    virtual double traj_running_cost(const std::vector<StateVector>& x_vec, const std::vector<CtrlVector>& u_vec, const mjData* d, const mjModel* m)
    {
        T& underlying = static_cast<T&>(*this);
        underlying.traj_running_cost(x_vec, u_vec, d, m);
    }


    virtual void compute_value(const std::vector<StateVector>& x_vec, const std::vector<CtrlVector>& u_vec, std::vector<double>& value_vec, const mjData* d, const mjModel* m) const
    {
        const T& underlying = static_cast<const T&>(*this);
        underlying.compute_value(x_vec, u_vec, value_vec,d, m);
    }


    virtual double running_cost(const mjData* d, const mjModel* m)
    {
        T &underlying = static_cast<T &>(*this);
        underlying.running_cost(d, m);
    }


protected:
    RunningCostPtr* m_r_cost = nullptr;
    const StateVector m_x_goal;
    const StateMatrix m_x_gain;
    const StateMatrix m_x_tgain;
    const CtrlMatrix m_u_gain;
    StateVector m_x_error;
    CtrlVector m_u_error;
    StateVector m_x;
    CtrlVector m_u;
};


class QRCst : public BaseCost<QRCst>
{
    friend class BaseCost<QRCst>;
public:
    QRCst(StateVector x_goal, StateMatrix x_gain, StateMatrix x_tgain, CtrlMatrix u_gain, RunningCostPtr* r_cost) :
            BaseCost<QRCst>(std::move(x_goal), std::move(x_gain), std::move(x_tgain), std::move(u_gain), r_cost)
    {
        m_r_cost = [](const StateVector& x_err, const CtrlVector& u_err, const StateMatrix& x_gain, const CtrlMatrix& u_gain, const mjData* d, const mjModel* m){
            return (x_err.transpose() * x_gain * x_err + u_err.transpose() * u_gain * u_err)(0, 0);
        };
    };


    void update_errors(const mjData *d, const mjModel* m)
    {
        MujocoUtils::fill_state_vector(d, m_x, m);
        MujocoUtils::fill_ctrl_vector(d, m_u, m);
        m_x_error = m_x - m_x_goal;
        m_u_error = m_u;
    }


    double traj_running_cost(const std::vector<StateVector> &x_vec, const std::vector<CtrlVector> &u_vec, const mjData* d, const mjModel *m) override
    {
        double cst = 0;

        for(unsigned int row = 0; row < u_vec.size(); ++row){
            const StateVector x_err = x_vec[row] - m_x_goal;
            cst += m_r_cost(x_err, u_vec[row], m_x_gain, m_u_gain, nullptr, nullptr);
        }

        m_x_error = m_x_goal - x_vec.back();
        //Running cost + terminal cost
        return cst + (m_x_error.transpose() * m_x_tgain * m_x_error)(0, 0);
    }


    void compute_value(const std::vector<StateVector>& x_vec, const std::vector<CtrlVector>& u_vec, std::vector<double>& value_vec, const mjData* d, const mjModel* m) const override
    {
        using namespace GenericMap;
        using T_op = double;

        std::for_each(value_vec.begin(), value_vec.end(),
                      [&, idx = 0](double& value)mutable {
                          value += m_r_cost(x_vec[idx], u_vec[idx], m_x_gain, m_u_gain, nullptr, nullptr); ++idx;
                      });

        auto add = [](double in1, double in2){return in1 + in2;};
        consecutive_map<T_op, T_op>(value_vec.data(), value_vec.size(), add);
    }


    double running_cost(const mjData* d, const mjModel* m) override
    {
        return  L(d, m);
    }


    double L(const mjData* d, const mjModel* m)
    {
        update_errors(d, m);
        return m_r_cost(m_x_error, m_u, m_x_gain, m_u_gain, nullptr, nullptr);
    }


    StateVector L_x(const mjData *d, const mjModel* m)
    {
        update_errors(d, m);
        return m_x_error.transpose() * (2 * m_x_gain);
    }


    StateMatrix L_xx(const mjData *d, const mjModel* m)
    {
        update_errors(d, m);
        return 2 * m_x_gain;
    }


    CtrlVector L_u(const mjData *d, const mjModel* m)
    {
        update_errors(d, m);
        return (m_u_error.transpose() * (2 * m_u_gain)) * 2;
    }


    CtrlMatrix L_uu(const mjData *d, const mjModel* m)
    {
        update_errors(d, m);
        return 2 * m_u_gain * 2;
    }


    CtrlStateMatrix L_ux(const mjData *d, const mjModel* m)
    {
        update_errors(d, m);
        return CtrlStateMatrix::Zero();
    }


    mjtNum Lf(const mjData *d, const mjModel* m)
    {
        update_errors(d, m);
        return (m_x_error.transpose() * m_x_tgain * m_x_error)(0, 0);
    }


    StateVector Lf_x(const mjData *d, const mjModel* m)
    {
        update_errors(d, m);
        return m_x_error.transpose() * (2 * m_x_tgain);
    }


    StateMatrix Lf_xx()
    {
        return 2 * m_x_tgain;
    }
};


struct MPPIDDPCstParams {
    const int m_importance = 0;
    const double m_lambda  = 0;
    const CtrlMatrix  m_ctrl_variance_inv;
    const std::function<double(const mjData* data, const mjModel *model)>&  m_importance_reg =
            [](const mjData* data=nullptr, const mjModel *model=nullptr){return 1.0;};
};

class PICost : public BaseCost<PICost>
{
    friend class BaseCost<PICost>;
public:
    PICost(StateVector x_goal, StateMatrix x_gain, StateMatrix x_tgain, CtrlMatrix u_gain, RunningCostPtr* r_cost, RunningCostPtr* terminal_cost, const MPPIDDPCstParams& cst_params) :
            BaseCost<PICost>(std::move(x_goal), std::move(x_gain), std::move(x_tgain), std::move(u_gain), r_cost), m_terminal_cost(terminal_cost), m_cst_params(cst_params)
    {
    };


    void update_errors(const mjData *d, const mjModel* m)
    {
        MujocoUtils::fill_state_vector(d, m_x, m);
        MujocoUtils::fill_ctrl_vector(d, m_u, m);
        m_x_error = m_x - m_x_goal;
        m_u_error = m_u;
    }


    double traj_running_cost(const std::vector<StateVector> &x_vec, const std::vector<CtrlVector> &u_vec, const mjData* d, const mjModel *m) override
    {
        double cst = 0;

        for(unsigned int row = 0; row < u_vec.size(); ++row){
            const StateVector x_err = x_vec[row] - m_x_goal;
            cst += m_r_cost(x_err, u_vec[row], m_x_gain, m_u_gain, d, m);
        }

        m_x_error = m_x_goal - x_vec.back();
        //Running cost + terminal cost
        return cst + (m_x_error.transpose() * m_x_tgain * m_x_error)(0, 0);
    }


    void compute_value(const std::vector<StateVector>& x_vec, const std::vector<CtrlVector>& u_vec, std::vector<double>& value_vec, const mjData* d, const mjModel* m) const override
    {
        using namespace GenericMap;
        using T_op = double;

        std::for_each(value_vec.begin(), value_vec.end(),
                      [&, idx = 0](double& value)mutable {
                          value += m_r_cost(x_vec[idx], u_vec[idx], m_x_gain, m_u_gain, nullptr, nullptr); ++idx;
                      });

        auto add = [](double in1, double in2){return in1 + in2;};
        consecutive_map<T_op, T_op>(value_vec.data(), value_vec.size(), add);
    }


    double pi_ddp_cost(const CtrlVector& old_u, const CtrlVector& ddp_mean_u, const CtrlMatrix& ddp_covariance_inv, const mjData* d, const mjModel* m)
    {
        update_errors(d, m);
        const auto importance = m_cst_params.m_importance * m_cst_params.m_importance_reg(d, m);
        double ddp_bias = ((m_u_error - ddp_mean_u).transpose() * ddp_covariance_inv * (m_u_error - ddp_mean_u)
                )(0, 0) * m_cst_params.m_importance;

        double passive_bias = (m_u_error.transpose() * m_cst_params.m_ctrl_variance_inv * m_u_error
                )(0, 0) * (- m_cst_params.m_importance);

        double common_bias = (
                old_u.transpose() * m_cst_params.m_ctrl_variance_inv * old_u +
                2 * m_u_error.transpose() * m_cst_params.m_ctrl_variance_inv * old_u)(0, 0);

        return 0.5 * (ddp_bias + passive_bias + common_bias) * m_cst_params.m_lambda + running_cost(d, m);

    };


    double running_cost(const mjData* d, const mjModel* m) override
    {
        update_errors(d, m);
        return m_r_cost(m_x_error, m_u_error, m_x_gain, m_u_gain, d, m);
    }


    double terminal_cost(const mjData* d, const mjModel* m)
    {
        update_errors(d, m);
        return m_terminal_cost(m_x_error, m_u_error, m_x_gain, m_u_gain, d, m);
    }

    RunningCostPtr* m_terminal_cost = nullptr;
    const MPPIDDPCstParams& m_cst_params;

};




#endif //OPTCONTROL_MUJOCO_GENERIC_COST_H
