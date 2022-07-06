
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
public:
    explicit BaseCost(StateVector x_goal, StateMatrix x_gain, StateMatrix x_tgain, CtrlMatrix u_gain, const mjModel* m) :
    m_x_goal(std::move(x_goal)), m_x_gain(std::move(x_gain)), m_x_tgain(std::move(x_tgain)), m_u_gain(std::move(u_gain)), m_m(m) {};

    virtual double traj_running_cost(const std::vector<StateVector>& x_vec, const std::vector<CtrlVector>& u_vec, const mjData* d)
    {
        T& underlying = static_cast<T&>(*this);
        underlying.traj_running_cost(x_vec, u_vec, d);
    }

    virtual void compute_value(const std::vector<StateVector>& x_vec, const std::vector<CtrlVector>& u_vec, std::vector<double>& value_vec) const
    {

        const T& underlying = static_cast<const T&>(*this);
        underlying.compute_value(x_vec, u_vec, value_vec);
    }


protected:
    using RunningCostPtr = double (const StateVector&, const CtrlVector&, const StateMatrix&, const CtrlMatrix&, const mjData* d);
    using RunningCostPtr_x = double (const StateVector&, const StateMatrix&, const mjData* d);
    using RunningCostPtr_u = double (const CtrlVector&, const CtrlMatrix&, const mjData* d);

    RunningCostPtr* r_cost = nullptr;
    RunningCostPtr_x* r_xcost = nullptr;
    RunningCostPtr_u* r_ucost = nullptr;
    StateVector m_x_goal;
    StateMatrix m_x_gain;
    StateMatrix m_x_tgain;
    CtrlMatrix m_u_gain;
    StateVector m_x_error;
    CtrlVector m_u_error;
    StateVector m_x;
    CtrlVector m_u;
    const mjModel *m_m;
};


class QRCst : public BaseCost<QRCst>
{
    friend class BaseCost<QRCst>;


public:
    QRCst(StateVector x_goal, StateMatrix x_gain, StateMatrix x_tgain, CtrlMatrix u_gain, const mjModel* m) :
            BaseCost<QRCst>(std::move(x_goal), std::move(x_gain), std::move(x_tgain), std::move(u_gain), m)
    {
        r_cost = [](const StateVector& x_err, const CtrlVector& u_err, const StateMatrix& x_gain, const CtrlMatrix& u_gain, const mjData* d){
            return (x_err.transpose() * x_gain * x_err + u_err.transpose() * u_gain * u_err)(0, 0);
        };

        r_xcost =  [](const StateVector& x_err, const StateMatrix& x_gain, const mjData* d){
            return (x_err.transpose() * x_gain * x_err)(0, 0);
        };

        r_ucost = [](const CtrlVector& u_err, const CtrlMatrix& u_gain, const mjData* d){
            return (u_err.transpose() * u_gain * u_err)(0, 0);
        };
    };


    void update_errors(const mjData *d)
    {
        MujocoUtils::fill_state_vector(d, m_x, m_m);
        MujocoUtils::fill_ctrl_vector(d, m_u, m_m);
        m_x_error = m_x - m_x_goal;
        m_u_error = m_u;
    }


    double traj_running_cost(const std::vector<StateVector> &x_vec, const std::vector<CtrlVector> &u_vec, const mjData* d) override
    {
        double cst = 0;

        for(unsigned int row = 0; row < u_vec.size(); ++row){
            const StateVector x_err = x_vec[row] - m_x_goal;
            cst += r_cost(x_err, u_vec[row], m_x_gain, m_u_gain, nullptr);
        }

        m_x_error = m_x_goal - x_vec.back();
        //Running cost + terminal cost
        return cst + (m_x_error.transpose() * m_x_tgain * m_x_error)(0, 0);
    }


    void compute_value(const std::vector<StateVector>& x_vec, const std::vector<CtrlVector>& u_vec, std::vector<double>& value_vec) const override
    {
        using namespace GenericMap;
        using T_op = double;

        std::for_each(value_vec.begin(), value_vec.end(),
                      [&, idx = 0](double& value)mutable {
                          value += r_cost(x_vec[idx], u_vec[idx], m_x_gain, m_u_gain, nullptr); ++idx;
                      });

        auto add = [](double in1, double in2){return in1 + in2;};
        consecutive_map<T_op, T_op>(value_vec.data(), value_vec.size(), add);
    }


    double L(const mjData* d)
    {
        update_errors(d);
        return r_cost(m_x_error, m_u, m_x_gain, m_u_gain, nullptr);
    }


    StateVector L_x(const mjData *d)
    {
        update_errors(d);
        return m_x_error.transpose() * (2 * m_x_gain);
    }


    StateMatrix L_xx(const mjData *d)
    {
        update_errors(d);
        return 2 * m_x_gain;
    }


    CtrlVector L_u(const mjData *d)
    {
        update_errors(d);
        return (m_u_error.transpose() * (2 * m_u_gain)) * 2;
    }


    CtrlMatrix L_uu(const mjData *d)
    {
        update_errors(d);
        return 2 * m_u_gain * 2;
    }


    CtrlStateMatrix L_ux(const mjData *d)
    {
        update_errors(d);
        return CtrlStateMatrix::Zero();
    }


    mjtNum Lf(const mjData *d)
    {
        update_errors(d);
        return (m_x_error.transpose() * m_x_tgain * m_x_error)(0, 0);
    }


    StateVector Lf_x(const mjData *d)
    {
        update_errors(d);
        return m_x_error.transpose() * (2 * m_x_tgain);
    }


    StateMatrix Lf_xx()
    {
        return 2 * m_x_tgain;
    }
};

#endif //OPTCONTROL_MUJOCO_GENERIC_COST_H
