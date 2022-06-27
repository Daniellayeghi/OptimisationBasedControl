#ifndef OPTCONTROL_MUJOCO_ILQR_H
#define OPTCONTROL_MUJOCO_ILQR_H

#include <vector>
#include <mujoco/mujoco.h>
#include "Eigen/Core"
#include "../parameters/simulation_params.h"
#include "../utilities/finite_diff.h"
#include "cost_function.h"
#include"generic_control.h"


struct ILQRParams
{
    double min_reg = 1e-6;
    double delta_init = 2.0;
    double delta = delta_init;
    double min_cost_red = 0;
    int simulation_time = 0;
    int iteration = 0;
    const bool m_grav_comp = false;
};


class ILQR : public BaseController<ILQR>
{
    friend class BaseController<ILQR>;
public:
    ILQR(FiniteDifference& fd,
         CostFunction& cf,
         ILQRParams& params,
         const mjModel * m,
         const mjData* d,
         const std::vector<CtrlVector>* init_u = nullptr);

    ~ILQR();
    void control(const mjData* d, bool skip = false) override;

private:
    void forward_simulate(const mjData* d);
    void forward_pass(const mjData* d);
    void update_regularizer(bool increase);
    double compute_expected_cost(double backtracker);
    void backward_pass();
    void temporal_average_covariance();
    bool minimal_grad();

    CtrlVector      Q_u(int time, const StateVector& _v_x);
    StateVector     Q_x(int time, const StateVector& _v_x);
    StateMatrix     Q_xx(int time, const StateMatrix& _v_xx);
    CtrlStateMatrix Q_ux(int time, const StateMatrix& _v_xx);
    StateCtrlMatrix Q_xu(int time, const StateMatrix& _v_xx);
    CtrlMatrix      Q_uu(int time, const StateMatrix& _v_xx);
    StateMatrix     Q_xx_reg(int time, const StateMatrix& _v_xx);
    CtrlStateMatrix Q_ux_reg(int time, const StateMatrix& _v_xx);
    StateCtrlMatrix Q_xu_reg(int time, const StateMatrix& _v_xx);
    CtrlMatrix      Q_uu_reg(int time, const StateMatrix& _v_xx);

    struct Derivatives{
        double l{}; StateVector lx; StateMatrix lxx; CtrlVector lu; CtrlMatrix luu;
        CtrlStateMatrix lux;StateMatrix fx; StateCtrlMatrix fu;
    };

    struct BackPassVars{
        CtrlVector ff_k; CtrlStateMatrix fb_k;
    };


    // Control containers
    std::vector<Derivatives> m_d_vector;
    std::vector<BackPassVars> m_bp_vector;

    //HJB Approximation
    std::vector<CtrlVector> m_Qu_traj;
    std::vector<CtrlMatrix> m_Quu_traj;
    std::vector<double> exp_cost_reduction;

public:
    std::vector<CtrlMatrix> _covariance;
    std::vector<CtrlMatrix> _covariance_new;
    std::vector<double> cost;

private:
    FiniteDifference& _fd;
    CostFunction& m_cf;
    const mjModel* _m;
    ILQRParams& m_params;

    mjData* _d_cp = nullptr;
    double _prev_total_cost = 0;
    bool m_good_backpass = true;
    std::array<double, 11> m_backtrackers{};
    StateMatrix m_regularizer;
};

#endif //OPTCONTROL_MUJOCO_ILQR_H
