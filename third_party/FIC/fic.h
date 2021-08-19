

#ifndef OPTCONTROL_MUJOCO_FIC_H
#define OPTCONTROL_MUJOCO_FIC_H

#include "Eigen/Core"
#include <cmath>

namespace uoe {

/**
 * FICController
 *
 * Implement 3d Fractal Impedance Controller
 * with smooth effort generation
 */
    class FICController
    {
    public:

        /**
         * Default initialization
         */
        FICController() :
                _posErrorMax(0.1),
                _forceMax(10.0),
                _stiffness(50.0),
                _diffEffortMax(1.0),
                _saturationRatio(0.9),
                _isInit(false),
                _lastPosError(),
                _lastDivergencePosError(),
                _lastDivergenceEffort(),
                _lastEffort()
        {
        }

        /**
         * Get and set controller parameters
         */
        double getErrorMax() const
        {
            return _posErrorMax;
        }
        void setErrorMax(double posErrorMax)
        {
            if (posErrorMax <= 0.0) {
                throw std::logic_error(
                        "uoe::FIC: Invalid pos error max");
            }
            _posErrorMax = posErrorMax;
        }
        double getForceMax() const
        {
            return _forceMax;
        }
        void setForceMax(double forceMax)
        {
            if (forceMax <= 0.0) {
                throw std::logic_error(
                        "uoe::FIC: Invalid force max");
            }
            _forceMax = forceMax;
        }
        double getStiffness() const
        {
            return _stiffness;
        }
        void setStiffness(double stiffness)
        {
            if (stiffness <= 0.0) {
                throw std::logic_error(
                        "uoe::FIC: Invalid stiffness");
            }
            _stiffness = stiffness;
        }
        double getDiffErrorMax() const
        {
            return _diffEffortMax;
        }
        void setDiffEffortMax(double diffEffortMax)
        {
            if (diffEffortMax <= 0.0) {
                throw std::logic_error(
                        "uoe::FIC: Invalid diff error max");
            }
            _diffEffortMax = diffEffortMax;
        }
        double getSaturationRatio() const
        {
            return _saturationRatio;
        }
        void setSaturationRatio(double saturationRatio)
        {
            if (saturationRatio <= 0.0 || saturationRatio >= 1.0) {
                throw std::logic_error(
                        "uoe::FIC: Invalid saturation ratio");
            }
            _saturationRatio = saturationRatio;
        }

        /**
         * Nonlinear 1d stiffness profile
         */
        double profileNonlinear(double error)
        {
            if (std::fabs(error) < _saturationRatio*_posErrorMax) {
                return _stiffness*error;
            } else {
                double tmpPos = (1.0-_saturationRatio)*_posErrorMax/(2.0*M_PI);
                double tmpForce = _saturationRatio*_stiffness*_posErrorMax;
                return Sign(error)*(
                        (_forceMax-tmpForce)*0.5*(1.0+std::tanh((std::fabs(error)-_posErrorMax)/tmpPos+M_PI))
                        + tmpForce);
            }
        }

        /**
         * Compute and return control effort from given
         * position error and time step in seconds
         */

        //TODO: remove dt
        Eigen::Vector3d control(const Eigen::Vector3d& error, double dt)
        {
            //Initialization
            if (!_isInit) {
                _isInit = true;
                _lastPosError.setZero();
                _lastDivergencePosError = error;
                _lastDivergenceEffort.setZero();
                _lastEffort.setZero();
            }

            Eigen::Vector3d effort = Eigen::Vector3d::Zero();
            for (int i=0;i<error.size();i++) {
                if (std::fabs(error(i)) > std::fabs(_lastPosError(i))) {
                    //Divergence phase
                    effort(i) = profileNonlinear(error(i));
                    _lastDivergencePosError(i) = error(i);
                    _lastDivergenceEffort(i) = effort(i);
                } else {
                    //Convergence phase
                    if (
                            std::fabs(_lastDivergencePosError(i)) < std::fabs(error(i)) ||
                            !IsSameSign(_lastDivergencePosError(i), error(i))
                            ) {
                        _lastDivergencePosError(i) = error(i);
                    }
                    double tmpPosMid = _lastDivergencePosError(i)/2.0;
                    double tmpGainOut = std::fabs(_lastDivergenceEffort(i))/std::max(std::fabs(tmpPosMid), 1e-5);
                    effort(i) = tmpGainOut*(error(i)-tmpPosMid);
                }

                //Clamp effort variation
                if (effort(i) > _lastEffort(i)+dt*_diffEffortMax) {
                    effort(i) = _lastEffort(i)+dt*_diffEffortMax;
                }
                if (effort(i) < _lastEffort(i)-dt*_diffEffortMax) {
                    effort(i) = _lastEffort(i)-dt*_diffEffortMax;
                }
            }

            //Update state
            _lastPosError = error;
            _lastEffort = effort;
            return effort;
        }

    private:

        /**
         * Controller parameters
         */
        double _posErrorMax;
        double _forceMax;
        double _stiffness;
        double _diffEffortMax;
        double _saturationRatio;

        /**
         * Controller states
         */
        bool _isInit;
        Eigen::Vector3d _lastPosError;
        Eigen::Vector3d _lastDivergencePosError;
        Eigen::Vector3d _lastDivergenceEffort;
        Eigen::Vector3d _lastEffort;
    };

/**
 * FICPlanner
 *
 * Implement 3d Fractal Impedance Control Planner
 */
    class FICPlanner
    {
    public:

        /**
         * Default initialization
         */
        FICPlanner() :
                _stiffness(15790.0),
                _velDesired(0.2),
                _accMax(5000.0),
                _dampingRatio(0.005),
                _frequency(20.0),
                _isInit(false),
                _lastPosError(),
                _lastDivergencePosError(),
                _lastPosDesired(),
                _posPlanner(),
                _velPlanner(),
                _accPlanner(),
                _stateVelMax(0.0),
                _stateAccMax()
        {
        }

        /**
         * Get and set planner parameters
         */
        double getStiffness() const
        {
            return _stiffness;
        }
        void setStiffness(double stiffness)
        {
            if (stiffness <= 0.0) {
                throw std::logic_error(
                        "uoe::FIC: Invalid stiffness");
            }
            _stiffness = stiffness;
        }
        double getVelDesired() const
        {
            return _velDesired;
        }
        void setVelDesired(double velDesired)
        {
            if (velDesired <= 0.0) {
                throw std::logic_error(
                        "uoe::FIC: Invalid vel desired");
            }
            _velDesired = velDesired;
        }
        double getAccMax() const
        {
            return _accMax;
        }
        void setAccMax(double accMax)
        {
            if (accMax <= 0.0) {
                throw std::logic_error(
                        "uoe::FIC: Invalid acc max");
            }
            _accMax = accMax;
        }
        double getDampingRatio() const
        {
            return _dampingRatio;
        }
        void setDampingRatio(double damppingRatio)
        {
            if (damppingRatio <= 0.0) {
                throw std::logic_error(
                        "uoe::FIC: Invalid damping ratio");
            }
            _dampingRatio = damppingRatio;
        }
        double getFrequency() const
        {
            return _frequency;
        }
        void setFrequency(double frequency)
        {
            if (frequency <= 0.0) {
                throw std::logic_error(
                        "uoe::FIC: Invalid frequency");
            }
            _frequency = frequency;
        }

        /**
         * Return internal position, velocity and acceleration state
         */
        const Eigen::Vector3d& getPos() const
        {
            return _posPlanner;
        }
        const Eigen::Vector3d& getVel() const
        {
            return _velPlanner;
        }
        const Eigen::Vector3d& getAcc() const
        {
            return _accPlanner;
        }

        /**
         * Reset planner internal integrated
         * position and velocity
         */
        void resetState(const Eigen::Vector3d& pos)
        {
            _posPlanner = pos;
            _velPlanner.setZero();
            _accPlanner.setZero();
            _isInit = false;
        }

        /**
         * Linear 1d stiffness profile
         */
        double profileLinear(double error, double effortMax)
        {
            double effort = _stiffness*error;
            if (std::fabs(effort) > effortMax) {
                effort = Sign(error)*effortMax;
            }
            return effort;
        }

        /**
         * Compute and return planned desired position from given
         * target position and time step in seconds
         */
        Eigen::Vector3d plan(const Eigen::Vector3d& posDesired, double dt)
        {
            Eigen::Vector3d error = posDesired - _posPlanner;

            //Initialization
            if (!_isInit) {
                _isInit = true;
                _lastPosError.setZero();
                _lastDivergencePosError = error;
                _lastPosDesired = _posPlanner;
                _stateVelMax = 0.0;
                _stateAccMax.setZero();
            }

            //Update maximum acceleration
            //Natural Frequency (desired planner bandpass)
            double omega_n = _frequency*2.0*M_PI;
            //Damping Coefficient
            double gainDamping = _dampingRatio*(2.0*omega_n);
            if ((posDesired-_lastPosDesired).norm() > 1e-6) {
                Eigen::Vector3d tmpError = (posDesired-_posPlanner).cwiseAbs();
                if (tmpError.norm() > 1e-6) {
                    //Clamp the error to the 0.001mm ball
                    tmpError = std::max(0.001, tmpError.norm())*tmpError.normalized();
                    //Params
                    double vel_n = std::min(_velDesired, omega_n*tmpError.norm());
                    _stateVelMax = 1.595*vel_n;
                    double vel_max = (_stateVelMax*_stateVelMax)/tmpError.norm();
                    for (size_t i=0;i<3;i++) {
                        _stateAccMax(i) = std::min(2.0*vel_max*tmpError(i)/tmpError.norm(), _accMax);
                    }
                }
            }

            for (int i=0;i<error.size();i++) {
                if (std::fabs(error(i)) > std::fabs(_lastPosError(i))) {
                    //Divergence phase
                    _accPlanner(i) = profileLinear(error(i), _stateAccMax(i));
                    _lastDivergencePosError(i) = error(i);
                } else {
                    //Convergence phase
                    if (
                            std::fabs(_lastDivergencePosError(i)) < std::fabs(error(i)) ||
                            !IsSameSign(_lastDivergencePosError(i), error(i))
                            ) {
                        _lastDivergencePosError(i) = error(i);
                    }
                    double tmpAcc = profileLinear(_lastDivergencePosError(i), _stateAccMax(i));
                    double tmpPosMid = _lastDivergencePosError(i)/2.0;
                    double tmpGainOut = std::fabs(tmpAcc)/std::max(std::fabs(tmpPosMid), 1e-5);
                    _accPlanner(i) = tmpGainOut*(error(i)-tmpPosMid);
                }
            }

            //Update state
            _lastPosError = error;
            _lastPosDesired = posDesired;

            //Acceleration damping
            _accPlanner -= gainDamping*_velPlanner;
            //Integration
            _velPlanner += dt*_accPlanner;
//            _velPlanner = ClampVectorNorm(_velPlanner, _stateVelMax);
            _posPlanner += dt*_velPlanner;
            return _posPlanner;
        }


    private:

        /**
         * Planner parameter
         */
        double _stiffness;
        double _velDesired;
        double _accMax;
        double _dampingRatio;
        double _frequency;

        /**
         * Planner states
         */
        bool _isInit;
        Eigen::Vector3d _lastPosError;
        Eigen::Vector3d _lastDivergencePosError;
        Eigen::Vector3d _lastPosDesired;
        Eigen::Vector3d _posPlanner;
        Eigen::Vector3d _velPlanner;
        Eigen::Vector3d _accPlanner;
        double _stateVelMax;
        Eigen::Vector3d _stateAccMax;
    };

}

#endif //OPTCONTROL_MUJOCO_FIC_H