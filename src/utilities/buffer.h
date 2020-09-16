
#ifndef OPTCONTROL_MUJOCO_BUFFER_H
#define OPTCONTROL_MUJOCO_BUFFER_H

#include "mujoco.h"
#include "eigen3/Eigen/Dense"
#include "../parameters/simulation_params.h"
#include "buffer_utils.h"

using namespace SimulationParameters;

template<typename T>
class BaseBuffer
{
public:
    void fill_buffer(const mjData *data)
    {
        T& underlying = static_cast<T&>(*this);
        underlying.fill(data);
    }


    void save_buffer(std::fstream& pos_file, std::fstream& vel_file, std::fstream& ctrl_file)
    {
        T& underlying = static_cast<T&>(*this);
        underlying.save_data(pos_file, vel_file, ctrl_file);
    }
};


class DataBuffer : public BaseBuffer<DataBuffer>
{
    friend class BaseBuffer<DataBuffer>;

    void fill(const mjData *data)
    {
        Eigen::Map<RowVectorXd> pos(data->qpos, n_jpos);
        Eigen::Map<RowVectorXd> vel(data->qpos, n_jvel);
        Eigen::Map<RowVectorXd> ctrl(data->ctrl, n_ctrl);

        pos_buffer.emplace_back(pos);
        vel_buffer.emplace_back(vel);
        ctrl_buffer.emplace_back(ctrl);
    }


    void save_data(std::fstream& pos_file, std::fstream& vel_file, std::fstream& ctrl_file)
    {
        BufferUtilities::save_to_file(pos_file, pos_buffer);
        BufferUtilities::save_to_file(vel_file, vel_buffer);
        BufferUtilities::save_to_file(ctrl_file, ctrl_buffer);
    }


    std::vector<Eigen::Matrix<double, n_jpos, 1>> pos_buffer;
    std::vector<Eigen::Matrix<double, n_jvel, 1>> vel_buffer;
    std::vector<Eigen::Matrix<double, n_ctrl, 1>> ctrl_buffer;
};


class DummyBuffer : public BaseBuffer<DummyBuffer>
{
    friend class BaseBuffer<DummyBuffer> ;
    void fill(const mjData *data)
    {

    }


    void save_data(std::fstream& pos_file, std::fstream& vel_file, std::fstream& ctrl_file)
    {

    }
};

#endif //OPTCONTROL_MUJOCO_BUFFER_H
