
#ifndef OPTCONTROL_MUJOCO_BUFFER_H
#define OPTCONTROL_MUJOCO_BUFFER_H

#include "mujoco.h"
#include "eigen3/Eigen/Dense"
#include "../parameters/simulation_params.h"
#include "buffer_utils.h"

using namespace SimulationParameters;


template<typename T>
struct GenericBuffer
{
    using scalar = typename T::scalar;
    unsigned int buff_size = sizeof (T);
    void update(const T input) {begin = input;}
    typename T::scalar* get_begin(){return begin;}
    unsigned int get_size(){return sizeof(T);}
private:
    typename T::scalar* begin;
};



template<typename T, typename BufferType>
class BaseBuffer
{
public:
    void fill_buffer(const mjData *data, BufferType& buffer)
    {
        T& underlying = static_cast<T&>(*this);
        underlying.fill(data, buffer);
    }


    void save_buffer(std::fstream& pos_file, std::fstream& vel_file, std::fstream& ctrl_file)
    {
        T& underlying = static_cast<T&>(*this);
        underlying.save_data(pos_file, vel_file, ctrl_file);
    }
};


template<typename BufferType>
class DataBuffer : public BaseBuffer<DataBuffer<BufferType>, BufferType>
{
    friend class BaseBuffer<DataBuffer<BufferType>, BufferType>;

    void fill(const mjData *data, BufferType& buffer)
    {
        using namespace Eigen;
        Eigen::Map<RowVectorXd> pos(data->qpos, n_jpos);
        Eigen::Map<RowVectorXd> vel(data->qpos, n_jvel);
        Eigen::Map<RowVectorXd> ctrl(data->ctrl, n_ctrl);

        auto& ref = buffer_vec.back();
        memcpy(ref, buffer.get_begin(), buffer.get_size());
        pos_buffer.emplace_back();
        vel_buffer.emplace_back(vel);
        ctrl_buffer.emplace_back(ctrl);
    }


    void save_data(std::fstream& pos_file, std::fstream& vel_file, std::fstream& ctrl_file)
    {
        BufferUtilities::save_to_file(pos_file, pos_buffer);
        BufferUtilities::save_to_file(vel_file, vel_buffer);
        BufferUtilities::save_to_file(ctrl_file, ctrl_buffer);
    }

    std::vector<typename BufferType::scalar [BufferType::buff_size]> buffer_vec;
    std::vector<Eigen::Matrix<double, n_jpos, 1>> pos_buffer;
    std::vector<Eigen::Matrix<double, n_jvel, 1>> vel_buffer;
    std::vector<Eigen::Matrix<double, n_ctrl, 1>> ctrl_buffer;
};


template<typename BufferType>
class DummyBuffer : public BaseBuffer<DataBuffer<BufferType>, BufferType>
{
    friend class BaseBuffer<DataBuffer<BufferType>, BufferType> ;
    void fill(const mjData *data, BufferType& buffer)
    {

    }


    void save_data(std::fstream& pos_file, std::fstream& vel_file, std::fstream& ctrl_file)
    {

    }
};

#endif //OPTCONTROL_MUJOCO_BUFFER_H
