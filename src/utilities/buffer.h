
#ifndef OPTCONTROL_MUJOCO_BUFFER_H
#define OPTCONTROL_MUJOCO_BUFFER_H

#include "mujoco.h"
#include "eigen3/Eigen/Dense"
#include "../parameters/simulation_params.h"
#include "generic_utils.h"
#include "buffer_utils.h"

using namespace SimulationParameters;


template<typename T>
struct GenericBuffer
{
    explicit GenericBuffer(typename RawType<T>::scalar* ptr) : begin(ptr){};
    using type = T;
    using scalar = typename RawType<T>::scalar;
    static constexpr const unsigned int size = RawType<T>::size;
    void update(typename RawType<T>::scalar* input) {begin = input;}
    typename RawType<T>::scalar* get_begin(){return begin;}
    unsigned int get_size(){return size;}
private:
    typename RawType<T>::scalar* begin;
};



template<typename T, typename BufferType>
class BaseBuffer
{
public:
    void add_buffer_and_file(const FastPair<BufferType*, std::fstream*>& buffer_file_pair)
    {
        T& underlying = static_cast<T&>(*this);
        underlying.add_buffer_file_pair(buffer_file_pair);
    }

    void push_buffer()
    {
        T& underlying = static_cast<T&>(*this);
        underlying.push_buffers();
    }

    void save_buffer()
    {
        T& underlying = static_cast<T&>(*this);
        underlying.save_data();
    }
};


template<typename BufferType>
class DataBuffer : public BaseBuffer<DataBuffer<BufferType>, BufferType>
{
    friend class BaseBuffer<DataBuffer<BufferType>, BufferType>;

    void add_buffer_file_pair(const FastPair<BufferType*, std::fstream*>& buffer_file_pair)
    {
        typename BufferType::type temp;
        memcpy(temp.data(), buffer_file_pair.first->get_begin(), BufferType::size);
        m_buffer_file_pair.template emplace_back(buffer_file_pair);
        m_buffer_vec.template emplace_back().template emplace_back(std::move(temp));

    }

    void push_buffers()
    {

        for(auto buffer = 0; buffer < m_buffer_file_pair.size(); ++buffer )
        {
            typename BufferType::type temp;
            memcpy(temp.data(), m_buffer_file_pair[buffer].first->get_begin(), BufferType::size);
            m_buffer_vec[buffer].emplace_back(std::move(temp));
        }
    }


    void save_data()
    {

        for(auto buffer = 0; buffer < m_buffer_vec.size(); ++buffer)
        {
            BufferUtilities::save_to_file(m_buffer_file_pair[buffer].second, m_buffer_vec[buffer]);
        }

    }

    std::vector<std::vector<typename BufferType::type>> m_buffer_vec;
    std::vector<FastPair<BufferType*, std::fstream*>> m_buffer_file_pair;
};


template<typename BufferType>
class DummyBuffer : public BaseBuffer<DataBuffer<BufferType>, BufferType>
{
    friend class BaseBuffer<DataBuffer<BufferType>, BufferType> ;
    void add_buffer_file_pair(const mjData *data, BufferType& buffer)
    {

    }


    void save_data(std::fstream& pos_file, std::fstream& vel_file, std::fstream& ctrl_file)
    {

    }
};

#endif //OPTCONTROL_MUJOCO_BUFFER_H
