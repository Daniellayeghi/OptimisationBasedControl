#ifndef OPTCONTROL_MUJOCO_ZMQ_UTILS_H
#define OPTCONTROL_MUJOCO_ZMQ_UTILS_H
#include <bitset>
#include <vector>
#include <array>
#include <zmq.h>
#include "Eigen/src/Core/IO.h"


template<typename T>
struct Buffer{
    explicit Buffer(const char buffer_id ='\0'): m_buffer_id(buffer_id){m_buffer.assign(m_size, '\0');};
    const char m_buffer_id = '\0';
    const int m_size = sizeof(T) + sizeof(m_buffer_id);
    using buffer_array = std::vector<char>;
    void update(const T input, const bool check) {memcpy(m_buffer.data(), input, m_size); m_buffer[m_size - 1] = m_buffer_id;}
    [[nodiscard]] const buffer_array& get() const {return m_buffer;}
private:
    buffer_array m_buffer;
};


template<typename T_ptr, typename T_id>
struct BufferParams
{
    const T_ptr* m_begin;
    const T_ptr* m_end;
    const T_id  m_id;
    const unsigned int m_data_size = m_end - m_begin;
    const unsigned int m_id_size = sizeof(m_id);
    const unsigned int m_byte_size = m_data_size * sizeof(T_ptr) + m_id_size;
};


template<typename T_ptr, typename T_id>
struct SimpleBuffer
{
    using buffer_params_t = std::vector<BufferParams<T_ptr, T_id>>;
    using main_buffer_t = std::vector<char>;


    explicit SimpleBuffer(buffer_params_t& buff_params):
            m_buffer_params(buff_params)
    {
        unsigned int total_size = 0;
        std::for_each(m_buffer_params.begin(), m_buffer_params.end(), [&](const auto& elem)
        {total_size += elem.m_byte_size;});

        m_buffer.assign(total_size, '\0');
    };


    void update_buffer()
    {
        auto ptr_id = m_buffer.data();
        for(auto idx = 0; idx < m_buffer_params.size(); ++idx)
        {
            const auto ptr_data = ptr_id + m_buffer_params[idx].m_id_size;
            memcpy(ptr_data, m_buffer_params[idx].m_begin, m_buffer_params[idx].m_data_size * sizeof(T_ptr));
            memcpy(ptr_id, &m_buffer_params[idx].m_id, m_buffer_params[idx].m_id_size * sizeof(T_id));
            ptr_id = ptr_data + m_buffer_params[idx].m_data_size * sizeof(T_ptr);
        }
    };


    auto get_buffer() {return m_buffer.data();}
    auto get_buffer_size(){return m_buffer.size();};

private:
    const buffer_params_t m_buffer_params;
    main_buffer_t  m_buffer;
};


template<typename T>
class ZMQUBuffer
{
public:
    // https://stackoverflow.com/questions/34439284/c-function-returning-reference-to-array
    // Buffer type returns by reference and buffers takes in a copy which is emplace back;

    ZMQUBuffer(int socket_type, std::string addr) : m_requester(zmq_socket(m_context, socket_type))
    {
        zmq_connect (m_requester, addr.c_str());
    }


    void push_buffer(Buffer<T>* buffer_type)
    {
        buffers.template emplace_back(buffer_type);
    }


    void send_buffers()
    {
        for(const Buffer<T>* buffer : buffers)
            zmq_send(m_requester, buffer->get().data(), buffer->m_size, 0);
    }


    void send_buffer(const char* buffer, const unsigned int size)
    {
        zmq_send(m_requester, buffer, size, 0);
    }


    bool rec_buffers()
    {
        char rec_r = '\0';
        zmq_recv(m_requester, &rec_r, sizeof(rec_r), 0);
        return (rec_r != '\0');
    }


    void buffer_wait_for_res()
    {
        while(not rec_buffers()){}
    }


    ~ZMQUBuffer()
    {
        zmq_close (m_requester);
        zmq_ctx_destroy (m_context);
    }

private:
    std::vector<Buffer<T>*> buffers;
    void* m_context = zmq_ctx_new();
    void* m_requester;
};


#endif //OPTCONTROL_MUJOCO_ZMQ_UTILS_H
