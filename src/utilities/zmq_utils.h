#ifndef OPTCONTROL_MUJOCO_ZMQ_UTILS_H
#define OPTCONTROL_MUJOCO_ZMQ_UTILS_H
#include <bitset>
#include <vector>
#include <array>
#include <zmq.h>

template<typename T>
struct Buffer{
    static const constexpr bool check_bit = true;
    static const constexpr int size = sizeof(T) + ((check_bit) ? sizeof(check_bit) : 0);
    using buffer_array = std::array<char, size>;
    void update(const T input, const bool check) {memcpy(m_buffer.begin(), input, size); m_buffer[size - 1] = check;}
    const buffer_array& get() const {return m_buffer;}
private:
    buffer_array m_buffer;
};


template<typename T>
class ZMQUBuffer
{
public:
    // https://stackoverflow.com/questions/34439284/c-function-returning-reference-to-array
    // Buffer type returns by reference and buffers takes in a copy which is emplace back;

    ZMQUBuffer(int socket_type, std::string addr)
    {
        m_requester = zmq_socket(m_context, socket_type);
        zmq_connect (m_requester, addr.c_str());
    }

    void push_buffer(Buffer<T>* buffer_type)
    {
        buffers.template emplace_back(buffer_type);
    }

    void send_buffers()
    {
        for(const Buffer<T>* buffer : buffers)
            zmq_send(m_requester, buffer->get().data(), buffer->size, 0);
    }

private:
    std::vector<Buffer<T>*> buffers;
    void* m_context = zmq_ctx_new();
    void* m_requester;
};


#endif //OPTCONTROL_MUJOCO_ZMQ_UTILS_H
