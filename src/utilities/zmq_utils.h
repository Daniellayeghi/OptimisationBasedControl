#ifndef OPTCONTROL_MUJOCO_ZMQ_UTILS_H
#define OPTCONTROL_MUJOCO_ZMQ_UTILS_H
#include <bitset>
#include <vector>
#include <array>
#include <zmq.h>

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


template<typename T_Buffer, typename T_Id>
struct SimpleBuffer
{
    using buffer_array = std::vector<char>;
    using begin_vecs = std::vector<T_Buffer*>;
    using size_vecs = std::vector<unsigned int>;
    using id_vecs = std::vector<T_Id>;

    SimpleBuffer(begin_vecs& buff_begins, size_vecs& buff_sizes, id_vecs& buff_ids):
            m_begins(buff_begins), m_sizes(buff_sizes), m_ids(buff_ids)
    {
        auto total_size =
                std::accumulate(buff_sizes.begin(), buff_sizes.end(), 0) + m_ids.size() * sizeof(T_Id);

        m_buffer.assign(total_size, '\0');
    };


    void update_buffer()
    {
        for(auto idx = 0; idx < m_begins.size(); ++idx)
        {
            const auto ptr_id = m_buffer.data() + idx * (m_sizes[idx] + sizeof(m_ids[idx]));
            const auto ptr_data = ptr_id + sizeof(m_ids[idx]);
            memcpy(ptr_data, m_begins[idx], m_sizes[idx]);
            memcpy(ptr_id, &m_ids[idx], sizeof(m_ids[idx]));
        }
        std::for_each(m_buffer.begin(), m_buffer.end(), [](const auto& elem){std::cout << elem;});
        std::cout << "\n";
    };


    auto get_buffer() {return m_buffer.data();}


    auto get_buffer_size(){return m_buffer.size();};

private:
    buffer_array m_buffer;
    const begin_vecs& m_begins;
    const size_vecs& m_sizes;
    const id_vecs& m_ids;
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
