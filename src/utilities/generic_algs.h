#ifndef OPTCONTROL_MUJOCO_GENERIC_ALGS_H
#define OPTCONTROL_MUJOCO_GENERIC_ALGS_H


namespace GenericAlgs
{
    template <typename T_in, typename T_out>
    struct op
    {
        using multi_op = T_in(*)(T_in, T_in);
        using single_op = T_out(*)(T_in);
    };


    template<typename T_in, typename T_out>
    void consecutive_op(T_in* src, unsigned long last_idx, typename op<T_in, T_out>::multi_op f)
    {
        /* [i] = [i] op [i+1] op ... op [N] assuming max is N */
        for (unsigned long idx = 0; idx < last_idx; ++idx)
            for(auto i = idx+1; i < last_idx; ++i)
                src[idx] = f(src[idx], src[i]);
    }


    template<typename T_in, typename T_out>
    void  singular_op(T_in* src, unsigned long last_idx, typename op<T_in, T_out>::single_op f)
    {
        /* [i] = op([i]) [N] assuming max is N */
        for (unsigned long idx = 0; idx < last_idx; ++idx)
            src[idx] = f(src[idx]);
    }
}

#endif //OPTCONTROL_MUJOCO_GENERIC_ALGS_H
