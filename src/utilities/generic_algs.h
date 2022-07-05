#ifndef OPTCONTROL_MUJOCO_GENERIC_ALGS_H
#define OPTCONTROL_MUJOCO_GENERIC_ALGS_H


namespace GenericMap
{
    template <typename T_in, typename T_out>
    struct OpFuncs
    {
        using multi_op = T_in(*)(T_in, T_in);
        using single_op = T_out(*)(T_in);
    };


    template<typename T_in, typename T_out>
    void consecutive_map(T_in* src, unsigned long last_idx, typename OpFuncs<T_in, T_out>::multi_op f)
    {
        /* [i] = [i] OpFuncs [i+1] OpFuncs ... OpFuncs [N] assuming max is N */
        for (unsigned long idx = 0; idx < last_idx; ++idx)
            for(auto i = idx+1; i < last_idx; ++i)
                src[idx] = f(src[idx], src[i]);
    }


    template<typename T_in, typename T_out>
    void  singular_map(T_in* src, unsigned long last_idx, typename OpFuncs<T_in, T_out>::single_op f)
    {
        /* [i] = OpFuncs([i]) [N] assuming max is N */
        for (unsigned long idx = 0; idx < last_idx; ++idx)
            src[idx] = f(src[idx]);
    }
}

#endif //OPTCONTROL_MUJOCO_GENERIC_ALGS_H
