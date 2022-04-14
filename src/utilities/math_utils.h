#ifndef OPTCONTROL_MUJOCO_MATH_UTILS_H
#define OPTCONTROL_MUJOCO_MATH_UTILS_H

#include <random>

namespace MathUtils
{
    template<typename T, size_t S>
    void random_iid_data (T* result, const T* bound_l, const T* bound_h) {
        using namespace std;
        random_device r;
        default_random_engine generator(r());
        for (auto dim = 0; dim < S; ++dim) {
            uniform_real_distribution<T> distribution(bound_l[dim], bound_h[dim]);
            result[dim] = distribution(generator);
        }
    }


    template<typename T, size_t S>
    void random_iid_data_sym_bound (T* result, const T* positive_bound) {
        using namespace std;
        random_device r;
        default_random_engine generator(r());
        for (auto dim = 0; dim < S; ++dim) {
            uniform_real_distribution<T> distribution(-positive_bound[dim], positive_bound[dim]);
            result[dim] = distribution(generator);
        }
    }


    template<typename T, size_t S>
    void random_iid_data_const_bound (T* result, const T positive_bound) {
        using namespace std;
        random_device r;
        default_random_engine generator(r());
        for (auto dim = 0; dim < S; ++dim) {
            uniform_real_distribution<T> distribution(-positive_bound, positive_bound);
            result[dim] = distribution(generator);
        }
    }
}

#endif //OPTCONTROL_MUJOCO_MATH_UTILS_H
