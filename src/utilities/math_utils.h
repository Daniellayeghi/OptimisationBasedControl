#ifndef OPTCONTROL_MUJOCO_MATH_UTILS_H
#define OPTCONTROL_MUJOCO_MATH_UTILS_H

#include <random>
#include "Eigen/Core"

namespace MathUtils::Rand
{
    template<typename T, size_t S>
    void random_iid_data(T *result, const T *bound_l, const T *bound_h) {
        using namespace std;
        random_device r;
        default_random_engine generator(r());
        for (auto dim = 0; dim < S; ++dim) {
            uniform_real_distribution<T> distribution(bound_l[dim], bound_h[dim]);
            result[dim] = distribution(generator);
        }
    }


    template<typename T, size_t S>
    void random_iid_data_sym_bound(T *result, const T *positive_bound) {
        using namespace std;
        random_device r;
        default_random_engine generator(r());
        for (auto dim = 0; dim < S; ++dim) {
            uniform_real_distribution<T> distribution(-positive_bound[dim], positive_bound[dim]);
            result[dim] = distribution(generator);
        }
    }


    template<typename T, size_t S>
    void random_iid_data_const_bound(T *result, const T positive_bound) {
        using namespace std;
        default_random_engine generator{static_cast<long unsigned int>(time(0))};
        for (auto dim = 0; dim < S; ++dim) {
            uniform_real_distribution<T> distribution(-positive_bound, positive_bound);
            result[dim] = distribution(generator);
        }
    }
}


namespace MathUtils::Coord
{
    template<typename T>
    struct Cart
    {
        T x, y, z;
        CartVector as_vec() {CartVector res; res << x, y, z; return res;}
    };


    template<typename T>
    struct Spherical
    {
        T theta, phi, r;
        Cart<T> to_cart(){ return {r * sin(theta) * cos(phi), r * sin(phi) * sin(theta), r*cos(theta)};}
        SphVector as_vec(){SphVector res; res << theta, phi, r; return res;}
    };
}

#endif //OPTCONTROL_MUJOCO_MATH_UTILS_H
