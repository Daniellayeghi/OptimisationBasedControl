#include "basic_math.h"
#include <cmath>

namespace BasicMath
{
    double wrap_to_max(double x, double max)
    {
        return fmod(max + fmod(x, max), max);
    }


    double wrap_to_min_max(double x, double min, double max)
    {
        if (x == max) return max;
        return min + wrap_to_max(x - min, max - min);
    }

    double wrap_to_2pi(double x)
    {
        //Comment wrapping for cartpole uncomment for acrobot
//        auto angle = (2*M_PI - x) < 1e-7 ?  0 :  x - 2 * M_PI * floor( x / (2 * M_PI));
//        return angle;
        return x;
    }
}