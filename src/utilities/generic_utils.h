#ifndef OPTCONTROL_MUJOCO_GENERIC_UTILS_H
#define OPTCONTROL_MUJOCO_GENERIC_UTILS_H

#include "vector"
#include "array"
#include <numeric>
#include <chrono>
#include <iostream>
#include <utility>

namespace GenericUtils
{
    template<typename T1, typename T2>
    struct FastPair
    {
        T1 first;
        T2 second;
    };


    template<typename T>
    void sg_filter(const std::vector<T>& input, std::vector<T>& result)
    {
        // Filter coefficients and normalisation
        const constexpr std::array<double, 3> filter_coeff {{-3, 12, 17}};
        const double norm = std::accumulate(filter_coeff.begin(), filter_coeff.end(), 0.0);

        // Actual filter
        const auto cleaned = [&](int iter)
                {
            return (filter_coeff[0] * input[iter-2] + filter_coeff[1] * input[iter-1] + filter_coeff[2] * input[iter] +
            filter_coeff[1] * input[iter+1] + filter_coeff[0] * input[iter+2])/norm;
        };

        for(auto iter = filter_coeff.size(); iter < input.size() - filter_coeff.size(); ++iter)
            result[iter] = cleaned(iter);

        // Hard set the first and last 2 elems
        result[result.size()-1] = input[input.size()-1];
        result[result.size()-2] = input[input.size()-2];
        result[0] = input[0];
        result[1] = input[1];
    }


    struct TimeBench
    {
        std::chrono::time_point<std::chrono::steady_clock> m_start, m_end;
        std::chrono::duration<double> duration;
        const std::string m_id;

        TimeBench(std::string id = "") : m_id(std::move(id))
        {
            m_start = std::chrono::steady_clock::now();
        }
    public:
        ~TimeBench()
        {
            m_end = std::chrono::steady_clock::now();
            duration = m_end - m_start;
            std::cout << "["<< m_id << "]" << " Computation took: " << duration.count() * 1000.0 << " ms" << "\n";
        }
    };
}

#endif //OPTCONTROL_MUJOCO_GENERIC_UTILS_H
