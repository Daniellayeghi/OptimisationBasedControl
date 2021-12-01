
#include "gtest/gtest.h"
#include "../src/utilities/generic_utils.h"
#include <omp.h>
#include <iostream>

class OpenMPTests : public testing::Test {

public:
    void SetUp()
    {

    }

    void TearDown()
    {

    }

};


TEST_F(OpenMPTests, Sequential_Integration)
{
    // Integrate 4/(1+x^2) from [0, 1] this should equal PI.
    constexpr const long duration_step = 1e5;
    double step = 1.0/duration_step, sum = 0, approx_pi;
    auto func = [](double x){return 4/(1+x*x);};
    GenericUtils::TimeBench timer("Sequentional");
    for (int iter = 0; iter < duration_step; ++iter) {
        sum += func((iter + 0.5) * step);
    }
    approx_pi = step * sum;
    ASSERT_NEAR(approx_pi, 3.14, 0.01);
}


TEST_F(OpenMPTests, Parallel_Integration_1)
{
    constexpr const long duration_step = 1e5;
    constexpr const int nthreads_req = 2;
    double step = 1.0/(double) duration_step, pi = 0;
    int nthreads_given = 0;
    std::array<double, nthreads_req> sum = {};
    omp_set_num_threads(nthreads_req);
    GenericUtils::TimeBench timer("Parallel_Integration_1");
    // The segment insisde the pragma is a program for each thread with any declaration as data per thread
    // Any decleration outside requires protection if writing or reading to
#pragma omp parallel default(none) shared(nthreads_given, sum, step)
        {
            auto func_local = [](double x){return 4/(1+x*x);};
            int id = omp_get_thread_num();
            int nthrds = omp_get_num_threads();
            if (id == 0) nthreads_given = nthrds;
            for (int i=id; i < duration_step; i= i + nthrds) {
                sum[id] += func_local((i+0.5)*step);
            }
        }

    for(int i=0;i<sum.size();i++) pi += sum[i] * step;
    ASSERT_NEAR(pi, 3.14, 0.01);
}


TEST_F(OpenMPTests, Parallel_Integration_2)
{
    // Divide the summation across the threads.
    // NOTE: it is important for the limit of summation for the threads to be seperated other wise the limit can grow
    // within the loop and never end.
    constexpr const long duration_step = 1e5;
    constexpr const int nthreads_req = 2;
    double step = 1.0/duration_step, approx_pi;
    static long segments = duration_step / nthreads_req;
    omp_set_num_threads(nthreads_req);
    std::array<double, nthreads_req> sum_per_thread = {};
    GenericUtils::TimeBench timer("Parallel_Integration_2");
#pragma omp parallel default(none) shared(segments, sum_per_thread, step)
    {
        auto func_local = [](double x){return 4/(1+x*x);};
        int id = omp_get_thread_num();
        for (int iter = id*segments; iter < (id+1) * segments; ++iter)
        {
            sum_per_thread[id] += func_local((iter + 0.5) * step);
        }
    }
    auto arr_sum = std::accumulate(sum_per_thread.begin(), sum_per_thread.end(), 0.0);
    approx_pi = arr_sum * step;
    ASSERT_NEAR(approx_pi, 3.14, 0.01);
}


TEST_F(OpenMPTests, Parallel_Integration_Cache_Line_Padding_1)
{
    // Assuming cache line to be 8 bytes with added padding we can force each thread to access a separate cache line to
    // false sharing
    constexpr const int cache_line_pad = 8;
    constexpr const long duration_step = 1e5;
    constexpr const int nthreads_req = 4;
    double step = 1.0/duration_step, approx_pi = 0;
    static long segments = duration_step / nthreads_req;
    omp_set_num_threads(nthreads_req);
    std::array<std::array<double, cache_line_pad>, nthreads_req> sum_per_thread = {};
    GenericUtils::TimeBench timer("Parallel_Integration_Cache_Line_Padding_1");
#pragma omp parallel default(none) shared(segments, sum_per_thread, step)
    {
        auto func_local = [](double x){return 4/(1+x*x);};
        int id = omp_get_thread_num();
        for (int iter = id*segments; iter < (id+1) * segments; ++iter)
        {
            sum_per_thread[id][0] += func_local((iter + 0.5) * step);
        }
    }

    for(int i = 0; i < nthreads_req; ++i)
         approx_pi += sum_per_thread[i][0];

    approx_pi *= step;
    ASSERT_NEAR(approx_pi, 3.14, 0.01);
}


TEST_F(OpenMPTests, Parallel_Integration_Cache_Line_Padding_2)
{
    // Assuming cache line to be 8 bytes with added padding we can force each thread to access a separate cache line to
    // avoid false sharing
    constexpr const int cache_line_pad = 8;
    constexpr const long duration_step = 1e5;
    constexpr const int nthreads_req = 4;
    double step = 1.0/(double) duration_step, pi = 0;
    int nthreads_given = 0;
    std::array<std::array<double, cache_line_pad>, nthreads_req> sum_per_thread = {};
    omp_set_num_threads(nthreads_req);
    GenericUtils::TimeBench timer("Parallel_Integration_Cache_Line_Padding_2");
    // The segment insisde the pragma is a program for each thread with any declaration as data per thread
    // Any decleration outside requires protection if writing or reading to
#pragma omp parallel default(none) shared(nthreads_given, sum_per_thread, step)
    {
        auto func_local = [](double x){return 4/(1+x*x);};
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads();
        if (id == 0) nthreads_given = nthrds;
        for (int i=id; i < duration_step; i= i + nthrds) {
            sum_per_thread[id][0] += func_local((i+0.5)*step);
        }
    }

    for(int i=0;i<sum_per_thread.size();i++) pi += sum_per_thread[i][0] * step;
    ASSERT_NEAR(pi, 3.14, 0.01);
}