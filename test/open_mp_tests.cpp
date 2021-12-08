
#include "gtest/gtest.h"
#include "../src/utilities/generic_utils.h"
#include <omp.h>
#include <iostream>
#include <complex>

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
    for (int iter = 0; iter < duration_step; ++iter)
        sum += func((iter + 0.5) * step);

    approx_pi = step * sum;
    ASSERT_NEAR(approx_pi, 3.14, 0.01);
}


TEST_F(OpenMPTests, Parallel_Integration_1)
{
    constexpr const long duration_step = 1e5;
    constexpr const int nthreads_req = 4;
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
            for (int i=id; i < duration_step; i= i + nthrds)
                sum[id] += func_local((i+0.5)*step);
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
    constexpr const int nthreads_req = 4;
    double step = 1.0/duration_step, approx_pi;
    // This division has to return an integer otherwise round and set the limit for las thread to be the "rest"
    // of the segments this is the same as static schedule without chunk size
    static long segments = duration_step / nthreads_req;
    omp_set_num_threads(nthreads_req);
    std::array<double, nthreads_req> sum_per_thread = {};
    GenericUtils::TimeBench timer("Parallel_Integration_2");
#pragma omp parallel default(none) shared(segments, sum_per_thread, step)
    {
        auto func_local = [](double x){return 4/(1+x*x);};
        int id = omp_get_thread_num();
        for (int iter = id*segments; iter < (id+1) * segments; ++iter)
            sum_per_thread[id] += func_local((iter + 0.5) * step);
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
            sum_per_thread[id][0] += func_local((iter + 0.5) * step);
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
        for (int i=id; i < duration_step; i= i + nthrds)
            sum_per_thread[id][0] += func_local((i+0.5)*step);
    }

    for(int i=0;i<sum_per_thread.size();i++) pi += sum_per_thread[i][0] * step;
    ASSERT_NEAR(pi, 3.14, 0.01);
}


TEST_F(OpenMPTests, Parallel_Integration_Critical_Sum_1)
{
    constexpr const long duration_step = 1e5;
    constexpr const int nthreads_req = 4;
    double step = 1.0/(double) duration_step, pi = 0;
    int nthreads_given = 0;
    omp_set_num_threads(nthreads_req);
    GenericUtils::TimeBench timer("Parallel_Integration_Critical_Sum_1");
    // The segment insisde the pragma is a program for each thread with any declaration as data per thread
    // Any decleration outside requires protection if writing or reading to
#pragma omp parallel default(none) shared(nthreads_given, step, pi)
    {
        // sum is local to each thread
        double sum = 0;
        auto func_local = [](double x){return 4/(1+x*x);};
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads();
        if (id == 0) nthreads_given = nthrds;
        for (int i=id; i < duration_step; i= i + nthrds)
            sum += func_local((i+0.5)*step);

#pragma omp critical
        pi += sum *step;
    }
    ASSERT_NEAR(pi, 3.14, 0.01);
}


TEST_F(OpenMPTests, Parallel_Integration_Critical_Sum_2)
{

    constexpr const long duration_step = 1e5;
    constexpr const int nthreads_req = 4;
    double step = 1.0/duration_step, approx_pi = 0;
    static long segments = duration_step / nthreads_req;
    omp_set_num_threads(nthreads_req);
    GenericUtils::TimeBench timer("Parallel_Integration_Critical_Sum_2");
#pragma omp parallel default(none) shared(segments, approx_pi, step)
    {
        // sum is local to each thread
        double sum = 0;
        auto func_local = [](double x){return 4/(1+x*x);};
        int id = omp_get_thread_num();
        for (int iter = id*segments; iter < (id+1) * segments; ++iter)
            sum += func_local((iter + 0.5) * step);
#pragma omp critical
        approx_pi += sum *step;
    }
    ASSERT_NEAR(approx_pi, 3.14, 0.01);
}


TEST_F(OpenMPTests, Parallel_Integration_Loop_Reduction_1)
{

    // duration step cannot be const since it is determined by openMP for the reduction
    long duration_step = 1e5;
    double step = 1.0/duration_step, approx_pi = 0;
    GenericUtils::TimeBench timer("Parallel_Integration_Loop_Reduction_1");
    // sum is local to each thread
    double sum = 0;
    auto func = [](double x){return 4/(1+x*x);};
#pragma omp parallel for reduction (+:sum) default(none) shared(func, step, duration_step)
    for (int iter = 0; iter < duration_step; ++iter)
        sum += func((iter + 0.5) * step);

    approx_pi += sum *step;
    ASSERT_NEAR(approx_pi, 3.14, 0.01);
}


TEST_F(OpenMPTests, Parallel_Integration_Loop_Reduction_2)
{
    // This is the same construct as above but it is clear that sum is a shared variable with local copies when you
    // define parallel for
    long duration_step = 1e5;
    double step = 1.0/duration_step, approx_pi = 0;
    double sum = 0;
    GenericUtils::TimeBench timer("Parallel_Integration_Loop_Reduction_2");
#pragma omp parallel default(none) shared(approx_pi, step, duration_step, sum)
    {
        // sum is local to each thread
        auto func_local = [](double x){return 4/(1+x*x);};
#pragma omp for reduction (+:sum)
        for (int iter = 0; iter < duration_step; ++iter)
            sum += func_local((iter + 0.5) * step);
    }
    approx_pi += sum *step;
    ASSERT_NEAR(approx_pi, 3.14, 0.01);
}


TEST_F(OpenMPTests, Mandelbrot_Area_Vanilla)
{
    int n_points = 1e3;
    constexpr const int max_iter = 1e3;
    constexpr const double eps = 1e-5;
    int outer_iteration = 0;

    using complex_num = std::complex<double>;
    complex_num c_num {};

    auto test_func = [&](const complex_num& area, int& outer_it) {
        complex_num temp_com = area;
        double temp;

        for (int iter=0; iter<max_iter; ++iter)
        {
            temp = (temp_com.real()*temp_com.real())-(temp_com.imag()*temp_com.imag())+area.real();
            temp_com.imag(temp_com.real()*temp_com.imag()*2+area.imag());
            temp_com.real(temp);
            if ((temp_com.real()*temp_com.real()+temp_com.imag()*temp_com.imag())>4.0) {
#pragma omp atomic
                outer_it++;
                break;
            }
        }
            };

    // j is defined here since the omp parallel directive only applies to the first for loop and make the second parallel
    // the index has to be private to each thread. However limits have to be shared between threads for omp to allocate
    // chunks. firstprivate eps is because private variable are not initialised firstprivate will initialise it to the
    // variable shadowing it outside the scope
    GenericUtils::TimeBench timer("Mandelbrot_Area_Vanilla");
    int j;
#pragma omp parallel for num_threads(16) default(none) private(j, c_num) shared(n_points, test_func, outer_iteration) firstprivate(eps)
    for (int i = 0; i < n_points; ++i)
        for (j = 0; j < n_points; ++j)
        {
            c_num.real(-2.0+2.5*(double)(i)/(double)(n_points)+eps);
            c_num.imag(1.125*(double)(j)/(double)(n_points)+eps);
            test_func(c_num, outer_iteration);
        }

    double final_area=2.0*2.5*1.125*(double)(n_points*n_points-outer_iteration)/(double)(n_points*n_points);
    double error = final_area/(double)n_points;
    ASSERT_NEAR(final_area, 1.510659, error );

}


TEST_F(OpenMPTests, Mandelbrot_Area_Lower_Lock_Rate)
{
    int n_points = 1e3;
    constexpr const int max_iter = 1e3;
    int outer_iteration = 0;

    using complex_num = std::complex<double>;
    complex_num c_num {};

    auto test_func = [&](const complex_num& area) {
        complex_num temp_com = area;
        double temp;

        for (int iter=0; iter<max_iter; ++iter)
        {
            temp = (temp_com.real()*temp_com.real())-(temp_com.imag()*temp_com.imag())+area.real();
            temp_com.imag(temp_com.real()*temp_com.imag()*2+area.imag());
            temp_com.real(temp);
            if ((temp_com.real()*temp_com.real()+temp_com.imag()*temp_com.imag())>4.0) {
                return 1;
            }
        }
        return 0;
    };

    GenericUtils::TimeBench timer("Mandelbrot_Area_Lower_Lock_Rate");
    int j;
#pragma omp parallel for num_threads(16) reduction(+:outer_iteration) default(none) private(j, c_num) shared(n_points, test_func)
    for (int i = 0; i < n_points; ++i)
        for (j = 0; j < n_points; ++j)
        {
            constexpr const double eps = 1e-5;
            c_num.real(-2.0+2.5*(double)(i)/(double)(n_points)+eps);
            c_num.imag(1.125*(double)(j)/(double)(n_points)+eps);
            outer_iteration += test_func(c_num);
        }

    double final_area=2.0*2.5*1.125*(double)(n_points*n_points-outer_iteration)/(double)(n_points*n_points);
    double error = final_area/(double)n_points;
    ASSERT_NEAR(final_area, 1.510659, error);

}



