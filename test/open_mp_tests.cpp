
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


TEST_F(OpenMPTests, Basic_Test)
{
    // Leads to interleaved statements due to threads running concurrently
	// Forking from master and joining to master happens sequentionally.
	#pragma omp parallel
	{
		std::cout << "Tests 1" << "\n";
		std::cout << "Tests 2" << "\n";
	}
    ASSERT_TRUE(true);
}


TEST_F(OpenMPTests, Parallel_Reigon)
{
    // Any declaration before the parallel section is shared between threads.
    // The number of threads requested may not be equal to the number of threads given by the OS.
    std::array<double, 1000> shared_memory;
    omp_set_num_threads(4);

#pragma omp parallel
    {
        // Variable declaration within the pragma are allocated on the specific thread stack;
        // Threads wait at the end "barrier" for all threads to finish.
        int thread_stack_variable = omp_get_thread_num();
        std::cout << thread_stack_variable << "\n";
    }
    ASSERT_TRUE(true);
}


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
    constexpr const int nthreads_req = 5;
    double step = 1.0/(double) duration_step, pi = 0;
    int nthreads_given = 0;
    std::array<double, nthreads_req> sum; sum.fill(0);
    omp_set_num_threads(nthreads_req);
    GenericUtils::TimeBench timer("Parallel_Integration_1");
#pragma omp parallel
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
    constexpr const int nthreads_req = 5;
    double step = 1.0/duration_step, approx_pi;
    static long segments = duration_step / nthreads_req;
    omp_set_num_threads(nthreads_req);
    std::array<double, nthreads_req> sum_per_thread; sum_per_thread.fill(0);
    GenericUtils::TimeBench timer("Parallel_Integration_2");
#pragma omp parallel
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
