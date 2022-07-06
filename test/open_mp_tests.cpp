
#include <omp.h>
#include <iostream>
#include <complex>
#include <functional>
#include <memory>
#include <thread>
#include "gtest/gtest.h"
#include "../src/utilities/generic_utils.h"
#include "../src/utilities/eigen_norm_dist.h"
#include "Eigen/Core"
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


//TEST_F(OpenMPTests, Linked_List_Access)
//{
//    constexpr const int fs = 38, n = 5;
//    struct Node
//    {
//        int data, fib_data; std::shared_ptr<Node> next{nullptr};
//        Node(int data, int fib_data) : data{data}, fib_data(fib_data), next{nullptr} {}
//    };
//
//    std::function<int(int)> fib;
//    fib = [&fib](int n)
//    {
//        int x, y;
//        if (n < 2) { return (n); }
//        else {
//            x = fib(n - 1);
//            y = fib(n - 2);
//            return (x + y);
//        }
//    };
//
//    auto process_work = [&fib](std::shared_ptr<Node> &p)
//    {
//        int n = p->data;
//        p->fib_data = fib(n);
//    };
//
//    std::vector<std::shared_ptr<Node>> nodes;
//    nodes.emplace_back(std::make_shared<Node>(fs, 0));
//
//    for (auto i = 0; i < n; ++i)
//    {
//        nodes.emplace_back(std::make_shared<Node>(fs + i + 1, i + 1));
//        if (i != n) nodes[i]->next = nodes[i + 1];
//        else nodes[i]->next = nullptr;
//    }
//
//    GenericUtils::TimeBench timer("Linked_List_Access");
//    while (nodes.front() != nullptr)
//    {
//        process_work(nodes.front());
//        std::cout << "Linked access value: " << nodes.front()->data << " " <<nodes.front()->fib_data << " " << nodes.front()->next << "\n";
//        nodes.front() = nodes.front()->next;
//    }
//}
//
//
//TEST_F(OpenMPTests, Linked_List_Access_2)
//{
//#ifndef N
//#define N 5
//#endif
//#ifndef FS
//#define FS 38
//#endif
//
//    struct node
//    {
//        int data;
//        int fibdata;
//        struct node* next;
//    };
//
//    std::function<int(int)> fib;
//    fib = [&fib](int n)
//    {
//        int x, y;
//        if (n < 2) { return (n); }
//        else {
//            x = fib(n - 1);
//            y = fib(n - 2);
//            return (x + y);
//        }
//    };
//
//    auto process_work = [&fib](struct node *p)
//    {
//        int n = p->data;
//        p->fibdata = fib(n);
//    };
//
//    std::function<struct node*(struct node* )> init_list;
//
//    init_list = [&init_list](struct node* p)
//    {
//        int i;
//        struct node* head = NULL;
//        struct node* temp = NULL;
//
//        head = (struct node*)malloc(sizeof(struct node));
//        p = head;
//        p->data = FS;
//        p->fibdata = 0;
//        for (i=0; i< N; i++)
//        {
//            temp  =  (struct node*)malloc(sizeof(struct node));
//            p->next = temp;
//            p = temp;
//            p->data = FS + i + 1;
//            p->fibdata = i+1;
//        }
//        p->next = NULL;
//        return head;
//    };
//
//    struct node *p=NULL;
//    struct node *temp=NULL;
//    struct node *head=NULL;
//
//    p = init_list(p);
//    head = p;
//
//    GenericUtils::TimeBench timer("Linked_List_Access");
//    while (p != NULL)
//    {
//        process_work(p);
////        std::cout << "Linked access value: " <<  p->data << " " <<p->fibdata << " " << p->next << "\n";
//        p = p->next;
//    }
//
//    p = head;
//    while (p != NULL)
//    {
//        temp = p->next;
//        free (p);
//        p = temp;
//    }
//    free (p);
//}


TEST_F(OpenMPTests, Basic_Path_Integral)
{
    using namespace Eigen;
    using StateVec = Matrix<double, 2, 1>;
    using StateMat = Matrix<double, 2, 2>;
    using CtrlVec  = Matrix<double, 1, 1>;
    using CtrlMat  = Eigen::Matrix<double, 1, 1>;
    const auto state_gain = StateMat::Identity();
    const auto ctrl_gain  = CtrlMat::Identity();

    struct Data{StateVec x = StateVec::Zero(); CtrlVec u = CtrlVec::Zero(); const double t = 0.01; double acc = 0;};

    auto step = [](Data &d)
    {
        constexpr const double m = 1;
        d.acc = d.u(0, 0) / m;
        d.x(0, 0) += d.acc * d.t;
        d.x(1, 0) += d.x(0, 0) * d.t;
    };

    auto cost = [&](Data& data)-> double
    {
        return (data.x.transpose() * state_gain * data.x + data.u.transpose() * ctrl_gain * data.u)(0, 0);
    };

    constexpr const int time = 75; int samples = 1e5;
    std::vector<std::vector<double>> cst(samples, {0, 0, 0, 0, 0, 0, 0, 0});
    EigenMultivariateNormal<double> normX_cholesk (CtrlVec::Zero(),CtrlMat::Identity(),time,true);
    std::vector<Eigen::Matrix<double, -1, -1>> ctrl_samples(samples);
    GenericUtils::TimeBench timer("Basic_Path_Integral");


#pragma omp parallel for default(none) shared(normX_cholesk, samples, ctrl_samples) num_threads(14)
    for(auto sample = 0; sample < samples; ++sample)
    {
        ctrl_samples[sample].resize(1, time);
        normX_cholesk.samples_fill(ctrl_samples[sample]);
    }

    int t;
    #pragma omp parallel for default(none) private(t) shared(step, cost, cst, samples, ctrl_samples) num_threads(10)
    for(int sample = 0; sample < samples; ++sample)
    {
        Data d;
        d.u = CtrlVec::Zero(); d.x = StateVec::Zero();
        const auto& ctrl_traj = ctrl_samples[sample];
        for(t = 0; t < time; ++t)
        {
            d.u = ctrl_traj.block(0, t, 1, 1);
            step(d);
            cst[sample][0] += cost(d);
        }
    }

    double total_sum = 0.0;
#pragma omp parallel for reduction(+:total_sum) default(none) shared(cst, samples) num_threads(14)
    for(auto i=0; i<samples; ++i)
        total_sum += cst[i][0];

    std::cout << total_sum << std::endl;
}


TEST_F(OpenMPTests, Basic_Path_Integral_Data_Vec)
{
    using namespace Eigen;
    using StateVec = Matrix<double, 2, 1>;
    using StateMat = Matrix<double, 2, 2>;
    using CtrlVec  = Matrix<double, 1, 1>;
    using CtrlMat  = Eigen::Matrix<double, 1, 1>;
    const auto state_gain = StateMat::Identity();
    const auto ctrl_gain  = CtrlMat::Identity();

    struct Data{StateVec x = StateVec::Zero(); CtrlVec u = CtrlVec::Zero(); const double t = 0.01; double acc = 0; double pad[3];};

    auto step = [](Data &d)
    {
        constexpr const double m = 1;
        d.acc = d.u(0, 0) / m;
        d.x(0, 0) += d.acc * d.t;
        d.x(1, 0) += d.x(0, 0) * d.t;
    };

    auto cost = [&](Data& data)-> double
    {
        return (data.x.transpose() * state_gain * data.x + data.u.transpose() * ctrl_gain * data.u)(0, 0);
    };

    constexpr const int time = 75, nthreads = 10; int samples = 1e5;
    std::vector<std::vector<double>> cst(samples, {0, 0, 0, 0, 0, 0, 0, 0});
    EigenMultivariateNormal<double> normX_cholesk (CtrlVec::Zero(),CtrlMat::Identity(),time,true);
    std::vector<Eigen::Matrix<double, -1, -1>> ctrl_samples(samples);
    std::vector<Data> data_vec(14);
    GenericUtils::TimeBench timer("Basic_Path_Integral");
    static int segments = samples/nthreads;


#pragma omp parallel for default(none) shared(normX_cholesk, samples, ctrl_samples) num_threads(14)
    for(auto sample = 0; sample < samples; ++sample)
    {
        ctrl_samples[sample].resize(1, time);
        normX_cholesk.samples_fill(ctrl_samples[sample]);
    }

    int t;
#pragma omp parallel default(none) private(t) shared(data_vec, step, cost, cst, samples, ctrl_samples, segments) num_threads(nthreads)
    {
        int id = omp_get_thread_num(); int adjust = 0;
        if (id == nthreads) adjust = samples % nthreads;
        for (int thread = id * segments; thread < (id+1) * segments - adjust; ++thread) {
            data_vec[id].u = CtrlVec::Zero();
            data_vec[id].x = StateVec::Zero();
            const auto &ctrl_traj = ctrl_samples[thread];
            for (t = 0; t < time; ++t)
            {
                data_vec[id].u = ctrl_traj.block(0, t, 1, 1);
                step(data_vec[id]);
                cst[thread][0] += cost(data_vec[id]);
            }
        }
    }

    double total_sum = 0.0;
#pragma omp parallel for reduction(+:total_sum) default(none) shared(cst, samples) num_threads(14)
    for(auto i=0; i<samples; ++i)
        total_sum += cst[i][0];

    std::cout << total_sum << std::endl;
}


TEST_F(OpenMPTests, Minimum_Array)
{

    struct Compare { double val = std::numeric_limits<double>::max(); std::size_t index = 0; };
#pragma omp declare reduction(minimum : struct Compare : omp_out = omp_in.val < omp_out.val ? omp_in : omp_out)

    auto par_min = [](std::vector<int>& input)
    {
        struct Compare min;
#pragma omp parallel for reduction(minimum:min)
        for (int i = 1; i < input.size(); i++) {
            if (input[i] < min.val) {
                min.val = input[i];
                min.index = i;
            }
        }
        return min;
    };

    constexpr const unsigned int samples = 100000;
    std::vector<int> rand_arr;
    for(auto i=0; i < samples; ++i)
    {
        rand_arr.push_back({rand()+1});
    }

    auto res = par_min(rand_arr);
    const auto min = std::min_element(rand_arr.begin(), rand_arr.end());
    ASSERT_NEAR(res.val, *min, 0);
}


TEST_F(OpenMPTests, Parrallel_Samples)
{
    using namespace Eigen;
    using CtrlVec  = Matrix<double, 1, 1>;
    using CtrlMat  = Eigen::Matrix<double, 1, 1>;

    constexpr const int time = 10, nthreads = 5, iter_limit = 5; int samples = 10;
    std::vector<EigenMultivariateNormal<double>>rng_vec;
    const EigenMultivariateNormal<double> normX_cholesk (CtrlVec::Zero(),CtrlMat::Identity(),time,true);
    std::vector<Eigen::Matrix<double, -1, -1>> ctrl_samples(samples);
    GenericUtils::TimeBench timer("Basic_Path_Integral");
    static int segments = samples/nthreads;

    for(auto i = 0; i < nthreads; ++i) {
        EigenMultivariateNormal<double> temp_rng(CtrlVec::Zero(),CtrlMat::Identity(),time,true, i+1);
        rng_vec.emplace_back(temp_rng);
    };

    auto m_carry_over = samples % nthreads;
    int m_per_thread_sample = (samples - m_carry_over)/ nthreads;

    for (int iteration = 0; iteration < iter_limit; ++iteration) {
#pragma omp  parallel default(none) shared(ctrl_samples, samples, m_per_thread_sample, rng_vec) num_threads(nthreads)
        {
            int id = omp_get_thread_num();
            unsigned int adjust = 0;
            if (id == nthreads - 1) adjust = samples % nthreads;
            for (int sample = id * m_per_thread_sample; sample < (id + 1) * m_per_thread_sample + adjust; ++sample) {
                ctrl_samples[sample].resize(1, time);
                rng_vec[id].samples_fill(ctrl_samples[sample]);
            };
        }
    }
}


TEST_F(OpenMPTests, Samples_Sequential)
{
    using namespace Eigen;
    using CtrlVec  = Matrix<double, 1, 1>;
    using CtrlMat  = Eigen::Matrix<double, 1, 1>;

    constexpr const int time = 45; int samples = 10, iter_limit = 5;
    EigenMultivariateNormal<double> normX_cholesk (CtrlVec::Zero(),CtrlMat::Identity(),time,true, 1);
    std::vector<Eigen::Matrix<double, -1, -1>> ctrl_samples(samples);
    GenericUtils::TimeBench timer("Basic_Path_Integral");

    for (int iteration = 0; iteration < iter_limit; ++iteration){
        for (int sample = 0; sample < samples; ++sample){
            ctrl_samples[sample].resize(1, time);
            normX_cholesk.samples_fill(ctrl_samples[sample]);
        }
    }
}


TEST_F(OpenMPTests, Samples_Sequential_2)
{
    using namespace Eigen;
    using CtrlVec  = Matrix<double, 1, 1>;
    using CtrlMat  = Eigen::Matrix<double, 1, 1>;

    constexpr const int time = 45; int samples = 10, iter_limit = 1;
    EigenMultivariateNormal<double> normX_cholesk2 (CtrlVec::Zero(),CtrlMat::Identity(),time,true, 1);
    std::vector<Eigen::Matrix<double, -1, -1>> ctrl_samples(samples);
    GenericUtils::TimeBench timer("Basic_Path_Integral");

    for (int iteration = 0; iteration < iter_limit; ++iteration){
        for (int sample = 0; sample < samples; ++sample){
            ctrl_samples[sample].resize(1, time);
            normX_cholesk2.samples_fill(ctrl_samples[sample]);
        };
    }
}


TEST_F(OpenMPTests, Samples_Sequential_3)
{
    using namespace Eigen;
    using CtrlVec  = Matrix<double, 1, 1>;
    using CtrlMat  = Eigen::Matrix<double, 1, 1>;

    constexpr const int time = 45; int samples = 10, iter_limit = 1;
    EigenMultivariateNormal<double> normX_cholesk2 (CtrlVec::Zero(),CtrlMat::Identity(),time,true, 1);
    Eigen::Matrix<double, -1, -1> ctrl_samples;
    ctrl_samples.resize(samples, time);
    GenericUtils::TimeBench timer("Basic_Path_Integral");

    for (int iteration = 0; iteration < iter_limit; ++iteration) {
        std::cout << "----------------------------------------------------" << "\n";
        for (int sample = 0; sample < samples; ++sample) {
            normX_cholesk2.samples_fill(ctrl_samples.row(sample));
        }
    }
}


TEST_F(OpenMPTests, Samples_Sequential_4) {
    using namespace Eigen;
    using CtrlVec = Matrix<double, 1, 1>;
    using CtrlMat = Eigen::Matrix<double, 1, 1>;

    constexpr const int time = 45;
    int samples = 10, iter_limit = 1;
    EigenMultivariateNormal<double> normX_cholesk2(CtrlVec::Zero(), CtrlMat::Identity(), time, true, 1);
    Eigen::Matrix<double, -1, -1> ctrl_samples;
    ctrl_samples.resize(samples, time);
    GenericUtils::TimeBench timer("Basic_Path_Integral");

    for (int iteration = 0; iteration < iter_limit; ++iteration) {
        std::cout << "----------------------------------------------------" << "\n";
        for (int sample = 0; sample < samples; ++sample) {
            normX_cholesk2.samples_fill(ctrl_samples.row(sample));
        }
    }
}


TEST_F(OpenMPTests, Eigen_Vec_Red) {
    using namespace Eigen;
    using CtrlVector = Matrix<double, 1, 1>;

    int samples = 100000; int iter_limit = 75; int n_thread = 1;
    auto m_carry_over = samples % n_thread;
    int per_thread_sample = (samples - m_carry_over)/ n_thread;
    std::vector<CtrlVector> temp_thread {1, CtrlVector::Zero()};
    std::vector<CtrlVector> ctrl_vec{75, CtrlVector::Zero()};
    std::vector<CtrlVector> ctrl_vec_seq{75, CtrlVector::Zero()};
    std::vector<Eigen::Matrix<double, -1, -1>> ctrl_sample{100000};
    std::for_each(ctrl_sample.begin(), ctrl_sample.end(), [iter_limit](auto &elem) {
        elem.resize(1, iter_limit);
        elem.setRandom();
    });

    GenericUtils::TimeBench timer("Basic_Path_Integral");

#pragma omp  parallel for default(none) collapse(2) shared(iter_limit, ctrl_sample, ctrl_vec, samples) num_threads(6)
    for(auto iter = 0; iter < iter_limit; ++iter)
            for(auto sample = 0; sample < samples; ++sample) {
                ctrl_vec[iter] +=  ctrl_sample[sample].block(0, iter * 1, 1, 1);
        }


//    for(auto iter = 0; iter < iter_limit; ++iter)
//        for(auto sample = 0; sample < samples; ++sample)
//            ctrl_vec_seq[iter] +=  ctrl_sample[sample].block(0, iter * 1, 1, 1);

    std::cout << "\n";
    std::for_each(ctrl_vec.begin(), ctrl_vec.end(), [](const auto& elem){std::cout << elem << ", ";});
    std::cout << "\n";
    std::for_each(ctrl_vec_seq.begin(), ctrl_vec_seq.end(), [](const auto& elem){std::cout << elem<< ", ";});
    std::cout << "\n";

}








































