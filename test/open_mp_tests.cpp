
#include "gtest/gtest.h"
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
	#pragma omp parallel
	{
		std::cout << "Tests 1" << "\n";
		std::cout << "Tests 2" << "\n";
	}

}
