FetchContent_Declare(
        pybind
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.6.0b1
)

set(PYBIND11_PYTHON_VERSION 3.7 CACHE STRING "")
FetchContent_MakeAvailable(pybind)
find_package(Python3 3.7 EXACT REQUIRED COMPONENTS Interpreter)
find_package(Python3 3.7 EXACT REQUIRED COMPONENTS Development)
message(${Python3_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(
        ${Python3_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_DIRS}
)


pybind11_add_module(mj_derivative example_mj_derivative.cpp)
target_link_libraries(mj_derivative PRIVATE mujoco::mujoco Eigen3::Eigen)
SET_TARGET_PROPERTIES( mj_derivative
        PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mj_derivative"
        )

CONFIGURE_FILE( mj_derivative/__init__.py
        ${CMAKE_CURRENT_BINARY_DIR}/mj_derivative/__init__.py
        )


# Ditto for the setup file.
CONFIGURE_FILE( mj_derivative/setup.py
        ${CMAKE_CURRENT_BINARY_DIR}/mj_derivative/setup.py
        )
