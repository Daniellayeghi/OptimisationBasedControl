FetchContent_Declare(
        pybind
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.6.0b1
)

FetchContent_MakeAvailable(pybind)

find_package(PythonInterp 3)
find_package(PythonLibs   3)

if(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
    INCLUDE_DIRECTORIES(
            ${PYTHON_INCLUDE_DIRS}
            ${PYBIND11_INCLUDE_DIRS}
    )

    pybind11_add_module(example MODULE example_binding.cpp)
    target_link_libraries(example PRIVATE ${LIB_MUJOCO})
    SET_TARGET_PROPERTIES( example
            PROPERTIES
            PREFIX ""
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/example"
            )

    CONFIGURE_FILE( example/__init__.py
            ${CMAKE_CURRENT_BINARY_DIR}/example/__init__.py
            )


    # Ditto for the setup file.
    CONFIGURE_FILE( example/setup.py
            ${CMAKE_CURRENT_BINARY_DIR}/example/setup.py
            )
else()
    message(FATAL_ERROR "Binding deps not found!")
endif()