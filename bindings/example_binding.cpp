#include <pybind11/pybind11.h>
#include <mujoco.h>
#include <iostream>

mjModel *model = NULL;
mjData  *d = NULL;


int load_file(std::string& file_path)
{
    char error[1000] = "Could not load binary model";
    mj_activate(MUJ_KEY_PATH);

    if (not file_path.empty())
        model = mj_loadXML(file_path.c_str(), 0, error, 1000);

    if(!model)
        mju_error_s("Load model error: %s", error);

    std::cout << model->nq << std::endl;
    return 0;
 }

 
namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}