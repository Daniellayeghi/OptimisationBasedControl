#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mujoco.h>
#include <iostream>

mjModel *model = nullptr;
mjData  *d = nullptr;

void load_xml(std::string& file_path)
{
    char error[1000] = "Could not load binary model";
    mj_activate(MUJ_KEY_PATH);

    if (not file_path.empty())
        model = mj_loadXML(file_path.c_str(), 0, error, 1000);

    if(!model)
        mju_error_s("Load model error: %s", error);
}


void print_array(std::array<double, 4>& arr)
{
    for(auto & element : arr)
        std::cout << element << std::endl;
}


namespace py = pybind11;

PYBIND11_MODULE(example, m) {

    m.def("load_xml_file", &load_xml);
    m.def("print", &print_array);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}