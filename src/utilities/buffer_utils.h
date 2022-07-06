
#ifndef OPTCONTROL_MUJOCO_BUFFER_UTILS_H
#define OPTCONTROL_MUJOCO_BUFFER_UTILS_H

#include "Eigen/Core"
#include <fstream>
#include <iomanip>

namespace BufferUtilities
{

    inline std::string as_string(double value)
    {
        char buf[32];
        return std::string(buf, std::snprintf(buf, sizeof buf, "%.16g", value));
    }


    template<int rows, int cols>
    inline void save_to_file(std::fstream *file, std::vector<Eigen::Matrix<double, rows, cols>> &buffer)
    {
        if (file->is_open()){
            for (auto const &element : buffer){
                for (int col = 0; col < element.cols(); ++col){
                    for (int row = 0; row < element.rows(); ++row){
                        if(row != element.rows() - 1)
                            *file << as_string(element(row, col)) + ", ";
                        else
                            *file << as_string(element(row, col));
                    }
                }
                *file << '\n';
            }
        }
        file->close();
    }


    template<int rows, int cols>
    inline void save_to_file(std::fstream *file, Eigen::Matrix<double, rows, cols> &buffer)
    {
        if (file->is_open()) {
            for (int row = 0; row < buffer.rows(); ++row) {
                for (int col = 0; col < buffer.cols(); ++col) {
                    if(col != buffer.cols() - 1)
                        *file << as_string(buffer(row, col)) + ", ";
                    else
                        *file << as_string(buffer(row, col));
                }
                *file << '\n';
            }
        }
        file->close();
    }


    template<typename T>
    inline void save_to_file(std::fstream *file, std::vector<T> &buffer)
    {
        if (file->is_open())
            for (auto const &element : buffer)
                *file << as_string(element) << "\n";

        file->close();
    }


    template<int rows>
    inline void read_csv_file(const std::string& file_name, std::vector<Eigen::Matrix<double, rows, 1>>& buffer)
    {
        using vector = Eigen::Matrix<double, rows, 1>;
        vector temp_cont = vector::Zero();
        std::ifstream file(file_name);
        if(file.is_open())
        {
            double value;
            std::string line;
            while(std::getline(file, line))
            {
                std::stringstream string_value(line);
                auto iter = 0;
                while (string_value >> value) {
                    temp_cont(iter, 0) = value;
                    if (string_value.peek() == ',')
                        string_value.ignore();
                    ++iter;
                }
                buffer.template emplace_back(temp_cont);
            }
        }
    }

}

#endif //OPTCONTROL_MUJOCO_BUFFER_UTILS_H
