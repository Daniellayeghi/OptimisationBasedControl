
#ifndef OPTCONTROL_MUJOCO_BUFFER_UTILS_H
#define OPTCONTROL_MUJOCO_BUFFER_UTILS_H

#include "Eigen/Core"
#include <fstream>


namespace BufferUtilities
{
    template<int rows, int cols>
    inline void save_to_file(std::fstream &file, std::vector<Eigen::Matrix<double, rows, cols>> &buffer) {
        if (file.is_open()) {
            for (auto const &element : buffer) {
                for (int row = 0; row < element.rows(); ++row) {
                    for (int col = 0; col < element.cols(); ++col) {
                        file << std::to_string(element(row, col)) + ", ";
                    }
                }
                file << std::endl;
            }
            file.close();
        }
    }

    template<typename T>
    inline void save_to_file(std::fstream &file, std::vector<T> &buffer)
    {
        if (file.is_open())
        {
            for (auto const &element : buffer)
            {
                file << std::to_string(element) << std::endl;
            }
        }
    }
}

#endif //OPTCONTROL_MUJOCO_BUFFER_UTILS_H
