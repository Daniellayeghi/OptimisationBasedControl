
#ifndef OPTCONTROL_MUJOCO_BUFFER_UTILS_H
#define OPTCONTROL_MUJOCO_BUFFER_UTILS_H

#include "Eigen/Core"
#include <fstream>


namespace BufferUtilities
{
    template<int rows, int cols>
    inline void save_to_file(std::fstream *file, std::vector<Eigen::Matrix<double, rows, cols>> &buffer)
    {
        if (file->is_open()){
            for (auto const &element : buffer){
                for (int row = 0; row < element.rows(); ++row){
                    for (int col = 0; col < element.cols(); ++col){
                        *file << std::to_string(element(row, col)) + ", ";
                    }
                }
                *file << '\n';
            }
            file->close();
        }
    }


    template<typename T>
    inline void save_to_file(std::fstream &file, std::vector<T> &buffer)
    {
        if (file.is_open())
        {
            for (auto const &element : buffer)
                file << std::to_string(element) << std::endl;
        }
        file.close();
    }



    template<int rows>
    inline void read_csv_file(const std::string& file_name, std::vector<Eigen::Matrix<double, rows, 1>>& buffer, char delm)
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
