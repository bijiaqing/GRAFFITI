#include <iomanip>  // for std::setprecision, std::setw
#include <iostream> // for std::endl

#include "cudust.cuh"

// =========================================================================================================================

__host__
std::string frame_num (int number, std::size_t length)  // the length is set to 5 by default in cudust.cuh
{
    std::string str = std::to_string(number);

    if (str.length() < length)
    {
        str.insert(0, length - str.length(), '0');
    }

    return str;
}

// =========================================================================================================================

template <typename DataType> __host__
bool save_binary (const std::string &file_name, DataType *data, int number)
{
    std::ofstream file(file_name, std::ios::binary);
    if (!file) return false;
    
    file.write(reinterpret_cast<char*>(data), sizeof(DataType)*number);
    return file.good();
}

template <typename DataType> __host__
bool load_binary (const std::string &file_name, DataType *data, int number)
{
    std::ifstream file(file_name, std::ios::binary);
    if (!file) return false;
    
    file.read(reinterpret_cast<char*>(data), sizeof(DataType)*number);
    return file.good();
}

// =========================================================================================================================

__host__
bool save_variable (const std::string &file_name)
{
    std::ofstream file(file_name);
    if (!file) return false;
    
    file << "[PARAMETERS]" << std::endl;
    // file << "N_PAR = \t" << std::scientific << std::setprecision(15) 
    //      << std::setw(24) << std::setfill(' ') << N_PAR << std::endl;
    
    return file.good();
}