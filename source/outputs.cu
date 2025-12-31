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

__host__
void open_bin_file (std::ofstream &bin_file, std::string file_name) 
{
    bin_file.open(file_name.c_str(), std::ios::out | std::ios::binary);
}

__host__
void save_bin_file (std::ofstream &bin_file, swarm *data, int number) 
{
    bin_file.write((char*)data, sizeof(swarm)*number);
    bin_file.close();
}

__host__
void save_bin_file (std::ofstream &bin_file, float *data, int number) 
{
    bin_file.write((char*)data, sizeof(float)*number);
    bin_file.close();
}

__host__
void save_bin_file (std::ofstream &bin_file, real *data, int number) 
{
    bin_file.write((char*)data, sizeof(real)*number);
    bin_file.close();
}

__host__
void load_bin_file (std::ifstream &bin_file, std::string file_name) 
{
    bin_file.open(file_name.c_str(), std::ios::in | std::ios::binary);
}

__host__
void read_bin_file (std::ifstream &bin_file, swarm *data, int number) 
{
    bin_file.read((char*)data, sizeof(swarm)*number);
    bin_file.close();
}

// =========================================================================================================================

__host__
void open_txt_file (std::ofstream &txt_file, std::string file_name) 
{
    txt_file.open(file_name.c_str(), std::ios::out);
}

__host__
void save_variable (std::ofstream &txt_file)
{
    txt_file << "[PARAMETERS]"                                                                                              << std::endl;

    // txt_file << "N_PAR = \t" << scientific << std::setprecision(15) << std::setw(24) << std::setfill(' ') << N_PAR            << std::endl;

    txt_file.close();
}
