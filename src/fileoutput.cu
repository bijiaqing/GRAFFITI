#include <ctime>            // for std::time_t, std::time, std::ctime
#include <chrono>           // for std::chrono::system_clock
#include <iomanip>          // for std::setw, std::setfill, std::setprecision, std::scientific, std::defaultfloat
#include <iostream>         // for std::cout, std::endl

#include "cudust.cuh"

__host__ 
void log_output (int idx_file)
{
    std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout   
        << std::endl << std::setfill('0')
        << std::setw(3) << idx_file << "/" 
        << std::setw(3) << SAVE_MAX << " finished on " << std::ctime(&end_time)
        << std::endl;
}

__host__ 
std::string frame_num (int number, std::size_t length)
{
    std::string str = std::to_string(number);
    if (str.length() < length) str.insert(0, length - str.length(), '0');
    return str;
}

__host__ 
bool save_variable (const std::string &file_name)
{
    std::ofstream file(file_name);
    if (!file) return false;
    
    file << "[PARAMETERS]"                                                                  << std::endl;
    file                                                                                    << std::endl;

    // Gas parameters
    #ifndef CONST_NU
    file << "ALPHA       = " << std::scientific     << std::setprecision(8) << ALPHA        << std::endl;
    #else
    file << "NU          = " << std::scientific     << std::setprecision(8) << NU           << std::endl;
    #endif
    file << "ASPR_0      = " << std::defaultfloat   << std::setprecision(8) << ASPR_0       << std::endl;
    file << "IDX_P       = " << std::defaultfloat   << std::setprecision(8) << IDX_P        << std::endl;
    file << "IDX_Q       = " << std::defaultfloat   << std::setprecision(8) << IDX_Q        << std::endl;
    file                                                                                    << std::endl;
    
    // Dust parameters
    file << "ST_0        = " << std::scientific     << std::setprecision(8) << ST_0         << std::endl;
    file << "M_D         = " << std::scientific     << std::setprecision(8) << M_D          << std::endl;
    file << "RHO_0       = " << std::scientific     << std::setprecision(8) << RHO_0        << std::endl;
    #ifdef RADIATION
    file << "BETA_0      = " << std::scientific     << std::setprecision(8) << BETA_0       << std::endl;
    file << "KAPPA_0     = " << std::scientific     << std::setprecision(8) << KAPPA_0      << std::endl;
    #endif
    #ifdef DIFFUSION
    file << "SC_R        = " << std::scientific     << std::setprecision(8) << SC_R    << std::endl;
    file << "SC_Z        = " << std::scientific     << std::setprecision(8) << SC_Z    << std::endl;
    #endif
    #ifdef COLLISION
    file << "LAMBDA_0    = " << std::scientific     << std::setprecision(8) << LAMBDA_0     << std::endl;
    file << "V_FRAG      = " << std::scientific     << std::setprecision(8) << V_FRAG       << std::endl;
    file << "KNN_SIZE    = " << std::defaultfloat   << std::setprecision(8) << KNN_SIZE     << std::endl;
    file << "MAX_DIST    = " << std::scientific     << std::setprecision(8) << MAX_DIST     << std::endl;
    #endif
    file                                                                                    << std::endl;

    // Mesh domain
    file << "N_PAR       = " << std::scientific     << std::setprecision(8) << N_PAR        << std::endl;
    file                                                                                    << std::endl;
    file << "N_X         = " << std::defaultfloat   << std::setprecision(8) << N_X          << std::endl;
    file << "X_MIN       = " << std::defaultfloat   << std::setprecision(8) << X_MIN        << std::endl;
    file << "X_MAX       = " << std::defaultfloat   << std::setprecision(8) << X_MAX        << std::endl;
    file                                                                                    << std::endl;
    file << "N_Y         = " << std::defaultfloat   << std::setprecision(8) << N_Y          << std::endl;
    file << "Y_MIN       = " << std::defaultfloat   << std::setprecision(8) << Y_MIN        << std::endl;
    file << "Y_MAX       = " << std::defaultfloat   << std::setprecision(8) << Y_MAX        << std::endl;
    file                                                                                    << std::endl;
    file << "N_Z         = " << std::defaultfloat   << std::setprecision(8) << N_Z          << std::endl;
    file << "Z_MIN       = " << std::defaultfloat   << std::setprecision(8) << Z_MIN        << std::endl;
    file << "Z_MAX       = " << std::defaultfloat   << std::setprecision(8) << Z_MAX        << std::endl;
    file                                                                                    << std::endl;

    // Initialization parameters
    file << "INIT_XMIN   = " << std::defaultfloat   << std::setprecision(8) << INIT_XMIN    << std::endl;
    file << "INIT_XMAX   = " << std::defaultfloat   << std::setprecision(8) << INIT_XMAX    << std::endl;
    file << "INIT_YMIN   = " << std::defaultfloat   << std::setprecision(8) << INIT_YMIN    << std::endl;
    file << "INIT_YMAX   = " << std::defaultfloat   << std::setprecision(8) << INIT_YMAX    << std::endl;
    file << "INIT_ZMIN   = " << std::defaultfloat   << std::setprecision(8) << INIT_ZMIN    << std::endl;
    file << "INIT_ZMAX   = " << std::defaultfloat   << std::setprecision(8) << INIT_ZMAX    << std::endl;
    file << "INIT_SMIN   = " << std::scientific     << std::setprecision(8) << INIT_SMIN    << std::endl;
    file << "INIT_SMAX   = " << std::scientific     << std::setprecision(8) << INIT_SMAX    << std::endl;
    file                                                                                    << std::endl;

    // Time step and output
    file << "SAVE_MAX    = " << std::defaultfloat   << std::setprecision(8) << SAVE_MAX     << std::endl;
    file << "SAVE_PAR    = " << std::defaultfloat   << std::setprecision(8) << SAVE_PAR     << std::endl;
    file << "DT_OUT      = " << std::scientific     << std::setprecision(8) << DT_OUT       << std::endl;
    file << "DT_DYN      = " << std::scientific     << std::setprecision(8) << DT_DYN       << std::endl;
    file << "PATH_OUT    = "                                                << PATH_OUT     << std::endl;
    file                                                                                    << std::endl;

    // Swarm structure as numpy dtype (configparser-compatible)
    // dtype = np.dtype([(name, dtype) for name, dtype in config['SWARM_DTYPE'].items()])
    file << "[SWARM_DTYPE]"                                                                 << std::endl;
    file << "position_x = f8"                                                               << std::endl;
    file << "position_y = f8"                                                               << std::endl;
    file << "position_z = f8"                                                               << std::endl;
    file << "velocity_x = f8"                                                               << std::endl;
    file << "velocity_y = f8"                                                               << std::endl;
    file << "velocity_z = f8"                                                               << std::endl;
    file << "par_size   = f8"                                                               << std::endl;
    file << "par_numr   = f8"                                                               << std::endl;
    #ifdef COLLISION
    file << "col_rate   = f8"                                                               << std::endl;
    #endif
    
    return file.good();
}