#ifndef FUNCLIB_CUH 
#define FUNCLIB_CUH

#include <fstream>          // for std::ofstream, std::ifstream
#include <random>           // for std::mt19937

#include "const.cuh"

#if defined(COLLISION) || defined(DIFFUSION)
#include "curand_kernel.h"  // for curandState
using curs = curandState;
#endif // COLLISION and DIFFUSION

// =========================================================================================================================
// initialization

__global__ void particle_init (swarm *dev_particle, 
    const real *dev_random_x, const real *dev_random_y, const real *dev_random_z, const real *dev_random_s);

#if defined(COLLISION)
__global__ void treenode_init (tree *dev_treenode, const swarm *dev_particle);
__global__ void rs_swarm_init (curs *dev_rs_swarm, int seed = 0);
__global__ void rs_grids_init (curs *dev_rs_grids, int seed = 1);
#else
#if defined(DIFFUSION)
__global__ void rs_swarm_init (curs *dev_rs_swarm, int seed = 2);
#endif // DIFFUSION
#endif // COLLISION

// =========================================================================================================================
// mesh calculation

#ifdef RADIATION // for optical depth calculation
__global__ void optdepth_init (real *dev_optdepth);
__global__ void optdepth_scat (real *dev_optdepth, const swarm *dev_particle);
__global__ void optdepth_calc (real *dev_optdepth);
__global__ void optdepth_csum (real *dev_optdepth);
__global__ void optdepth_mean (real *dev_optdepth);
#endif // RADIATION

__global__ void dustdens_init (real *dev_dustdens);
__global__ void dustdens_scat (real *dev_dustdens, const swarm *dev_particle);
__global__ void dustdens_calc (real *dev_dustdens);

// =========================================================================================================================
// collision

#ifdef COLLISION
__global__ void col_rate_init (real *dev_col_rate, real *dev_col_expt, real *dev_col_rand);
__global__ void col_rate_calc (real *dev_col_rate, swarm *dev_particle, const tree *dev_treenode, const bbox *dev_boundbox);
__global__ void col_flag_calc (int *dev_col_flag, curs *dev_rs_grids, real *dev_col_rand, const real *dev_col_rate, real dt_col);

__global__ void run_collision (swarm *dev_particle, curs *dev_rs_swarm, real *dev_col_expt, const real *dev_col_rand, 
    const int *dev_col_flag, const tree *dev_treenode, const bbox *dev_boundbox);

__device__ real _get_grid_volume (int idx, real *y0_ptr = nullptr, real *dy_ptr = nullptr); // defined in gridfields.cu
#endif // COLLISION

// =========================================================================================================================
// dust dynamics

#ifdef RADIATION
__global__ void ssa_substep_1 (swarm *dev_particle, real dt);
__global__ void ssa_substep_2 (swarm *dev_particle, const real *dev_optdepth, real dt);
#else
__global__ void ssa_integrate (swarm *dev_particle, real dt);
#endif // RADIATION

#ifdef DIFFUSION
__global__ void pos_diffusion (swarm *dev_particle, curs *dev_rs_swarm, real dt);
#endif // DIFFUSION

// =========================================================================================================================
// interpolation

__device__ interp _3d_interp_midpt_y (real loc_x, real loc_y, real loc_z);
__device__ interp _3d_interp_outer_y (real loc_x, real loc_y, real loc_z);

__device__ __forceinline__ real _get_loc_x (real x)
{ 
    return (N_X > 1) ? (static_cast<real>(N_X)*   (x - X_MIN) /    (X_MAX - X_MIN)) : 0.0; 
}

__device__ __forceinline__ real _get_loc_y (real y)
{ 
    return (N_Y > 1) ? (static_cast<real>(N_Y)*log(y / Y_MIN) / log(Y_MAX / Y_MIN)) : 0.0; 
}

__device__ __forceinline__ real _get_loc_z (real z)
{ 
    return (N_Z > 1) ? (static_cast<real>(N_Z)*   (z - Z_MIN) /    (Z_MAX - Z_MIN)) : 0.0; 
}

// Shared device helpers (defined in helperfunc.cu)

__device__ real _get_eta     (real R, real Z);
__device__ real _get_hgas    (real R);
__device__ real _get_hdust   (real R, real stokes);
__device__ real _get_stokes  (real R, real Z, real size);
__device__ real _get_omegaK  (real R);
__device__ real _get_delta_R (real R, real h_gas);
__device__ real _get_delta_Z (real R, real h_gas);

// =========================================================================================================================
// randprofile.cu

extern std::mt19937 rand_generator;

__host__ void rand_uniform  (real *profile, int number, real p_min, real p_max);
__host__ void rand_gaussian (real *profile, int number, real p_min, real p_max, real mu, real std);
__host__ void rand_pow_law  (real *profile, int number, real p_min, real p_max, real idx_pow);
__host__ void rand_convpow  (real *profile, int number, real p_min, real p_max, real idx_pow, real smooth, int bins);

// =========================================================================================================================
// file I/O helpers

std::string frame_num (int number, std::size_t length = 5);
bool save_variable (const std::string &file_name);
void log_output (int idx_file);

// =========================================================================================================================
// files open and save
// Template functions must be defined in header

template <typename DataType> __host__ inline
bool save_binary (const std::string &file_name, DataType *data, int number)
{
    std::ofstream file(file_name, std::ios::binary);
    if (!file) return false;
    
    file.write(reinterpret_cast<char*>(data), sizeof(DataType)*number);
    return file.good();
}

template <typename DataType> __host__ inline
bool load_binary (const std::string &file_name, DataType *data, int number)
{
    std::ifstream file(file_name, std::ios::binary);
    if (!file) return false;
    
    file.read(reinterpret_cast<char*>(data), sizeof(DataType)*number);
    return file.good();
}

// =========================================================================================================================

#endif // FUNCLIB_CUH