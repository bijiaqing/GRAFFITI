#ifndef FUNCLIB_CUH 
#define FUNCLIB_CUH

#include <fstream>          // for std::ofstream, std::ifstream
#include <random>           // for std::mt19937

#include "const.cuh"
#include "curand_kernel.h"  // for curandState

// =========================================================================================================================
// mesh calculation

__global__ void optdepth_init (real *dev_optdepth);
__global__ void optdepth_enum (real *dev_optdepth, swarm *dev_particle);
__global__ void optdepth_calc (real *dev_optdepth);
__global__ void optdepth_intg (real *dev_optdepth);
__global__ void optdepth_mean (real *dev_optdepth);

__global__ void dustdens_init (real *dev_dustdens);
__global__ void dustdens_enum (real *dev_dustdens, swarm *dev_particle);
__global__ void dustdens_calc (real *dev_dustdens);

// =========================================================================================================================
// initialization 

__global__ void particle_init (swarm *dev_particle, real *dev_random_x, real *dev_random_y, real *dev_random_z, real *dev_random_s);
__global__ void velocity_init (swarm *dev_particle);
__global__ void velocity_init (swarm *dev_particle, real *dev_optdepth);
__global__ void treenode_init (swarm *dev_particle, tree *dev_treenode);

__global__ void rngs_par_init (curandState *dev_rngs_par, int seed = 0);
__global__ void rngs_grd_init (curandState *dev_rngs_grd, int seed = 1);

// =========================================================================================================================
// collision related

__global__ void col_rate_init (real  *dev_col_rate, real  *dev_col_rand, real *dev_col_real, float *dev_max_rate);
__global__ void col_rate_calc (swarm *dev_particle, tree  *dev_treenode, real *dev_col_rate, boxf  *dev_boundbox);
__global__ void col_rate_peak (real  *dev_col_rate, float *dev_max_rate);
__global__ void col_flag_calc (real  *dev_col_rate, real  *dev_col_rand, int  *dev_col_flag, real  *dev_timestep, curandState *dev_rngs_grd);
__global__ void particle_evol (swarm *dev_particle, tree  *dev_treenode, int  *dev_col_flag, real  *dev_col_rand, real *dev_col_real, boxf *dev_boundbox, curandState *dev_rngs_par);

// =========================================================================================================================
// integrator

__global__ void ssa_substep_1 (swarm *dev_particle);
__global__ void ssa_substep_2 (swarm *dev_particle, real *dev_optdepth);
__global__ void pos_diffusion (swarm *dev_particle, curandState *dev_rngs_par);

// =========================================================================================================================
// interpolation

__device__ interp _linear_interp_cent (real loc_x, real loc_y, real loc_z);
__device__ interp _linear_interp_edge (real loc_x, real loc_y, real loc_z);

__device__ real _get_optdepth (real *dev_optdepth, real loc_x, real loc_y, real loc_z);
__device__ real _col_rate_ij (swarm *dev_particle, int idx_old_i, int idx_old_j);

// =========================================================================================================================
// files open and save

__host__ std::string frame_num (int number, std::size_t length = 5);

__host__ void open_txt_file (std::ofstream &txt_file, std::string  file_name);
__host__ void save_variable (std::ofstream &txt_file);

__host__ void open_bin_file (std::ofstream &bin_file, std::string  file_name);
__host__ void save_bin_file (std::ofstream &bin_file, swarm *data, int number);
__host__ void save_bin_file (std::ofstream &bin_file, float *data, int number);
__host__ void save_bin_file (std::ofstream &bin_file, real  *data, int number);

__host__ void load_bin_file (std::ifstream &bin_file, std::string  file_name);
__host__ void read_bin_file (std::ifstream &bin_file, swarm *data, int number);

// =========================================================================================================================
// profile generators

extern std::mt19937 rand_generator;

__host__ void rand_uniform  (real *profile, int number, real p_min, real p_max);
__host__ void rand_gaussian (real *profile, int number, real p_min, real p_max, real p_0, real sigma);
__host__ void rand_pow_law  (real *profile, int number, real p_min, real p_max, real idx_pow);
__host__ void rand_convpow  (real *profile, int number, real p_min, real p_max, real idx_pow, real smooth, int bins);

// =========================================================================================================================

#endif
