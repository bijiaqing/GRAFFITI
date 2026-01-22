#ifndef CUDUST_KERN_CUH 
#define CUDUST_KERN_CUH

#include "const.cuh"

// =========================================================================================================================
// Kernel declarations (organized by compiler flags)
// =========================================================================================================================

// Always compiled
__global__ void particle_init (swarm *dev_particle, const real *dev_random_x, const real *dev_random_y, const real *dev_random_z, const real *dev_random_s);
__global__ void dustdens_init (real *dev_dustdens);
__global__ void dustdens_scat (real *dev_dustdens, const swarm *dev_particle);
__global__ void dustdens_calc (real *dev_dustdens);

// =========================================================================================================================
#ifdef COLLISION

__global__ void col_rate_init (real *dev_col_rate, real *dev_col_expt, real *dev_col_rand);
__global__ void col_rate_calc (real *dev_col_rate, swarm *dev_particle, const tree *dev_treenode, const bbox *dev_boundbox);
__global__ void col_flag_calc (int *dev_col_flag, curs *dev_rs_grids, real *dev_col_rand, const real *dev_col_rate, real dt_col);
__global__ void col_proc_exec (swarm *dev_particle, curs *dev_rs_swarm, real *dev_col_expt, const real *dev_col_rand, const int *dev_col_flag, const tree *dev_treenode, const bbox *dev_boundbox);
__global__ void treenode_init (tree *dev_treenode, const swarm *dev_particle);
__global__ void rs_swarm_init (curs *dev_rs_swarm, int seed = 0);
__global__ void rs_grids_init (curs *dev_rs_grids, int seed = 1);

#endif // COLLISION

// =========================================================================================================================
#ifdef RADIATION

__global__ void optdepth_init (real *dev_optdepth);
__global__ void optdepth_scat (real *dev_optdepth, const swarm *dev_particle);
__global__ void optdepth_calc (real *dev_optdepth);
__global__ void optdepth_csum (real *dev_optdepth);
__global__ void optdepth_mean (real *dev_optdepth);
__global__ void ssa_substep_1 (swarm *dev_particle, real dt);
__global__ void ssa_substep_2 (swarm *dev_particle, const real *dev_optdepth, real dt);

#else

__global__ void ssa_transport (swarm *dev_particle, real dt);

#endif // RADIATION

// =========================================================================================================================
#ifdef DIFFUSION

#ifndef COLLISION
__global__ void rs_swarm_init (curs *dev_rs_swarm, int seed = 2);
#endif // COLLISION

__global__ void diffusion_pos (swarm *dev_particle, curs *dev_rs_swarm, real dt);
__global__ void diffusion_vel (swarm *dev_particle, curs *dev_rs_swarm, real dt);

#endif // DIFFUSION

// =========================================================================================================================

#endif // CUDUST_KERN_CUH