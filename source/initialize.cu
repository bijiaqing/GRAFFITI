#include "cudust.cuh"

// =========================================================================================================================

__global__ 
void particle_init (swarm *dev_particle, real *dev_prof_azi, real *dev_prof_rad, real *dev_prof_col, real *dev_prof_siz)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        real azi  = dev_prof_azi[idx];
        real rad  = dev_prof_rad[idx];
        real col  = dev_prof_col[idx];
        real size = dev_prof_siz[idx];

        real N_norm; // see N_1 below

        // NOTE: N_par(s) is the number of super-particles of size s, and 
        // N_dust(s) is the number of dust grains in a super-particle of size s
        // (1) dm = N_par(s) * N_dust(s) * RHO_DUST * s^3 ds (total mass of grains of size s)
        // (2) dn = N_par(s) * N_dust(s) ds = N_0 * s^-3.5 ds (total number of grains of size s)
        // to achieve all swarms having the same total surface area, (3) N_dust(s) = N_1 * s^-2
        // from (2) and (3) there is (4) N_par(s) = N_2 * s^-1.5, which explains why pow_idx = -1.5 in main.cu
        // since (5) integrate( N_par(s) ds ) = integrate( N_2 * s^-1.5 ds ) = NUM_PAR
        // there is (6) N_2 = 0.5*NUM_PAR / (s_min^-0.5 - s_max^-0.5)
        // since (7) integrate dm = integrate( N_par(s) * N_dust(s) * RHO_DUST * s^3 ds ) = M_DUST
        // there is (8) 2*N_1*N_2*RHO_DUST*(s_max^0.5 - s_min^0.5) = M_DUST
        // finally, (9) N_1 = M_DUST / NUM_PAR / RHO_DUST / (s_max^0.5 - s_min^0.5) * (s_min^-0.5 - s_max^-0.5)

        N_norm  = M_DUST / NUM_PAR / RHO_DUST;
        N_norm *= (pow(SIZE_INIT_MIN, -0.5) - pow(SIZE_INIT_MAX, -0.5));
        N_norm /= (pow(SIZE_INIT_MAX,  0.5) - pow(SIZE_INIT_MIN,  0.5));
        
        dev_particle[idx].position.x = azi;
        dev_particle[idx].position.y = rad;
        dev_particle[idx].position.z = col;

        dev_particle[idx].grain_size = size;
        // dev_particle[idx].grain_numr = N_norm / size / size; // see equation (3) above
        dev_particle[idx].grain_numr = M_DUST / NUM_PAR / RHO_DUST / size / size / size;
    }
}

// =========================================================================================================================

__global__ 
void velocity_init (swarm *dev_particle, real *dev_optdepth)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx >= 0 && idx < NUM_PAR)
    {
        dev_particle[idx].velocity.x = sqrt(G*M_STAR*dev_particle[idx].position.y);
        dev_particle[idx].velocity.y = 0.0;
        dev_particle[idx].velocity.z = 0.0;
    }
}

// =========================================================================================================================

__global__
void treenode_init (swarm *dev_particle, tree *dev_treenode)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        float azi = static_cast<float>(dev_particle[idx].position.x);
        float rad = static_cast<float>(dev_particle[idx].position.y);
        float col = static_cast<float>(dev_particle[idx].position.z);

        dev_treenode[idx].cartesian.x = rad*sin(col)*cos(azi);
        dev_treenode[idx].cartesian.y = rad*sin(col)*sin(azi);
        dev_treenode[idx].cartesian.z = rad*cos(col);
        dev_treenode[idx].index_old   = idx;
    }
}

// =========================================================================================================================

__global__
void rngs_par_init (curandState *dev_rngs_par, int seed) // the seed is set to 0 by default in cudust.cuh
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        curand_init(seed, idx, 0, &dev_rngs_par[idx]);
    }
}

__global__
void rngs_dim_init (curandState *dev_rngs_dim, int seed) // the seed is set to 1 by default in cudust.cuh
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_DIM)
    {
        curand_init(seed, idx, 0, &dev_rngs_dim[idx]);
    }
}