#include "cudust.cuh"

// =========================================================================================================================

__global__ 
void particle_init (swarm *dev_particle, real *dev_random_x, real *dev_random_y, real *dev_random_z, real *dev_random_s)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    // the question is, if we want 
    // (1) the -3.5 power-law of grain size distribution, 
    // (2) all superparticles have an equal total surface area, and 
    // (3) a certain total particle number N_PAR and total dust mass M_D,
    // what is the size distribution of superparticles (to be fixed in main.cu), and
    // and what is the grain number inside each superparticle (to be solved here)

    if (idx < N_PAR)
    {
        real x = dev_random_x[idx];
        real y = dev_random_y[idx];
        real z = dev_random_z[idx];
        real s = dev_random_s[idx];

        // N_par(s) is the number of super-particles of grain size s, and 
        // N_dust(s) is the number of dust grains in a super-particle of grain size s
        // (1) dm = N_par(s) * N_dust(s) * RHO_0 * s^3 ds     (total mass   of grains of size s)
        // (2) dn = N_par(s) * N_dust(s) ds = N_0 * s^-3.5 ds (total number of grains of size s)
        // to achieve all swarms having the same total surface area, (3) N_dust(s) = N_1 * s^-2
        // combining (2) and (3) there is (4) N_par(s) = N_2 * s^-1.5, which explains why pow_idx = -1.5 in main.cu
        // since (5) integrate( N_par(s) ds ) = integrate( N_2 * s^-1.5 ds ) = N_PAR
        // there is (6) N_2 = 0.5*N_PAR / (s_min^-0.5 - s_max^-0.5)
        // since (7) integrate dm = integrate( N_par(s) * N_dust(s) * RHO_0 * s^3 ds ) = M_D
        // there is (8) 2*N_1*N_2*RHO_0*(s_max^0.5 - s_min^0.5) = M_D
        // so, (9) N_1 = M_D / N_PAR / RHO_0 / (s_max^0.5 - s_min^0.5) * (s_min^-0.5 - s_max^-0.5)
        // finally we put (9) into (3) to get what we need

        real n =  M_D / N_PAR / RHO_0 / s / s;

        if (INIT_SMIN == INIT_SMAX)
        {
            n /= s;
        }
        else
        {
            n *= (pow(INIT_SMIN, -0.5) - pow(INIT_SMAX, -0.5));
            n /= (pow(INIT_SMAX,  0.5) - pow(INIT_SMIN,  0.5));
        }

        dev_particle[idx].position.x = x;
        dev_particle[idx].position.y = y;
        dev_particle[idx].position.z = z;
        dev_particle[idx].par_size   = s;
        dev_particle[idx].par_numr   = n;
        dev_particle[idx].col_rate   = 0.0;
    }
}

// =========================================================================================================================

__global__ 
void velocity_init (swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        dev_particle[idx].velocity.x = sqrt(G*M_S*dev_particle[idx].position.y);
        dev_particle[idx].velocity.y = 0.0;
        dev_particle[idx].velocity.z = 0.0;
    }
}

__global__ 
void velocity_init (swarm *dev_particle, real *dev_optdepth)
{
    // dev_optdepth is included in case one wants to initialized velocity for particles in the shadow
    
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        dev_particle[idx].velocity.x = sqrt(G*M_S*dev_particle[idx].position.y);
        dev_particle[idx].velocity.y = 0.0;
        dev_particle[idx].velocity.z = 0.0;
    }
}

// =========================================================================================================================

__global__
void treenode_init (swarm *dev_particle, tree *dev_treenode)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        float x = static_cast<float>(dev_particle[idx].position.x);
        float y = static_cast<float>(dev_particle[idx].position.y);
        float z = static_cast<float>(dev_particle[idx].position.z);

        dev_treenode[idx].cartesian.x = y*sin(z)*cos(x);
        dev_treenode[idx].cartesian.y = y*sin(z)*sin(x);
        dev_treenode[idx].cartesian.z = y*cos(z);
        dev_treenode[idx].index_old   = idx;
    }
}

// =========================================================================================================================

__global__
void rngs_par_init (curandState *dev_rngs_par, int seed) // the seed is set to 0 by default in cudust.cuh
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        curand_init(seed, idx, 0, &dev_rngs_par[idx]);
    }
}

__global__
void rngs_grd_init (curandState *dev_rngs_grd, int seed) // the seed is set to 1 by default in cudust.cuh
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {
        curand_init(seed, idx, 0, &dev_rngs_grd[idx]);
    }
}