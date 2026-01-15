#include "cudust.cuh"

#ifdef COLLISION

#include <cassert>          // for assert
#include "cukd/knn.h"       // for cukd::cct::knn, cukd::HeapCandidateList

using candidatelist = cukd::HeapCandidateList<KNN_SIZE>;

// Coagulation kernel types
enum KernelType { CONSTANT_KERNEL = 0, LINEAR_KERNEL = 1, PRODUCT_KERNEL = 2 };

// =========================================================================================================================

__global__
void col_rate_init (real *dev_col_rate, real *dev_col_expt, real *dev_col_rand)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {
        dev_col_rate[idx] = 0.0;
        dev_col_expt[idx] = 0.0;
        dev_col_rand[idx] = 0.0;
    }
}

// =========================================================================================================================

template <KernelType kernel> __device__ __forceinline__
real _get_col_rate_ij (const swarm *dev_particle, int idx_old_i, int idx_old_j)
{
    // text mainly from Drazkowska et al. 2013:
    // we assume that a limited number n representative particles represent all N physical particles
    // each representative particle i describes a swarm of N_i identical physical particles
    // as n << N, we only need to consider the collisions between representative and non-representative particles
    // the probability of a collision between particles i and j is determined as 
    // lambda_ij = N_j * K_ij / V, where K_ij is the coagulation kernel and V is the volume of the cell
    
    real lam_ij = LAMBDA_0;
    real numr_j = dev_particle[idx_old_j].par_numr;
    real volume = 1.0; // to be replaced by actual cell volume if needed

    if constexpr (kernel == CONSTANT_KERNEL)
    {
        lam_ij *= 1.0;
        return lam_ij * numr_j / volume;
    }
    else if constexpr (kernel == LINEAR_KERNEL)
    {
        real size_i = dev_particle[idx_old_i].par_size;
        real size_j = dev_particle[idx_old_j].par_size;

        // m_i + m_j
        lam_ij *= 0.5*RHO_0*(size_i*size_i*size_i + size_j*size_j*size_j);
        return lam_ij * numr_j / volume;
    }
    else if constexpr (kernel == PRODUCT_KERNEL)
    {
        real size_i = dev_particle[idx_old_i].par_size;
        real size_j = dev_particle[idx_old_j].par_size;
        
        // m_i * m_j
        lam_ij *= RHO_0*(size_i*size_i*size_i)*RHO_0*(size_j*size_j*size_j);
        return lam_ij * numr_j / volume;
    }
    else
    {
        // Invalid kernel type - print error only once from first thread
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("ERROR: Invalid COL_KERNEL value = %d \n", static_cast<int>(kernel));
        }
        
        assert(false);
        return 0.0; // unreachable, but prevents compiler warning
    }
}

// =========================================================================================================================

__global__
void col_rate_calc (real *dev_col_rate, swarm *dev_particle, const tree *dev_treenode, const bbox *dev_boundbox)
{
    // calculates the total collision rate for each cell, to help determine whether a collision is going to happen in the cell
    // it goes through each particle (i) and calculates the colision rate of it, then adds the rate to the corresponding cell
    
    int idx_tree = threadIdx.x+blockDim.x*blockIdx.x; // this is the index for the particle (idx_old_i) on the k-d tree

    if (idx_tree < N_PAR)
    {
        int idx_old_i = dev_treenode[idx_tree].index_old;
        
        real loc_x = _get_loc_x(dev_particle[idx_old_i].position.x);
        real loc_y = _get_loc_y(dev_particle[idx_old_i].position.y);
        real loc_z = _get_loc_z(dev_particle[idx_old_i].position.z);

        bool in_x = loc_x >= 0.0 && loc_x < static_cast<real>(N_X);
        bool in_y = loc_y >= 0.0 && loc_y < static_cast<real>(N_Y);
        bool in_z = loc_z >= 0.0 && loc_z < static_cast<real>(N_Z);

        if (!(in_x && in_y && in_z)) return; // particle is out of bounds, do nothing

        int idx_cell = static_cast<int>(loc_z)*NG_XY + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);
        
        candidatelist query_result(MAX_DIST);
        cukd::cct::knn <candidatelist, tree, tree_traits> (query_result, 
            dev_treenode[idx_tree].cartesian, *dev_boundbox, dev_treenode, N_PAR);

        real col_rate_ij = 0.0; // collision rate between particle i and j
        real col_rate_i  = 0.0; // total collision rate for particle i

        int idx_old_j, idx_query;

        for(int j = 0; j < KNN_SIZE; j++)
        {
            col_rate_ij = 0.0;
            idx_query = query_result.returnIndex(j);

            if (idx_query != -1) // if the j-th nearest neighbor exists within MAX_DIST
            {
                idx_old_j = dev_treenode[idx_query].index_old;
                col_rate_ij = _get_col_rate_ij<static_cast<KernelType>(COL_KERNEL)>(dev_particle, idx_old_i, idx_old_j);
            }

            col_rate_i += col_rate_ij;
        }

        dev_particle[idx_old_i].col_rate = col_rate_i;
        atomicAdd(&dev_col_rate[idx_cell], col_rate_i);
    }
}

// =========================================================================================================================
// col_rate_peak has been inlined in main.cu using Thrust
// Legacy implementation using atomicMax on float (commented out):
// 
// __global__
// void col_rate_peak (float *dev_max_rate, const real *dev_col_rate)
// {
//     int idx = threadIdx.x+blockDim.x*blockIdx.x;
// 
//     if (idx < N_GRD)
//     {
//         // the line below applies atomicMax to float numbers
//         // it reinterprets the 32-bit float's memory representation as a 32-bit int and vice versa
//         // this only works correctly for positive numbers!
//         __int_as_float(atomicMax((int*)dev_max_rate, __float_as_int(static_cast<float>(dev_col_rate[idx]))));
//     }
// }
// =========================================================================================================================

__global__
void col_flag_calc (int *dev_col_flag, curs *dev_rs_grids, real *dev_col_rand, const real *dev_col_rate, real dt_col)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {
        curs rs_grids = dev_rs_grids[idx];
        
        // if expt > rand, there will be a collision in the cell in the current timestep
        real col_rand = curand_uniform_double(&rs_grids); // the range is (0,1]
        real col_expt = dt_col*dev_col_rate[idx]; // total expected number of collisions in the cell within the current timestep

        if (col_expt >= col_rand)
        {
            dev_col_flag[idx] = 1;

            // need to pass on dev_col_rand to determine which particle in the cell collides
            dev_col_rand[idx] = dev_col_rate[idx]*curand_uniform_double(&rs_grids);
        }
        else
        {
            dev_col_flag[idx] = 0;
        }

        dev_rs_grids[idx] = rs_grids;
    }
}

// =========================================================================================================================

__global__
void run_collision (swarm *dev_particle, curs *dev_rs_swarm, real *dev_col_expt, 
    const real *dev_col_rand, const int *dev_col_flag, const tree *dev_treenode, const bbox *dev_boundbox)
{
    // calculates the outcome of collision and update the property of the particle
    // if the collision flag of the cell that the particle is in is 1, it examines whether the particle is going to collide
    // if the particle *first* makes real >= rand, it is the particle that is going to collide in the cell
    
    int idx_tree = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx_tree < N_PAR)
    {
        int idx_old_i = dev_treenode[idx_tree].index_old;

        real loc_x = _get_loc_x(dev_particle[idx_old_i].position.x);
        real loc_y = _get_loc_y(dev_particle[idx_old_i].position.y);
        real loc_z = _get_loc_z(dev_particle[idx_old_i].position.z);

        bool in_x = loc_x >= 0.0 && loc_x < static_cast<real>(N_X);
        bool in_y = loc_y >= 0.0 && loc_y < static_cast<real>(N_Y);
        bool in_z = loc_z >= 0.0 && loc_z < static_cast<real>(N_Z);

        if (!(in_x && in_y && in_z))
        {
            return; // particle is out of bounds, do nothing
        }

        int idx_cell = static_cast<int>(loc_z)*NG_XY + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);

        if (dev_col_flag[idx_cell] == 0)
        {
            return; // no collision in the cell, do nothing
        }
        
        // collision rate of the particle
        real col_rate_i = dev_particle[idx_old_i].col_rate;
        
        // collision parameters of the cell that the particle is in
        real col_rand = dev_col_rand[idx_cell]; // dev_col_rand is now a value between 0 and collision rate of the cell
        real col_expt = atomicAdd(&dev_col_expt[idx_cell], col_rate_i); // atomicAdd returns the value BEFORE the addition

        if (col_expt < col_rand && col_expt + col_rate_i >= col_rand)   // if this is the particle that is going to collide
        {
            curs rs_swarm = dev_rs_swarm[idx_old_i];
            
            candidatelist query_result(MAX_DIST);
            cukd::cct::knn <candidatelist, tree, tree_traits> (query_result, 
                dev_treenode[idx_tree].cartesian, *dev_boundbox, dev_treenode, N_PAR);

            int idx_old_j, idx_query, j = 0;

            // col_rand_ij is now a value between 0 and the total collision rate of the particle
            real col_rand_ij = col_rate_i*curand_uniform_double(&rs_swarm);
            real col_expt_ij = 0.0;

            real rel_velocity = 0.0; 
            // real rms_velocity = 0.0;
            // int rms_count = 0;

            // if particle idx_old_j *first* makes expt > rand, it is the one that idx_old_i is going to collide with
            while (col_expt_ij < col_rand_ij && j < KNN_SIZE)
            {
                idx_query = query_result.returnIndex(j);

                if (idx_query != -1)
                {
                    idx_old_j = dev_treenode[idx_query].index_old;
                    col_expt_ij += _get_col_rate_ij<static_cast<KernelType>(COL_KERNEL)>(dev_particle, idx_old_i, idx_old_j);
                    // rms_velocity += _sqr_velocity(dev_particle, idx_old_j);
                    // rms_count++;
                }

                j++;
            }

            // // calculate the rms velocity
            // rms_velocity /= rms_count;
            // rms_velocity  = sqrt(rms_velocity);

            if (rel_velocity <= V_FRAG)
            {
                // collide with idx_old_j and MERGE
                real size_i = dev_particle[idx_old_i].par_size;
                real size_j = dev_particle[idx_old_j].par_size;
                real size_k = cbrt(size_i*size_i*size_i + size_j*size_j*size_j);

                dev_particle[idx_old_i].par_size  = size_k;
                dev_particle[idx_old_i].par_numr *= (size_i/size_k)*(size_i/size_k)*(size_i/size_k);
            }
            else
            {
                // collide with idx_old_j and BREAK-UP
                real size_i = dev_particle[idx_old_i].par_size;
                real size_j = dev_particle[idx_old_j].par_size;
                real size_k = cbrt(size_i*size_i*size_i + size_j*size_j*size_j);
                real rand   = curand_uniform_double(&rs_swarm);  // distribution in (0, 1]

                // determine the fragmentation outcome using **inverse transform sampling**
                // see https://en.wikipedia.org/wiki/Inverse_transform_sampling
                // since n(s) ds = N*s^-3.5 ds is the grain size distribution and 
                // integrate( m(s)*n(s) ds) = mi + mj = si^3 + sj^3, where s is 0 < s < (si^3 + sj^3)^(1/3)
                // we get N = 0.5*(si^3 + sj^3)^(5/6), so that n(s) ds = 0.5*(si^3 + sj^3)^(5/6) s^-3.5 ds
                // since the probability distribution function (PDF) P(s) = m(s)*n(s) / (si^3 + sj^3), 
                // the cumulative distribution function (CDF) becomes C(s) = integrate( P(s) ds) = (si^3 + sj^3)^(-1/6) s^(1/2)
                // which naturally satisfies C(s) in [0,1] when s in [0, (si^3 + sj^3)^(1/3)]
                // Since the definition of inverse fucntion gives C(C^-1(s)) = s,
                // the inverse fucntion of the CDF is C^-1(s) = (si^3 + sj^3)^(1/3) s^2    <--- this is the code below
                // so we draw a random number x from a unifrom distribution between 0 and 1,
                // put it in C^-1(x) will give a random s that satisfies the PDF P(s) = m(s)*n(s) / (si^3 + sj^3)

                size_k *= rand*rand;

                dev_particle[idx_old_i].par_size  = size_k;
                dev_particle[idx_old_i].par_numr *= (size_i/size_k)*(size_i/size_k)*(size_i/size_k);
            }

            dev_rs_swarm[idx_old_i] = rs_swarm;
        }
        
    }
}

#endif // COLLISION

// =========================================================================================================================