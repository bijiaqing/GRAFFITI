#ifdef COLLISION

#include "cudust_kern.cuh"
#include "helpers_collision.cuh"

// =========================================================================================================================
// Kernel: col_rate_calc
// Purpose: Calculate total collision rate for each grid cell
// Dependencies: helpers_collision.cuh (provides candidatelist, KernelType, vrel helpers, _get_col_rate_ij)
// =========================================================================================================================

__global__
void col_rate_calc (real *dev_col_rate, swarm *dev_particle, const tree *dev_col_tree, const bbox *dev_boundbox)
{
    // calculates the total collision rate for each cell, to help determine whether a collision is going to happen in the cell
    // it goes through each particle (i) and calculates the colision rate of it, then adds the rate to the corresponding cell
    
    int idx_tree = threadIdx.x+blockDim.x*blockIdx.x; // this is the index for the particle (idx_old_i) on the k-d tree

    if (idx_tree < N_P)
    {
        int idx_old_i = dev_col_tree[idx_tree].index_old;
        
        real loc_x = _get_loc_x(dev_particle[idx_old_i].position.x);
        real loc_y = _get_loc_y(dev_particle[idx_old_i].position.y);
        real loc_z = _get_loc_z(dev_particle[idx_old_i].position.z);

        bool in_x = loc_x >= 0.0 && loc_x < static_cast<real>(N_X);
        bool in_y = loc_y >= 0.0 && loc_y < static_cast<real>(N_Y);
        bool in_z = loc_z >= 0.0 && loc_z < static_cast<real>(N_Z);

        if (!(in_x && in_y && in_z)) return; // particle is out of bounds, do nothing

        int  idx_cell = static_cast<int>(loc_z)*NG_XY + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);

        // maximum search distance for KNN neighbor search defined by gas scale height
        real polar_R = dev_particle[idx_old_i].position.y*sin(dev_particle[idx_old_i].position.z);
        float max_search_dist = static_cast<float>(_get_hgas(polar_R)*polar_R);
        
        candidatelist query_result(max_search_dist);
        cukd::cct::knn <candidatelist, tree, tree_traits> (query_result, 
            dev_col_tree[idx_tree].cartesian, *dev_boundbox, dev_col_tree, N_P);

        real col_rate_ij = 0.0; // collision rate between particle i and j
        real col_rate_i  = 0.0; // total collision rate for particle i

        float dist2 = 0.0f;
        float max_dist2 = 0.0f;

        int idx_old_j, idx_query;

        for(int j = 0; j < N_K; j++)
        {
            col_rate_ij = 0.0;
            idx_query = query_result.returnIndex(j);

            if (idx_query != -1) // if the j-th nearest neighbor exists
            {
                dist2 = query_result.returnDist2(j);
                max_dist2 = fmaxf(max_dist2, dist2);
                
                idx_old_j = dev_col_tree[idx_query].index_old;
                col_rate_ij = _get_col_rate_ij <static_cast<KernelType>(COAG_KERNEL)> (dev_particle, idx_old_i, idx_old_j);
            }

            col_rate_i += col_rate_ij;
        }

        real radius = sqrtf(static_cast<real>(max_dist2));
        real volume = (4.0/3.0)*M_PI*radius*radius*radius;

        volume = 1.0; // for testing purposes

        col_rate_i /= volume;

        dev_particle[idx_old_i].max_dist = radius;
        dev_particle[idx_old_i].col_rate = col_rate_i;
        atomicAdd(&dev_col_rate[idx_cell], col_rate_i);
    }
}

// =================================================================================================================================

#endif // COLLISION
