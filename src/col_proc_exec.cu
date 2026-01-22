#ifdef COLLISION

#include "cudust_kern.cuh"
#include "helpers_collision.cuh" 

// =========================================================================================================================
// Kernel: col_proc_exec
// Purpose: Execute collision process (merge/fragmentation) for selected particles
// Dependencies: helpers_collision.cuh (provides candidatelist, _get_col_rate_ij, _get_vrel via KernelType template)
// =========================================================================================================================

__global__
void col_proc_exec (swarm *dev_particle, curs *dev_rs_swarm, real *dev_col_expt, 
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
        
        real vol_cell = _get_grid_volume(idx_cell); // volume of the cell that particle i is in
        
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

            // if particle idx_old_j *first* makes expt >= rand, it is the one that idx_old_i is going to collide with
            while (col_expt_ij < col_rand_ij && j < KNN_SIZE)
            {
                idx_query = query_result.returnIndex(j);

                if (idx_query != -1)
                {
                    idx_old_j = dev_treenode[idx_query].index_old;
                    col_expt_ij += _get_col_rate_ij<static_cast<KernelType>(K_COAG)>(dev_particle, idx_old_i, idx_old_j, vol_cell);
                }

                j++;
            }

            real v_rel = 0.0;
            // real v_rel = _get_vrel(dev_particle, idx_old_i, idx_old_j);

            if (v_rel <= V_FRAG)
            {
                // collide with idx_old_j and MERGE
                real size_i = dev_particle[idx_old_i].par_size;
                real size_j = dev_particle[idx_old_j].par_size;
                real size_k = cbrt(size_i*size_i*size_i + size_j*size_j*size_j);

                dev_particle[idx_old_i].par_size  = size_k;
                dev_particle[idx_old_i].par_numr *= (size_i*size_i*size_i) / (size_k*size_k*size_k);
            }
            else
            {
                // collide with idx_old_j and BREAK-UP
                real size_i = dev_particle[idx_old_i].par_size;
                real size_j = dev_particle[idx_old_j].par_size;
                real size_k = cbrt(size_i*size_i*size_i + size_j*size_j*size_j);
                real rand   = curand_uniform_double(&rs_swarm);  // distribution in (0, 1]

                // determine the fragmentation outcome using inverse transform sampling
                // see https://en.wikipedia.org/wiki/Inverse_transform_sampling
                // since n(s) ds = N*s^-3.5 ds is the grain size distribution and 
                // integrate( m(s)*n(s) ds) = mi + mj = si^3 + sj^3, where s is 0 < s < (si^3 + sj^3)^(1/3)
                // we get N = 0.5*(si^3 + sj^3)^(5/6), so that n(s) ds = 0.5*(si^3 + sj^3)^(5/6) s^-3.5 ds
                // since the probability distribution function (PDF) P(s) = m(s)*n(s) / (si^3 + sj^3), 
                // the cumulative distribution function (CDF) becomes C(s) = integrate( P(s) ds) = (si^3 + sj^3)^(-1/6) s^(1/2)
                // which naturally satisfies C(s) in [0,1] when s in [0, (si^3 + sj^3)^(1/3)]
                // Since the definition of inverse function gives C(C^-1(s)) = s,
                // the inverse function of the CDF is C^-1(s) = (si^3 + sj^3)^(1/3) s^2    <--- this is the code below
                // so we draw a random number x from a uniform distribution between 0 and 1,
                // put it in C^-1(x) will give a random s that satisfies the PDF P(s) = m(s)*n(s) / (si^3 + sj^3)

                size_k *= rand*rand;

                dev_particle[idx_old_i].par_size  = size_k;
                dev_particle[idx_old_i].par_numr *= (size_i*size_i*size_i) / (size_k*size_k*size_k);
            }

            dev_rs_swarm[idx_old_i] = rs_swarm;
        }
        
    }
}

#endif // COLLISION
