#include "cudust.cuh"
#include "cukd/knn.h"   // for cukd::cct::knn, cukd::HeapCandidateList

using candidatelist = cukd::HeapCandidateList<KNN_SIZE>;

// =========================================================================================================================

__global__
void col_rate_init (real *dev_col_rate, real *dev_col_rand, real *dev_col_real, float *dev_max_rate)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {
        dev_col_rate[idx] = 0.0;
        dev_col_rand[idx] = 0.0;
        dev_col_real[idx] = 0.0;

        if (idx == 0) 
        {
            *dev_max_rate = 0.0;
        }
    }
}

// =========================================================================================================================

__device__
real _col_rate_ij (swarm *dev_particle, int idx_old_i, int idx_old_j)
{
    // constant kernel
    real col_rate_ij = LAMBDA_0*dev_particle[idx_old_j].par_numr;
    
    // // linear kernel
    // real size_i = dev_particle[idx_old_i].par_size;
    // real size_j = dev_particle[idx_old_j].par_size;
    // real col_rate_ij = LAMBDA_0*RHO_0*(size_i*size_i*size_i + size_j*size_j*size_j);
    
    return col_rate_ij;
}

// __device__
// real _sqr_velocity (swarm *dev_particle, int idx_old_j)
// {
//     real rad   = dev_particle[idx_old_j].position.y;
//     real col   = dev_particle[idx_old_j].position.z;
//     real l_azi = dev_particle[idx_old_j].velocity.x;
//     real v_rad = dev_particle[idx_old_j].velocity.y;
//     real l_col = dev_particle[idx_old_j].velocity.z;

//     real sqr_velocity = 0.0;

//     sqr_velocity += (l_azi / rad / sin(col) - sqrt(G*M_S / rad / sin(col)))*(l_azi / rad / sin(col) - sqrt(G*M_S / rad / sin(col)));
//     sqr_velocity += v_rad*v_rad;
//     sqr_velocity += l_col*l_col / rad / rad;
    
//     return sqr_velocity;
// }

// =========================================================================================================================

__global__
void col_rate_calc (swarm *dev_particle, tree *dev_treenode, real *dev_col_rate, boxf *dev_boundbox)
{
    // calculates the total collision rate for each cell, to help determine whether a collision is going to happen in the cell
    // it goes through each particle (i) and calculates the colision rate of it, then adds the rate to the corresponding cell
    
    int idx_tree = threadIdx.x+blockDim.x*blockIdx.x; // this is the index for the particle (idx_old_i) on the k-d tree

    if (idx_tree < N_PAR)
    {
        int idx_old_i = dev_treenode[idx_tree].index_old;
        
        real loc_x = (N_X > 1) ? (static_cast<real>(N_X)*   (dev_particle[idx_old_i].position.x - X_MIN) /    (X_MAX - X_MIN)) : 0.0;
        real loc_y = (N_Y > 1) ? (static_cast<real>(N_Y)*log(dev_particle[idx_old_i].position.y / Y_MIN) / log(Y_MAX / Y_MIN)) : 0.0;
        real loc_z = (N_Z > 1) ? (static_cast<real>(N_Z)*   (dev_particle[idx_old_i].position.z - Z_MIN) /    (Z_MAX - Z_MIN)) : 0.0;

        bool in_x = loc_x >= 0.0 && loc_x < static_cast<real>(N_X);
        bool in_y = loc_y >= 0.0 && loc_y < static_cast<real>(N_Y);
        bool in_z = loc_z >= 0.0 && loc_z < static_cast<real>(N_Z);

        if (!(in_x && in_y && in_z)) return; // particle is out of bounds, do nothing

        int idx_cell = static_cast<int>(loc_z)*NG_XY + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);
        
        candidatelist query_result(MAX_DIST);
        cukd::cct::knn<candidatelist, tree, tree_traits>(query_result, dev_treenode[idx_tree].cartesian, *dev_boundbox, dev_treenode, N_PAR);

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
                col_rate_ij = _col_rate_ij(dev_particle, idx_old_i, idx_old_j);
            }

            col_rate_i += col_rate_ij;
        }

        dev_particle[idx_old_i].col_rate = col_rate_i;
        atomicAdd(&dev_col_rate[idx_cell], col_rate_i);
    }
}

// =========================================================================================================================

__global__
void col_rate_peak (real *dev_col_rate, float *dev_max_rate)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {
        // the line below applies atomicMax to float numbers, see https://stackoverflow.com/questions/17399119
        __int_as_float(atomicMax((int*)dev_max_rate, __float_as_int(static_cast<float>(dev_col_rate[idx]))));
    }
}

// =========================================================================================================================

__global__
void col_flag_calc (real *dev_col_rate, real *dev_col_rand, int *dev_col_flag, real *dev_timestep, curandState *dev_rngs_grd)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {
        curandState rngs_grd = dev_rngs_grd[idx];
        
        // if real > rand, there will be a collision in the cell in the current timestep
        real col_rand = curand_uniform_double(&rngs_grd); // the range is (0,1]
        real col_real = (*dev_timestep)*dev_col_rate[idx];

        if (col_real >= col_rand)
        {
            dev_col_flag[idx] = 1;

            // need to pass on dev_col_rand to determine which particle in the cell collides
            dev_col_rand[idx] = dev_col_rate[idx]*curand_uniform_double(&rngs_grd);
        }
        else
        {
            dev_col_flag[idx] = 0;
        }

        dev_rngs_grd[idx] = rngs_grd;
    }
}

// =========================================================================================================================

__global__
void particle_evol (swarm *dev_particle, tree *dev_treenode, int *dev_col_flag, real *dev_col_rand, real *dev_col_real, boxf *dev_boundbox, curandState *dev_rngs_par)
{
    // calculates the outcome of collision and update the property of the particle
    // if the collision flag of the cell that the particle is in is 1, it examines whether the particle is going to collide
    // if the particle *first* makes real >= rand, it is the particle that is going to collide in the cell
    
    int idx_tree = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx_tree < N_PAR)
    {
        int idx_old_i = dev_treenode[idx_tree].index_old;

        real loc_x = (N_X > 1) ? (static_cast<real>(N_X)*   (dev_particle[idx_old_i].position.x - X_MIN) /    (X_MAX - X_MIN)) : 0.0;
        real loc_y = (N_Y > 1) ? (static_cast<real>(N_Y)*log(dev_particle[idx_old_i].position.y / Y_MIN) / log(Y_MAX / Y_MIN)) : 0.0;
        real loc_z = (N_Z > 1) ? (static_cast<real>(N_Z)*   (dev_particle[idx_old_i].position.z - Z_MIN) /    (Z_MAX - Z_MIN)) : 0.0;

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
        real col_rand = dev_col_rand[idx_cell];                         // dev_col_rand is now a value between 0 and the total collision rate of the cell
        real col_real = atomicAdd(&dev_col_real[idx_cell], col_rate_i); // atomicAdd returns the value BEFORE the addition

        if (col_real < col_rand && col_real + col_rate_i >= col_rand)   // if this is the particle that is going to collide
        {
            curandState rngs_par = dev_rngs_par[idx_old_i];
            
            candidatelist query_result(MAX_DIST);
            cukd::cct::knn<candidatelist, tree, tree_traits>(query_result, dev_treenode[idx_tree].cartesian, *dev_boundbox, dev_treenode, N_PAR);

            int idx_old_j, idx_query, j = 0;

            // col_rand_ij is now a value between 0 and the total collision rate of the particle
            real col_rand_ij = col_rate_i*curand_uniform_double(&rngs_par);
            real col_real_ij = 0.0;

            real rel_velocity = 0.0; 
            real rms_velocity = 0.0;
            int rms_count = 0;

            // if particle idx_old_j *first* makes real > rand, it is the one that idx_old_i is going to collide with
            while (col_real_ij < col_rand_ij && j < KNN_SIZE)
            {
                idx_query = query_result.returnIndex(j);

                if (idx_query != -1)
                {
                    idx_old_j = dev_treenode[idx_query].index_old;
                    col_real_ij += _col_rate_ij(dev_particle, idx_old_i, idx_old_j);
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
                real rand   = curand_uniform_double(&rngs_par);  // distribution in (0, 1]

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

            dev_rngs_par[idx_old_i] = rngs_par;
        }
        
    }
}

// // =========================================================================================================================

// __device__
// real get_distance (swarm *dev_particle, int idx_1, int idx_2)
// {
//     real azi_1, rad_1, col_1;
//     real azi_2, rad_2, col_2;

//     azi_1  = dev_particle[idx_1].position.x;
//     rad_1  = dev_particle[idx_1].position.y;
//     col_1  = dev_particle[idx_1].position.z;

//     azi_2  = dev_particle[idx_2].position.x;
//     rad_2  = dev_particle[idx_2].position.y;
//     col_2  = dev_particle[idx_2].position.z;

//     return sqrt(rad_1*rad_1 + rad_2*rad_2 - 2.0*rad_1*rad_2*(sin(col_1)*sin(col_2)*cos(azi_1 - azi_2) + cos(col_1)*cos(col_2)));
// }

// __device__
// real get_col_rate (swarm *dev_particle, int idx_1, int idx_2, real dist_max)
// {
//     real azi, rad, col, v_azi, v_rad, v_col;
//     real vx_1, vy_1, vz_1, size_1, mass_1;
//     real vx_2, vy_2, vz_2, size_2, mass_2;
//     real delta_v, lambda;

//     azi    = dev_particle[idx_1].position.x;
//     rad    = dev_particle[idx_1].position.y;
//     col    = dev_particle[idx_1].position.z;
//     v_azi  = dev_particle[idx_1].velocity.x / rad / sin(col);
//     v_rad  = dev_particle[idx_1].velocity.y;
//     v_col  = dev_particle[idx_1].velocity.z / rad;
//     size_1 = dev_particle[idx_1].par_size;
//     mass_1 = dev_particle[idx_1].par_numr*size_1*size_1*size_1;

//     vx_1 = v_rad*sin(col)*cos(azi) + v_col*cos(col)*cos(azi) - v_azi*sin(azi);
//     vy_1 = v_rad*sin(col)*sin(azi) + v_col*cos(col)*sin(azi) + v_azi*cos(azi);
//     vz_1 = v_rad*cos(col)          + v_col*sin(col);

//     azi    = dev_particle[idx_2].position.x;
//     rad    = dev_particle[idx_2].position.y;
//     col    = dev_particle[idx_2].position.z;
//     v_azi  = dev_particle[idx_2].velocity.x / rad / sin(col);
//     v_rad  = dev_particle[idx_2].velocity.y;
//     v_col  = dev_particle[idx_2].velocity.z / rad;
//     size_2 = dev_particle[idx_2].par_size;
//     mass_2 = dev_particle[idx_2].par_numr*size_2*size_2*size_2;

//     vx_2 = v_rad*sin(col)*cos(azi) + v_col*cos(col)*cos(azi) - v_azi*sin(azi);
//     vy_2 = v_rad*sin(col)*sin(azi) + v_col*cos(col)*sin(azi) + v_azi*cos(azi);
//     vz_2 = v_rad*cos(col)          + v_col*sin(col);

//     delta_v = sqrt((vx_1 - vx_2)*(vx_1 - vx_2) + (vy_1 - vy_2)*(vy_1 - vy_2) + (vz_1 - vz_2)*(vz_1 - vz_2));
    
//     lambda  = (size_1 + size_2)*(size_1 + size_2)*delta_v / dist_max / dist_max / dist_max; // the choice of V is not very physical...
//     lambda *= mass_2 / size_2 / size_2 / size_2;
    
//     if (mass_2 < 0.1*mass_1)
//     {
//         lambda /= 0.1*mass_1 / mass_2;
//     }
    
//     return lambda;
// }

// __global__
// void col_rate_calc (swarm *dev_particle, swarm_tmp *dev_tmp_info, tree *dev_treenode, int *dev_col_rate, 
//     const cukd::box_t<float3> *dev_boundbox)
// {
//     int idx = threadIdx.x+blockDim.x*blockIdx.x;

//     if (idx < N_PAR)
//     {
//         // float max_query_dist = 0.05;    // change to gas scale height if needed

//         using candidatelist = cukd::HeapCandidateList<KNN_SIZE>;
//         candidatelist query_result(static_cast<float>(MAX_DIST));
//         cukd::cct::knn <candidatelist, tree, tree_traits> (query_result, dev_treenode[idx].cartesian, *dev_boundbox, dev_treenode, N_PAR);

//         real col_rate_ij = 0.0; // collision rate between particle i and j
//         real col_rate_i  = 0.0; // total collision rate for particle i

//         int tmp_idx   = query_result.returnIndex(0); // this is always the farthest one, if exists
//         int idx_old_i = dev_treenode[idx].index_old;
//         int idx_old_j;
//         real dist_max = MAX_DIST;

//         if (tmp_idx != -1)
//         {
//             idx_old_j = dev_treenode[tmp_idx].index_old;
//             dist_max = get_distance(dev_particle, idx_old_i, idx_old_j);
//         }

//         for(int j = 0; j < KNN_SIZE; j++)
//         {
//             col_rate_ij = 0.0;
//             tmp_idx  = query_result.returnIndex(j);

//             if (tmp_idx != -1)
//             {
//                 idx_old_j = dev_treenode[tmp_idx].index_old;
//                 // col_rate_ij = col_rate_NORM*get_col_rate(dev_particle, idx_old_i, idx_old_j, dist_max);
//                 col_rate_ij = 1.0;
//             }

//             col_rate_i += col_rate_ij;
//         }
        
//         // if (idx < 200) printf("%.8e\n", col_rate_i);
        
//         dev_tmp_info[idx_old_i].col_rate_i = col_rate_i;
//         atomicMax(dev_col_rate, static_cast<int>(col_rate_i));
//     }
// }