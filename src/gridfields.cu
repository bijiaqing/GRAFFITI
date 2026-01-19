#include "cudust.cuh"

// =========================================================================================================================

__global__
void dustdens_init (real *dev_dustdens)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    	
    if (idx < N_GRD)
    {
        dev_dustdens[idx] = 0.0;
    }
}

#ifdef RADIATION

__global__
void optdepth_init (real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    	
    if (idx < N_GRD)
    {
        dev_optdepth[idx] = 0.0;
    }
}

#endif // RADIATION

// =========================================================================================================================

template <GridFieldType field_type> __device__
void _particle_to_grid_core(real *dev_grid, const swarm *dev_particle, int idx)
{
    real loc_x = _get_loc_x(dev_particle[idx].position.x);
    real loc_y = _get_loc_y(dev_particle[idx].position.y);
    real loc_z = _get_loc_z(dev_particle[idx].position.z);

    bool in_x = loc_x >= 0.0 && loc_x < static_cast<real>(N_X);
    bool in_y = loc_y >= 0.0 && loc_y < static_cast<real>(N_Y);
    bool in_z = loc_z >= 0.0 && loc_z < static_cast<real>(N_Z);

    if (!(in_x && in_y && in_z))
    {
        return; // particle is out of the grid, do nothing
    }

    int idx_cell = static_cast<int>(loc_z)*NG_XY + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);
    auto [next_x, next_y, next_z, frac_x, frac_y, frac_z] = _3d_interp_midpt_y(loc_x, loc_y, loc_z);

    real size = dev_particle[idx].par_size;    
    real weight = 0.0;

    #ifdef RADIATION
    if (field_type == OPTDEPTH)
    {
        weight  = _get_dust_mass(size);
        weight *= dev_particle[idx].par_numr;
        weight *= KAPPA_0*(S_0 / size); // cross section per unit mass
    }
    else
    #endif // RADIATION
    if (field_type == DUSTDENS)
    {
        weight  = _get_dust_mass(size);
        weight *= dev_particle[idx].par_numr;
    }

    atomicAdd(&dev_grid[idx_cell                           ], (1.0 - frac_x)*(1.0 - frac_y)*(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell + next_x                  ],        frac_x *(1.0 - frac_y)*(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell          + next_y         ], (1.0 - frac_x)*       frac_y *(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell + next_x + next_y         ],        frac_x *       frac_y *(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell                   + next_z], (1.0 - frac_x)*(1.0 - frac_y)*       frac_z *weight);
    atomicAdd(&dev_grid[idx_cell + next_x          + next_z],        frac_x *(1.0 - frac_y)*       frac_z *weight);
    atomicAdd(&dev_grid[idx_cell          + next_y + next_z], (1.0 - frac_x)*       frac_y *       frac_z *weight);
    atomicAdd(&dev_grid[idx_cell + next_x + next_y + next_z],        frac_x *       frac_y *       frac_z *weight);
}

#ifdef RADIATION

__global__
void optdepth_scat (real *dev_optdepth, const swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        _particle_to_grid_core <OPTDEPTH> (dev_optdepth, dev_particle, idx);
    }
}

#endif // RADIATION

__global__
void dustdens_scat (real *dev_dustdens, const swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        _particle_to_grid_core <DUSTDENS> (dev_dustdens, dev_particle, idx);
    }
}

// =========================================================================================================================

__device__ __forceinline__
real _get_grid_volume (int idx, real *y0_ptr = nullptr, real *dy_ptr = nullptr)
{
    int idx_x = idx % N_X;
    int idx_y = (idx % NG_XY - idx_x) / N_X;
    int idx_z = (idx - idx_y*N_X - idx_x) / NG_XY;

    bool enable_x = (N_X > 1);
    bool enable_z = (N_Z > 1);
    
    real idx_dim = static_cast<real>(enable_x) + static_cast<real>(enable_z) + 1.0;

    real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
    real vol_x = enable_x ? dx : 1.0;
    
    real dy = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
    real y0 = Y_MIN*pow(dy, static_cast<real>(idx_y));
    real vol_y = pow(y0, idx_dim)*(pow(dy, idx_dim) - 1.0) / idx_dim;
    
    real dz = (Z_MAX - Z_MIN) / static_cast<real>(N_Z);
    real z0 = Z_MIN + dz*static_cast<real>(idx_z);
    real vol_z = enable_z ? (cos(z0) - cos(z0 + dz)) : 1.0;
    
    if (y0_ptr) *y0_ptr = y0;
    if (dy_ptr) *dy_ptr = dy;
    
    return vol_x*vol_y*vol_z;
}

#ifdef RADIATION

__global__
void optdepth_calc (real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {	
        real y0, dy;
        real volume = _get_grid_volume(idx, &y0, &dy);
        dev_optdepth[idx] /= volume;
        dev_optdepth[idx] *= y0*(dy - 1.0); // prepare for radial integration
    }
}

#endif // RADIATION

__global__
void dustdens_calc (real *dev_dustdens)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {	
        real volume = _get_grid_volume(idx);
        dev_dustdens[idx] /= volume;
    }
}

// =========================================================================================================================

#ifdef RADIATION

__global__
void optdepth_csum (real *dev_optdepth) // cumulative sum in the radial direction
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < NG_XZ)
    {	
        int idx_x, idx_z, idx_cell;

        idx_x = idx % N_X;
        idx_z = (idx - idx_x) / N_X;

        // cumulative sum in Y direction
        // no race condition since each thread works on a unique Y row
        for (int i = 1; i < N_Y; i++)
        {
            idx_cell = idx_z*NG_XY + i*N_X + idx_x;
            dev_optdepth[idx_cell] += dev_optdepth[idx_cell - N_X];
        }
    }
}

__global__
void optdepth_mean (real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < NG_YZ)
    {	
        int idx_y, idx_z, idx_cell;

        idx_y = idx % N_Y;
        idx_z = (idx - idx_y) / N_Y;

        real optdepth_sum = 0.0;

        // summation in X direction
        // no race condition since each thread works on a unique X row
        for (int i = 0; i < N_X; i++)
        {
            idx_cell = idx_z*NG_XY + idx_y*N_X + i;
            optdepth_sum += dev_optdepth[idx_cell];
        }

        real optdepth_avg = optdepth_sum / N_X;

        for (int j = 0; j < N_X; j++)
        {
            idx_cell = idx_z*NG_XY + idx_y*N_X + j;
            dev_optdepth[idx_cell] = optdepth_avg;
        }
    }
}

#endif // RADIATION

// =========================================================================================================================