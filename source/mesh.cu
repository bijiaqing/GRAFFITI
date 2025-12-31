#include <cfloat>   // for DBL_MAX

#include "cudust.cuh"

// =========================================================================================================================

__global__
void optdepth_init (real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    	
    if (idx < N_GRD)
    {
        dev_optdepth[idx] = 0.0;
    }
}

__global__
void dustdens_init (real *dev_dustdens)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    	
    if (idx < N_GRD)
    {
        dev_dustdens[idx] = 0.0;
    }
}

// =========================================================================================================================

template <GridFieldType field_type> __device__
void _particle_to_grid_core(real *dev_grid, swarm *dev_particle, int idx)
{
    real loc_x = (N_X > 1) ? (static_cast<real>(N_X) *    (dev_particle[idx].position.x - X_MIN) /     (X_MAX - X_MIN)) : 0.0;
    real loc_y = (N_Y > 1) ? (static_cast<real>(N_Y) * log(dev_particle[idx].position.y / Y_MIN) /  log(Y_MAX / Y_MIN)) : 0.0;
    real loc_z = (N_Z > 1) ? (static_cast<real>(N_Z) *    (dev_particle[idx].position.z - Z_MIN) /     (Z_MAX - Z_MIN)) : 0.0;

    bool in_x = loc_x >= 0.0 && loc_x < static_cast<real>(N_X);
    bool in_y = loc_y >= 0.0 && loc_y < static_cast<real>(N_Y);
    bool in_z = loc_z >= 0.0 && loc_z < static_cast<real>(N_Z);

    if (!(in_x && in_y && in_z))
    {
        return; // particle is out of the grid, do nothing
    }

    interp result = _linear_interp_cent(loc_x, loc_y, loc_z);

    int  next_x = result.next_x;
    int  next_y = result.next_y;
    int  next_z = result.next_z;
    real frac_x = result.frac_x;
    real frac_y = result.frac_y;
    real frac_z = result.frac_z;

    int idx_cell = static_cast<int>(loc_z)*NG_XY + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);

    real par_size = dev_particle[idx].par_size;
    
    real weight = 0.0;

    if (field_type == OPTDEPTH)
    {
        weight  = RHO_0;
        weight *= dev_particle[idx].par_numr;
        weight *= par_size*par_size*par_size;
        weight *= KAPPA_0*(S_0 / par_size); // cross section per unit mass
    }
    else if (field_type == DUSTDENS)
    {
        weight  = RHO_0;
        weight *= dev_particle[idx].par_numr;
        weight *= par_size*par_size*par_size;
    }
    // Add more field types here in the future

    atomicAdd(&dev_grid[idx_cell                           ], (1.0 - frac_x)*(1.0 - frac_y)*(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell + next_x                  ],        frac_x *(1.0 - frac_y)*(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell          + next_y         ], (1.0 - frac_x)*       frac_y *(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell + next_x + next_y         ],        frac_x *       frac_y *(1.0 - frac_z)*weight);
    atomicAdd(&dev_grid[idx_cell                   + next_z], (1.0 - frac_x)*(1.0 - frac_y)*       frac_z *weight);
    atomicAdd(&dev_grid[idx_cell + next_x          + next_z],        frac_x *(1.0 - frac_y)*       frac_z *weight);
    atomicAdd(&dev_grid[idx_cell          + next_y + next_z], (1.0 - frac_x)*       frac_y *       frac_z *weight);
    atomicAdd(&dev_grid[idx_cell + next_x + next_y + next_z],        frac_x *       frac_y *       frac_z *weight);
}

__global__
void optdepth_enum (real *dev_optdepth, swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        _particle_to_grid_core<OPTDEPTH>(dev_optdepth, dev_particle, idx);
    }
}

__global__
void dustdens_enum (real *dev_dustdens, swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        _particle_to_grid_core<DUSTDENS>(dev_dustdens, dev_particle, idx);
    }
}

// =========================================================================================================================

__device__
void _grid_indices (int idx, int &idx_x, int &idx_y, int &idx_z)
{
    idx_x = idx % N_X;
    idx_y = (idx % NG_XY - idx_x) / N_X;
    idx_z = (idx - idx_y*N_X - idx_x) / NG_XY;
}

__device__
void _grid_volume (int idx_y, int idx_z, real &vol_x, real &vol_y, real &vol_z)
{
    real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
    real dy = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
    real dz = (Z_MAX - Z_MIN) / static_cast<real>(N_Z);

    bool has_x = (N_X > 1);
    bool has_z = (N_Z > 1);

    vol_x = has_x ? dx : 1.0;

    real y_in = Y_MIN*pow(dy, static_cast<real>(idx_y));
    real idx_p = static_cast<real>(has_x) + static_cast<real>(has_z) + 1.0;
    vol_y = pow(y_in, idx_p)*(pow(dy, idx_p) - 1.0) / idx_p;

    real z_in = Z_MIN + dz*static_cast<real>(idx_z);
    vol_z = has_z ? (cos(z_in) - cos(z_in + dz)) : 1.0;
}

__global__
void optdepth_calc (real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {	
        int idx_x, idx_y, idx_z;
        _grid_indices(idx, idx_x, idx_y, idx_z);

        real vol_x, vol_y, vol_z;
        _grid_volume(idx_y, idx_z, vol_x, vol_y, vol_z);

        dev_optdepth[idx] /= vol_x*vol_y*vol_z;
        
        real dy = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));
        real y_in = Y_MIN*pow(dy, static_cast<real>(idx_y));
        
        dev_optdepth[idx] *= y_in*(dy - 1.0); // prepare for radial integration
    }
}

__global__
void dustdens_calc (real *dev_dustdens)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {	
        int idx_x, idx_y, idx_z;
        _grid_indices(idx, idx_x, idx_y, idx_z);

        real vol_x, vol_y, vol_z;
        _grid_volume(idx_y, idx_z, vol_x, vol_y, vol_z);

        dev_dustdens[idx] /= vol_x*vol_y*vol_z;
    }
}

// =========================================================================================================================

__global__
void optdepth_intg (real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < NG_XZ)
    {	
        int idx_x, idx_z, idx_cell;

        idx_x = idx % N_X;
        idx_z = (idx - idx_x) / N_X;

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
