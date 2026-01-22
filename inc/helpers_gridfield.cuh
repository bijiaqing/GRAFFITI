#ifndef HELPERS_GRIDFIELD_CUH
#define HELPERS_GRIDFIELD_CUH

#include "const.cuh"
#include "helpers_diskparam.cuh" // for _get_loc_x/y/z, _get_dust_mass, _get_grid_volume

// =========================================================================================================================
// Grid field type definitions
// =========================================================================================================================

struct interp                               // interpolation result for trilinear interpolation
{
    int  next_x, next_y, next_z;            // indices of the next grid cell in each direction
    real frac_x, frac_y, frac_z;            // fractional distance to the next grid cell in each direction
};

enum GridFieldType                          // field type for particle-to-grid scattering
{
    DUSTDENS,                               // dust density field
    #ifdef RADIATION
    OPTDEPTH,                               // optical depth field
    #endif // RADIATION
};

// =========================================================================================================================
// Trilinear interpolation helpers for grid field operations
// =========================================================================================================================

__device__ __forceinline__
void _1d_interp_midpt_x (real loc_x, real deci_x, real &frac_x, int &next_x)
{
    if (N_X == 1)                   // if there is only one cell in X
    {
        frac_x = 0.0;               // the share for the current cell is '1.0 - frac_x'
        next_x = 0;                 // no other cells to share the particle
    }
    else
    {
        bool edge_x = loc_x < 0.5 || loc_x > static_cast<real>(N_X) - 0.5;

        if (not edge_x)             // still in the interior of the X domain
        {
            if (deci_x >= 0.5)      // share with the cell on the right
            {
                frac_x = deci_x - 0.5;
                next_x = 1;
            }
            else                    // share with the cell on the left
            {
                frac_x = 0.5 - deci_x;
                next_x = -1;
            }
        }
        else                        // too close to the inner or the outer X boundary 
        {
            if (deci_x >= 0.5)      // too close to the outer X boundary
            {
                frac_x = deci_x - 0.5;
                next_x = 1 - N_X;   // share with the first cell of its row
            }
            else                    // too close to the inner X boundary
            {
                frac_x = 0.5 - deci_x;
                next_x = N_X - 1;   // share with the last  cell of its row
            }
        }
    }
}

__device__ __forceinline__
void _1d_interp_midpt_z (real loc_z, real deci_z, real &frac_z, int &next_z)
{
    if (N_Z == 1)                   // if there is only one cell in Z
    {
        frac_z = 0.0;               // the share for the current cell is '1.0 - frac_z'
        next_z = 0;                 // no other cells to share the particle
    }
    else
    {
        bool edge_z = loc_z < 0.5 || loc_z > static_cast<real>(N_Z) - 0.5;
        
        if (not edge_z)             // still in the interior of the Z domain
        {
            if (deci_z >= 0.5)
            {
                frac_z = deci_z - 0.5;
                next_z = NG_XY;     // the index distance to the next Z cell on the right is N_X*N_Y = NG_XY
            }
            else
            {
                frac_z = 0.5 - deci_z;
                next_z = -NG_XY;
            }
        }
        else                        // at the Z domain boundaries, the current cell take it all like N_Z = 1
        {
            frac_z = 0.0;
            next_z = 0;
        }
    }
}

__device__ __forceinline__
void _1d_interp_midpt_y (real loc_y, real deci_y, real &frac_y, int &next_y)
{
    if (N_Y == 1)                   // if there is only one cell in Y
    {
        frac_y = 0.0;               // the share for the current cell is '1.0 - frac_y'
        next_y = 0;                 // no other cells to share the particle
    }
    else
    {
        real d_y = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y));    // exponential base for logarithmic spacing
        real m_y = log(0.5*(1.0 + d_y)) / log(d_y);                     // d_y^m_y is the geometric center of the Y cell
        
        bool edge_y = loc_y < m_y || loc_y > static_cast<real>(N_Y) - m_y;

        if (not edge_y)             // still in the interior of the Y domain
        {
            if (deci_y >= m_y)      // share with the cell on the right
            {
                frac_y = (pow(d_y, deci_y - m_y) - 1.0) / (d_y - 1.0);
                next_y = N_X;       // the index distance to the next Y cell on the right is N_X
            }
            else                    // share with the cell on the left
            {
                frac_y = (pow(d_y, deci_y - m_y) - 1.0) / (1.0 / d_y - 1.0);
                next_y = -N_X;
            }
        }
        else                        // at the Y domain boundaries, the current cell take it all like N_Y = 1
        {
            frac_y = 0.0;
            next_y = 0;
        }
    }
}

__device__ __forceinline__
void _1d_interp_outer_y (real loc_y, real deci_y, real &frac_y, int &next_y)
{
    // values are interpolated to the outer edge of each radial cell
    
    if (N_Y == 1)                   // if there is only one cell in Y
    {
        frac_y = 0.0;               // the share for the current cell is '1.0 - frac_y'
        next_y = 0;                 // no other cells to share the particle
    }
    else
    {
        real d_y = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y)); // exponential base for logarithmic spacing
        real m_y = 1.0;
        
        bool edge_y = loc_y < m_y || loc_y > static_cast<real>(N_Y) + m_y - 1.0;

        if (not edge_y)
        {
            frac_y = (d_y - pow(d_y, deci_y)) / (d_y - 1.0);
            next_y = -N_X;          // share with the cell on its left
        }
        else                        // if the particle is in the innermost radial layer
        {
            frac_y = (d_y - pow(d_y, deci_y)) / (d_y - 1.0);
            next_y = 0;             // the particles are unfortunately 100% self-shadowed
        }
    }
}

__device__ __forceinline__
interp _3d_interp_midpt_y (real loc_x, real loc_y, real loc_z)
{
    // interpolating to the geometric center of each cell in all three directions
    // x and z are uniform grids, y is logarithmic grid

    real frac_x, frac_y, frac_z;
    int  next_x, next_y, next_z;

    real deci_x = loc_x - floor(loc_x);
    real deci_y = loc_y - floor(loc_y);
    real deci_z = loc_z - floor(loc_z);

    _1d_interp_midpt_x(loc_x, deci_x, frac_x, next_x);
    _1d_interp_midpt_y(loc_y, deci_y, frac_y, next_y);
    _1d_interp_midpt_z(loc_z, deci_z, frac_z, next_z);

    return {next_x, next_y, next_z, frac_x, frac_y, frac_z};
}

__device__ __forceinline__
interp _3d_interp_outer_y (real loc_x, real loc_y, real loc_z)
{
    // this function exists because the optical depth field is defined at the outer radial boundary of each cell
    // the particle needs to be interpolated based on the location of the radial cell edges to get the shadow

    real frac_x, frac_y, frac_z;
    int  next_x, next_y, next_z;

    real deci_x = loc_x - floor(loc_x);
    real deci_y = loc_y - floor(loc_y);
    real deci_z = loc_z - floor(loc_z);

    _1d_interp_midpt_x(loc_x, deci_x, frac_x, next_x);
    _1d_interp_outer_y(loc_y, deci_y, frac_y, next_y);
    _1d_interp_midpt_z(loc_z, deci_z, frac_z, next_z);

    return {next_x, next_y, next_z, frac_x, frac_y, frac_z};
}

// =========================================================================================================================
// Grid field scattering operations
// =========================================================================================================================

template <GridFieldType field_type> __device__ __forceinline__
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
    {
        if (field_type == OPTDEPTH)
        {
            weight  = _get_dust_mass(size);
            weight *= dev_particle[idx].par_numr;
            weight *= KAPPA_0*(S_0 / size); // cross section per unit mass
        }
        else
    }
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

#endif // HELPERS_GRIDFIELD_CUH
