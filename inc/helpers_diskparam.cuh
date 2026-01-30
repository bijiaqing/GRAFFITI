#ifndef HELPERS_DISKPARAM_CUH
#define HELPERS_DISKPARAM_CUH

#include "const.cuh"

#ifdef IMPORTGAS
#include "helpers_gridfield.cuh"  // for _interp_field
#endif

// =========================================================================================================================
// Shared Helper: Calculate grid cell indices
// Used by: col_proc_exec, col_rate_calc, ssa_substep_2, _particle_to_grid_core
// =========================================================================================================================

__device__ __forceinline__
real _get_loc_x (real x)
{
    return (N_X > 1) ? (static_cast<real>(N_X)*   (x - X_MIN) /    (X_MAX - X_MIN)) : 0.0; 
}

__device__ __forceinline__
real _get_loc_y (real y)
{
    return (N_Y > 1) ? (static_cast<real>(N_Y)*log(y / Y_MIN) / log(Y_MAX / Y_MIN)) : 0.0; 
}

__device__ __forceinline__
real _get_loc_z (real z)
{
    return (N_Z > 1) ? (static_cast<real>(N_Z)*   (z - Z_MIN) /    (Z_MAX - Z_MIN)) : 0.0; 
}

// =========================================================================================================================
// Shared Helper: Disk parameter calculations
// Used by: ssa_substep_2, diffusion_pos, diffusion_vel, col_rate_calc, col_proc_exec
// =========================================================================================================================

__device__ __forceinline__
real _get_grain_mass (real s)
{
    return M_PI*RHO_0*s*s*s / 6.0;
}

__device__ __forceinline__
real _get_omegaK (real R)
{
    return sqrt(G*M_S / R / R / R);
}

__device__ __forceinline__
real _get_hg (real R)
{
    return ASPR_0*pow(R / R_0, 0.5*(IDX_Q + 1.0));
}

__device__ __forceinline__
real _get_cs (real R, real h_g)
{
    return h_g*_get_omegaK(R)*R;
}

__device__ __forceinline__
real _get_eta (real R, real Z, real h_g)
{
    return -0.5*((IDX_P + 0.5*IDX_Q - 1.5)*h_g*h_g + IDX_Q*(1.0 - R / sqrt(R*R + Z*Z)));
}

#if defined(DIFFUSION) || defined(COLLISION)
__device__ __forceinline__
real _get_nu (real R, real h_g)
{
    #ifndef CONST_NU
    return ALPHA*h_g*h_g*R*R*_get_omegaK(R);
    #else  // CONST_NU
    return NU;
    #endif // NOT CONST_NU
}
#endif // DIFFUSION || COLLISION

#ifdef COLLISION
__device__ __forceinline__
real _get_sigma_g (real R)
{
    return SIGMA_0*pow(R / R_0, IDX_P);
}

__device__ __forceinline__
real _get_alpha (real R, real h_g)
{
    #ifndef CONST_NU
    return ALPHA;
    #else  // CONST_NU
    return NU / (h_g*h_g*R*R*_get_omegaK(R));
    #endif // NOT CONST_NU
}

__device__ __forceinline__
real _get_hd (real R, real St)
{
    // h_g cannot be passed by a parameter here because _get_hd is for individual particles
    // whereas h_g is, in most cases, evaluated at the midpoint between particles
    
    real h_g = _get_hg(R);
    real delta_Z = _get_nu(R, h_g) / SC_Z;
    
    return h_g*sqrt(delta_Z / (delta_Z + St));
}
#endif // COLLISION

__device__ __forceinline__
real _get_St (real R, real Z, real s, real h_g
    #ifdef IMPORTGAS
    , real x, real y, real z, const real *dev_gasdens
    #endif
)
{
    real St = ST_0*(s / S_0);

    #ifndef CONST_ST
    #ifdef IMPORTGAS
    if (dev_gasdens != nullptr)
    {
        // Use imported gas density
        real loc_x = _get_loc_x(x);
        real loc_y = _get_loc_y(y);
        real loc_z = _get_loc_z(z);
        
        real rhog  = _interp_field(dev_gasdens, loc_x, loc_y, loc_z);
        real rhog0 = SIGMA_0 / (sqrt(2.0*M_PI)*h_g*R);
        
        if (rhog > 0.0) 
        {
            St *= rhog0 / rhog;
        }
    }
    else
    #endif // IMPORTGAS
    {
        // Analytical gas density profile
        St /= pow(R / R_0, IDX_P);              // scale with radial   profile for volumetric gas density and sound speed
        St /= exp(-Z*Z / (2.0*h_g*h_g*R*R));    // scale with vertical profile for volumetric gas density
    }
    #endif // NOT CONST_ST

    return St;
}

// =========================================================================================================================
// Shared Helper: Calculate grid cell volume
// Used by: optdepth_calc, dustdens_calc, col_proc_exec, col_rate_calc
// =========================================================================================================================

__device__ __forceinline__
real _get_grid_volume (int idx_cell, real *y0_ptr = nullptr, real *dy_ptr = nullptr)
{
    // Convert 1D cell index to 3D grid indices
    int idx_x = idx_cell % N_X;
    int idx_y = (idx_cell / N_X) % N_Y;
    int idx_z = idx_cell / (N_X*N_Y);

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

// =========================================================================================================================

#endif // NOT HELPERS_DISKPARAM_CUH
