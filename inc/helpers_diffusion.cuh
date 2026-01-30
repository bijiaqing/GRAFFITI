#ifndef HELPERS_DIFFUSION_CUH
#define HELPERS_DIFFUSION_CUH

#if defined(TRANSPORT) && defined(DIFFUSION)

#include "const.cuh"

#ifdef IMPORTGAS
#include "helpers_gridfield.cuh"  // for _interp_field
#else  // NO IMPORTGAS
#include "helpers_diskparam.cuh"  // for _get_hg
#endif // IMPORTGAS

// =========================================================================================================================
// Function: _get_term_grad_cyl
// Purpose: Calculate the gradient terms in cylindrical coordinates needed for diffusion calculation
// Dependencies: helpers_diskparam.cuh (for _get_hg),
//               helpers_gridfield.cuh (for _interp_field) [if IMPORTGAS defined]
// =========================================================================================================================

__device__ __forceinline__
void _get_term_grad_cyl (real x, real y, real z, real &term_x, real &term_R, real &term_Z
    #ifdef IMPORTGAS
    , const real *dev_gasdens
    #endif // IMPORTGAS
)
{   
    #ifdef IMPORTGAS

    real rho = _interp_field(dev_gasdens, x, y, z); // returns 0.0 if out of bounds

    if (rho <= 0.0) // out of bounds or zero density
    {
        term_x = 0.0;
        term_R = 0.0;
        term_Z = 0.0;
        
        return;
    }

    real drho_dx, drho_dy, drho_dz;

    if (N_X > 1 && rho > 0.0)
    {
        real dx = (X_MAX - X_MIN) / static_cast<real>(N_X);
        
        real x_p = x + dx;
        real x_m = x - dx;

        if (x_p >= X_MAX) x_p -= (X_MAX - X_MIN);
        if (x_m <  X_MIN) x_m += (X_MAX - X_MIN);
        
        real rho_xp = _interp_field(dev_gasdens, x_p, y, z);
        real rho_xm = _interp_field(dev_gasdens, x_m, y, z);
        
        drho_dx = (rho_xp - rho_xm) / (2.0*dx);
    }
    else // if azimuthal direction disabled
    {
        drho_dx = 0.0;
    }
    
    if (N_Y > 1)
    {
        real loc_y = _get_loc_y(y);
        real dy = y*(log(Y_MAX / Y_MIN) / static_cast<real>(N_Y));
        
        if (loc_y < 0.5) // near inner boundary, use forward difference
        {
            real rho_yp = _interp_field(dev_gasdens, x, y + dy, z);
            
            drho_dy = (rho_yp - rho) / dy;
        }
        else if (loc_y > static_cast<real>(N_Y) - 0.5) // near outer boundary, use backward difference
        {
            real rho_ym = _interp_field(dev_gasdens, x, y - dy, z);
            
            drho_dy = (rho - rho_ym) / dy;
        }
        else  // interior, central difference
        {
            real rho_yp = _interp_field(dev_gasdens, x, y + dy, z);
            real rho_ym = _interp_field(dev_gasdens, x, y - dy, z);
            
            drho_dy = (rho_yp - rho_ym) / (2.0*dy);
        }
    }
    else // if radial direction disabled, not gonna happen
    {
        drho_dy = 0.0;
    }

    if (N_Z > 1)
    {
        real loc_z = _get_loc_z(z);
        real dz = (Z_MAX - Z_MIN) / static_cast<real>(N_Z);
        
        if (loc_z < 0.5) // near lower boundary, use forward difference
        {
            real rho_zp = _interp_field(dev_gasdens, x, y, z + dz);
            
            drho_dz = (rho_zp - rho) / dz;
        }
        else if (loc_z > static_cast<real>(N_Z) - 0.5) // near upper boundary, use backward difference
        {
            real rho_zm = _interp_field(dev_gasdens, x, y, z - dz);
            
            drho_dz = (rho - rho_zm) / dz;
        }
        else  // interior: central difference
        {
            real rho_zp = _interp_field(dev_gasdens, x, y, z + dz);
            real rho_zm = _interp_field(dev_gasdens, x, y, z - dz);
            
            drho_dz = (rho_zp - rho_zm) / (2.0*dz);
        }
    }
    else // if vertical direction disabled
    {
        drho_dz = 0.0;
    }
    
    real sin_z = sin(z);
    real cos_z = cos(z);
    
    // convert spherical to cylindrical derivatives
    real drho_dR = drho_dy*sin_z + drho_dz*cos_z / y;
    real drho_dZ = drho_dy*cos_z - drho_dz*sin_z / y;

    term_x = drho_dx / rho;
    term_R = drho_dR / rho;
    term_Z = drho_dZ / rho;

    #else  // NO IMPORTGAS (using analytical prescriptions)
    
    // assuming azimuthal axisymmetry
    term_x = 0.0;

    real R = y*sin(z);
    real Z = y*cos(z);
    real h_g = _get_hg(R);

    // assuming sigma_g ~ pow(R, IDX_P), then
    // (1) rho_g ~ sigma_g / H_gas ~ pow(R, IDX_P - 0.5*IDX_Q - 1.5)
    // (2) d(rhog)/dR / rho = (IDX_P - 0.5*IDX_Q - 1.5) / R
    term_R = (IDX_P - 0.5*IDX_Q - 1.5) / R;

    // assuming rho_g ~ exp(-Z^2 / (2*H_gas^2)), then
    // (1) d(rhog)/dZ/rhog = -Z / H_gas^2
    term_Z = -Z / (h_g*h_g*R*R);
    
    #endif // IMPORTGAS
}

#endif // TRANSPORT && DIFFUSION

#endif // HELPERS_DIFFUSION_CUH
