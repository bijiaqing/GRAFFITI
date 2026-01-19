#include "cudust.cuh"

// =========================================================================================================================
// Shared device helper functions used across multiple source files

__device__
real _get_dust_mass (real size)
{
    return M_PI*RHO_0*size*size*size / 6.0;
}

__device__
real _get_sigma_gas (real R)
{
    return SIGMA_0*pow(R / R_0, -IDX_P);
}

__device__
real _get_omegaK (real R)
{
    return sqrt(G*M_S / R / R / R);
}

__device__
real _get_hgas (real R)
{
    return ASPR_0*pow(R / R_0, 0.5*(IDX_Q + 1.0));
}

__device__
real _get_cs (real R, real h_gas)
{
    return h_gas*_get_omegaK(R)*R;
}

__device__
real _get_eta (real R, real Z, real h_gas)
{
    return -0.5*((IDX_P + 0.5*IDX_Q - 1.5)*h_gas*h_gas + IDX_Q*(1.0 - R / sqrt(R*R + Z*Z)));
}

__device__
real _get_St (real R, real Z, real size, real h_gas)
{
    real St = ST_0;
    
    St *= size / S_0;                           // scale with grain size
    #ifndef CONST_ST
    St *= pow(R / R_0, -IDX_P);                 // scale with radial   profile for gas density and sound speed
    St /= exp(-Z*Z / (2.0*h_gas*h_gas*R*R));    // scale with vertical profile for gas density
    #endif // CONST_ST

    return St;
}

__device__
real _get_alpha (real R, real h_gas)
{
    real alpha;
    
    #ifndef CONST_NU
    alpha = ALPHA;
    #else
    alpha = NU / (h_gas*h_gas*R*R*_get_omegaK(R));
    #endif // CONST_NU

    return alpha;
}

__device__
real _get_delta_R (real R, real h_gas)
{
    real nu;
    
    #ifndef CONST_NU
    nu = ALPHA*h_gas*h_gas*R*R*_get_omegaK(R);  // kinematic viscosity
    #else
    nu = NU;
    #endif // CONST_NU
    
    return nu / SC_R;
}

__device__
real _get_delta_Z (real R, real h_gas)
{
    real nu;
    
    #ifndef CONST_NU
    nu = ALPHA*h_gas*h_gas*R*R*_get_omegaK(R);  // kinematic viscosity
    #else
    nu = NU;
    #endif // CONST_NU
    
    return nu / SC_Z;
}

__device__
real _get_hdust (real R, real St)
{
    // h_gas cannot be passed by a parameter here because _get_hdust is for individual particles
    // whereas h_gas is, in most cases, evaluated at the midpoint between particles
    
    real h_gas = _get_hgas(R);
    real delta_Z = _get_delta_Z(R, h_gas);
    
    return h_gas*sqrt(delta_Z / (delta_Z + St));
}

// =========================================================================================================================
