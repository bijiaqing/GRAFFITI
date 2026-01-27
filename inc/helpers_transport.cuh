#ifndef HELPERS_TRANSPORT_CUH
#define HELPERS_TRANSPORT_CUH

#include "const.cuh"
#include "helpers_diskparam.cuh"

__device__ __forceinline__
void _load_particle (const swarm *dev_particle, int idx, real &x, real &y, real &z, real &lx, real &vy, real &lz)
{
    x  = dev_particle[idx].position.x;
    y  = dev_particle[idx].position.y;
    z  = dev_particle[idx].position.z;
    lx = dev_particle[idx].velocity.x;
    vy = dev_particle[idx].velocity.y;
    lz = dev_particle[idx].velocity.z;
}

__device__ __forceinline__
void _save_particle (swarm *dev_particle, int idx, real x, real y, real z, real lx, real vy, real lz)
{
    dev_particle[idx].position.x =  x;
    dev_particle[idx].position.y =  y;
    dev_particle[idx].position.z =  z;
    dev_particle[idx].velocity.x = lx;
    dev_particle[idx].velocity.y = vy;
    dev_particle[idx].velocity.z = lz;
}

__device__ __forceinline__
void _eject_to_outer (real &y, real &z, real &lx, real &vy, real &lz)
{
    y  = Y_MAX;
    z  = 0.5*M_PI;
    lx = _get_omegaK(y)*y*y;
    vy = 0.0;
    lz = 0.0;
}

__device__ __forceinline__
void _if_out_of_box (real &x, real &y, real &z, real &lx, real &vy, real &lz)
{
    if (N_X == 1) 
    {
        x = 0.5*(X_MIN + X_MAX);
    }
    else // keep x within [X_MIN, X_MAX)
    {
        while (x >= X_MAX) x -= X_MAX - X_MIN;
        while (x <  X_MIN) x += X_MAX - X_MIN;
    }

    if (y < Y_MIN) // throw it to the end of the radial domain
    {
        _eject_to_outer(y, z, lx, vy, lz);
    }

    if (N_Z == 1)
    {
        z = 0.5*M_PI;
    }
    else if (z < Z_MIN || z >= Z_MAX) // throw it to the end of the radial domain
    {
        _eject_to_outer(y, z, lx, vy, lz);
    }
}

__device__ __forceinline__
void _ssa_substep_1 (real dt, real x_i, real y_i, real z_i, real lx_i, real vy_i, real lz_i, 
    real &x_1, real &y_1, real &z_1)
{
    // updating the position to the first staggered step _1 using the velocity at the starting time step _i
    y_1 = y_i + 0.5*vy_i*dt;
    z_1 = z_i + 0.5*lz_i*dt / y_i / y_1;
    x_1 = x_i + 0.5*lx_i*dt / y_i / y_1 / sin(z_i) / sin(z_1);
}

__device__ __forceinline__
void _get_force_term (real y, real z, real R, real l_x, real l_z, real beta, real &F_y, real &Fc_y, real &Tc_z)
{
    F_y  = -(1.0 - beta)*_get_omegaK(y)*_get_omegaK(y)*y;
    Fc_y = l_x*l_x / R / R / y + l_z*l_z / y / y / y;
    Tc_z = l_x*l_x / R / R / sin(z) * cos(z);
}

__device__ __forceinline__
void _ssa_substep_2 (real dt, real size, real beta, real lx_i, real vy_i, real lz_i, real x_1, real y_1, real z_1, 
    real &x_j, real &y_j, real &z_j, real &lx_j, real &vy_j, real &lz_j)
{
    real polar_R_1 = y_1*sin(z_1);
    real polar_Z_1 = y_1*cos(z_1);
    
    real h_gas = _get_hgas(polar_R_1);
    real omega = _get_omegaK(polar_R_1);
    real eta = _get_eta(polar_R_1, polar_Z_1, h_gas);
    
    // get gas velocity in hydrostatic equilibrium for calculating the drag force
    
    real lxg_1 = sqrt(1.0 - 2.0*eta)*omega*polar_R_1*polar_R_1;
    real vyg_1 = 0.0;
    real lzg_1 = 0.0;

    // get the stopping time of dust for calculating the drag force
    real ts_1 = _get_St(polar_R_1, polar_Z_1, size, h_gas) / omega;
    real tau_1 = dt / ts_1;

    // updating the force terms to the first staggered step _1 using the position at _1 and velocity at _i
    real Fy_1, Fcy_1, Tcz_1;
    _get_force_term(y_1, z_1, polar_R_1, lx_i, lz_i, beta, Fy_1, Fcy_1, Tcz_1);

    // updating the velocity to the first staggered step _1 using force terms at _1
    real lx_1 = lx_i + (lxg_1 - lx_i)*(1.0 - exp(-0.5*tau_1));
    real vy_1 = vy_i + ((Fy_1 + Fcy_1)*ts_1 + vyg_1 - vy_i)*(1.0 - exp(-0.5*tau_1));
    real lz_1 = lz_i + (Tcz_1*ts_1 + lzg_1 - lz_i)*(1.0 - exp(-0.5*tau_1));

    // updating the force terms to the second staggered step _2 using both the position and the velocity at _1
    // this takes into account that beta is independent of velocity, so we can reuse it here
    // essentially, beta_2 = beta_1 and is unified as beta here
    real Fy_2, Fcy_2, Tcz_2;
    _get_force_term(y_1, z_1, polar_R_1, lx_1, lz_1, beta, Fy_2, Fcy_2, Tcz_2);

    // updating the velocity to the next time step _j
    lx_j = lx_i + (lxg_1 - lx_i)*(1.0 - exp(-tau_1));
    vy_j = vy_i + ((Fy_2 + Fcy_2)*ts_1 + vyg_1 - vy_i)*(1.0 - exp(-tau_1));
    lz_j = lz_i + (Tcz_2*ts_1 + lzg_1 - lz_i)*(1.0 - exp(-tau_1));

    // Updating the position to the next time step _j
    y_j = y_1 + 0.5*vy_j*dt;
    z_j = z_1 + 0.5*lz_j*dt / y_1 / y_j;
    x_j = x_1 + 0.5*lx_j*dt / y_1 / y_j / sin(z_1) / sin(z_j);
}

// =========================================================================================================================

#endif // NOT HELPERS_TRANSPORT_CUH
