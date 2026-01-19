#include <cfloat>           // for DBL_MAX
#include "cudust.cuh"

// =========================================================================================================================

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

// =========================================================================================================================

__device__ __forceinline__
void _eject_to_outer (real &y, real &z, real &lx, real &vy, real &lz)
{
    y  = Y_MAX;
    z  = 0.5*M_PI;
    lx = sqrt(G*M_S*Y_MAX);
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
        _eject_to_outer (y, z, lx, vy, lz);
    }

    if (N_Z == 1)
    {
        z = 0.5*M_PI;
    }
    else if (z < Z_MIN || z >= Z_MAX) // throw it to the end of the radial domain
    {
        _eject_to_outer (y, z, lx, vy, lz);
    }
}

__device__ __forceinline__
void _get_ctfg_term (real y, real z, real R, real l_x, real l_z, real &Fc_y, real &Tc_z)
{
    Fc_y = l_x*l_x / R / R / y + l_z*l_z / y / y / y;
    Tc_z = l_x*l_x / R / R / sin(z) * cos(z);
}

// =========================================================================================================================

#ifdef RADIATION

__device__ __forceinline__
real _get_optdepth (const real *dev_optdepth, real loc_x, real loc_y, real loc_z)
{
    real optdepth = 0.0;

    bool in_x = loc_x >= 0.0 && loc_x < static_cast<real>(N_X);
    bool in_y = loc_y >= 0.0 && loc_y < static_cast<real>(N_Y);
    bool in_z = loc_z >= 0.0 && loc_z < static_cast<real>(N_Z);

    if (in_x && in_y && in_z)
    {
        int idx_cell = static_cast<int>(loc_z)*NG_XY + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);
        auto [next_x, next_y, next_z, frac_x, frac_y, frac_z] = _3d_interp_outer_y(loc_x, loc_y, loc_z);

        optdepth += dev_optdepth[idx_cell                           ]*(1.0 - frac_x)*(1.0 - frac_y)*(1.0 - frac_z);
        optdepth += dev_optdepth[idx_cell + next_x                  ]*       frac_x *(1.0 - frac_y)*(1.0 - frac_z);
        optdepth += dev_optdepth[idx_cell          + next_y         ]*(1.0 - frac_x)*       frac_y *(1.0 - frac_z);
        optdepth += dev_optdepth[idx_cell + next_x + next_y         ]*       frac_x *       frac_y *(1.0 - frac_z);
        optdepth += dev_optdepth[idx_cell                   + next_z]*(1.0 - frac_x)*(1.0 - frac_y)*       frac_z ;
        optdepth += dev_optdepth[idx_cell + next_x          + next_z]*       frac_x *(1.0 - frac_y)*       frac_z ;
        optdepth += dev_optdepth[idx_cell          + next_y + next_z]*(1.0 - frac_x)*       frac_y *       frac_z ;
        optdepth += dev_optdepth[idx_cell + next_x + next_y + next_z]*       frac_x *       frac_y *       frac_z ;
    }
    else if (loc_y < 0)
    {
        optdepth = 0;
    }
    else
    {
        optdepth = DBL_MAX;
    }

    return optdepth;
}

#endif // RADIATION

// =========================================================================================================================

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
void _ssa_substep_2 (real dt, real size, real beta, real lx_i, real vy_i, real lz_i, real x_1, real y_1, real z_1, 
    real &x_j, real &y_j, real &z_j, real &lx_j, real &vy_j, real &lz_j)
{
    real polar_R_1 = y_1*sin(z_1);
    real polar_Z_1 = y_1*cos(z_1);
    real hgas_sq = ASPR_0*ASPR_0*pow(polar_R_1, IDX_Q + 1.0);
    
    // get gas velocity in hydrostatic equilibrium for calculating the drag force
    real eta = (IDX_P + 0.5*IDX_Q - 1.5)*hgas_sq + IDX_Q*(1.0 - polar_R_1 / y_1);
    real lxg_1 = sqrt(G*M_S*polar_R_1)*sqrt(1.0 + eta);
    real vyg_1 = 0.0;
    real lzg_1 = 0.0;

    // get the stopping time of dust for calculating the drag force
    real ts_1 = ST_0 / sqrt(G*M_S/R_0/R_0/R_0);                 // the reference stopping time at R_0
    ts_1 *= size / S_0;                                         // scale with grain size
    #ifndef CONST_ST
    ts_1 *= pow(polar_R_1, -IDX_P + 1.5);                          // scale with radial   profile for gas density and sound speed
    ts_1 *= exp(-polar_Z_1*polar_Z_1 / (2.0*hgas_sq*polar_R_1*polar_R_1));    // scale with vertical profile for gas density
    #endif // CONST_ST
    real tau_1 = dt / ts_1;

    // updating the force terms to the first staggered step _1 using the position at _1 and velocity at _i
    real Fy_1 = -(1.0 - beta)*G*M_S / y_1 / y_1;
    real Fcy_1, Tcz_1;
    _get_ctfg_term(y_1, z_1, polar_R_1, lx_i, lz_i, Fcy_1, Tcz_1);

    // updating the velocity to the first staggered step _1 using force terms at _1
    real lx_1 = lx_i + (lxg_1 - lx_i)*(1.0 - exp(-0.5*tau_1));
    real vy_1 = vy_i + ((Fy_1 + Fcy_1)*ts_1 + vyg_1 - vy_i)*(1.0 - exp(-0.5*tau_1));
    real lz_1 = lz_i + (Tcz_1*ts_1 + lzg_1 - lz_i)*(1.0 - exp(-0.5*tau_1));

    // updating the force terms to the second staggered step _2 using both the position and the velocity at _1
    // this takes into account that beta is independent of velocity, so we can reuse it here
    // essentially, beta_2 = beta_1 and is unified as beta here
    real Fy_2 = -(1.0 - beta)*G*M_S / y_1 / y_1;
    real Fcy_2, Tcz_2;
    _get_ctfg_term(y_1, z_1, polar_R_1, lx_1, lz_1, Fcy_2, Tcz_2);

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

#ifdef RADIATION // ssa needs to be interrupted for optical depth calculation

__global__
void ssa_substep_1 (swarm *dev_particle, real dt)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        real x_i, y_i, z_i;
        real x_1, y_1, z_1;
        
        real lx_i, vy_i, lz_i;

        _load_particle(dev_particle, idx, x_i, y_i, z_i, lx_i, vy_i, lz_i);
        _ssa_substep_1(dt, x_i, y_i, z_i, lx_i, vy_i, lz_i, x_1, y_1, z_1);
        _if_out_of_box(x_1, y_1, z_1, lx_i, vy_i, lz_i);
        _save_particle(dev_particle, idx, x_1, y_1, z_1, lx_i, vy_i, lz_i);
    }
}

__global__
void ssa_substep_2 (swarm *dev_particle, const real *dev_optdepth, real dt)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        real x_1, y_1, z_1;
        real x_j, y_j, z_j;
        
        real lx_i, vy_i, lz_i;
        real lx_j, vy_j, lz_j;
        
        _load_particle(dev_particle, idx, x_1, y_1, z_1, lx_i, vy_i, lz_i);

        // retrieve the optical depth and calculate beta
        real loc_x = _get_loc_x(x_1);
        real loc_y = _get_loc_y(y_1);
        real loc_z = _get_loc_z(z_1);

        real optdepth = _get_optdepth(dev_optdepth, loc_x, loc_y, loc_z);
        real size = dev_particle[idx].par_size;
        real beta = BETA_0*exp(-optdepth) / (size / S_0);

        _ssa_substep_2(dt, size, beta, lx_i, vy_i, lz_i, x_1, y_1, z_1, x_j, y_j, z_j, lx_j, vy_j, lz_j);
        _if_out_of_box(x_j, y_j, z_j, lx_j, vy_j, lz_j);
        _save_particle(dev_particle, idx, x_j, y_j, z_j, lx_j, vy_j, lz_j);
    }
}

#else // ssa can be done in one kernel call

__global__
void ssa_integrate (swarm *dev_particle, real dt)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        real x_i, y_i, z_i;
        real x_1, y_1, z_1;
        real x_j, y_j, z_j;
        
        real lx_i, vy_i, lz_i;
        real lx_j, vy_j, lz_j;

        _load_particle(dev_particle, idx, x_i, y_i, z_i, lx_i, vy_i, lz_i);
        _ssa_substep_1(dt, x_i, y_i, z_i, lx_i, vy_i, lz_i, x_1, y_1, z_1);

        real size = dev_particle[idx].par_size;
        real beta = 0.0; // no radiation pressure

        _ssa_substep_2(dt, size, beta, lx_i, vy_i, lz_i, x_1, y_1, z_1, x_j, y_j, z_j, lx_j, vy_j, lz_j);
        _if_out_of_box(x_j, y_j, z_j, lx_j, vy_j, lz_j);
        _save_particle(dev_particle, idx, x_j, y_j, z_j, lx_j, vy_j, lz_j);
    }
}

#endif // RADIATION