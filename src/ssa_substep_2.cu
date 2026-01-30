#if defined(TRANSPORT) && defined(RADIATION)

#include <cfloat>                // for DBL_MAX

#include "cudust_kern.cuh"
#include "helpers_transport.cuh"
#include "helpers_gridfield.cuh" // for _3d_interp_outer_y

// =========================================================================================================================
// Kernel: ssa_substep_2
// Purpose: Second substep of the staggered semi-analytic integrator for particles
// Dependencies: helpers_transport.cuh (_load_particle, _save_particle, _if_out_of_box, _get_ctfg_term, _ssa_substep_2),
//               helpers_gridfield.cuh (_3d_interp_outer_y), private _get_optdepth
// =========================================================================================================================

__device__ __forceinline__
real _get_optdepth (const real *dev_optdepth, real loc_x, real loc_y, real loc_z)
{
    if (loc_y < 0) return 0.0;
    if (!_is_in_bounds(loc_x, loc_y, loc_z)) return DBL_MAX; // particle is out of bounds, return a large optical depth

    int idx_cell = _get_cell_index(loc_x, loc_y, loc_z);
    auto [next_x, next_y, next_z, frac_x, frac_y, frac_z] = _3d_interp_outer_y(loc_x, loc_y, loc_z);

    real optdepth = 0.0;

    optdepth += dev_optdepth[idx_cell                           ]*(1.0 - frac_x)*(1.0 - frac_y)*(1.0 - frac_z);
    optdepth += dev_optdepth[idx_cell + next_x                  ]*       frac_x *(1.0 - frac_y)*(1.0 - frac_z);
    optdepth += dev_optdepth[idx_cell          + next_y         ]*(1.0 - frac_x)*       frac_y *(1.0 - frac_z);
    optdepth += dev_optdepth[idx_cell + next_x + next_y         ]*       frac_x *       frac_y *(1.0 - frac_z);
    optdepth += dev_optdepth[idx_cell                   + next_z]*(1.0 - frac_x)*(1.0 - frac_y)*       frac_z ;
    optdepth += dev_optdepth[idx_cell + next_x          + next_z]*       frac_x *(1.0 - frac_y)*       frac_z ;
    optdepth += dev_optdepth[idx_cell          + next_y + next_z]*(1.0 - frac_x)*       frac_y *       frac_z ;
    optdepth += dev_optdepth[idx_cell + next_x + next_y + next_z]*       frac_x *       frac_y *       frac_z ;

    return optdepth;
}

__global__
void ssa_substep_2 (swarm *dev_particle, const real *dev_optdepth, real dt
    #ifdef IMPORTGAS
    , const real *dev_gasvelx, const real *dev_gasvely, const real *dev_gasvelz, const real *dev_gasdens
    #endif
)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
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

        _ssa_substep_2(dt, size, beta, lx_i, vy_i, lz_i, x_1, y_1, z_1, x_j, y_j, z_j, lx_j, vy_j, lz_j
            #ifdef IMPORTGAS
            , dev_gasvelx, dev_gasvely, dev_gasvelz, dev_gasdens
            #endif
        );
        _if_out_of_box(x_j, y_j, z_j, lx_j, vy_j, lz_j);
        _save_particle(dev_particle, idx, x_j, y_j, z_j, lx_j, vy_j, lz_j);
    }
}

// ========================================================================================================================

#endif // RADIATION
