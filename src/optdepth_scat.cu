#ifdef RADIATION

#include "cudust_kern.cuh"
#include "helpers_gridfield.cuh"

// =========================================================================================================================
// Kernel: optdepth_scat
// Purpose: Scatter particle optical depths to grid using trilinear interpolation
// Dependencies: helpers_gridfield.cuh (provides _particle_to_grid_core template)
// =========================================================================================================================

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
