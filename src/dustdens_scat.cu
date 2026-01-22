

#include "cudust_kern.cuh"
#include "helpers_gridfield.cuh"

// =========================================================================================================================
// Kernel: dustdens_scat
// Purpose: Scatter particle dust densities to grid using trilinear interpolation
// Dependencies: helpers_gridfield.cuh (provides _particle_to_grid_core template)
// =========================================================================================================================

__global__
void dustdens_scat (real *dev_dustdens, const swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        _particle_to_grid_core <DUSTDENS> (dev_dustdens, dev_particle, idx);
    }
}
