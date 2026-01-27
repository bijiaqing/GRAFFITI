#ifdef SAVE_DENS

#include "cudust_kern.cuh"
#include "helpers_diskparam.cuh"

// =========================================================================================================================
// Kernel: dustdens_calc
// Purpose: Normalize dust density by grid cell volume
// Dependencies: helpers_diskparam.cuh (_get_grid_volume)
// =========================================================================================================================

__global__
void dustdens_calc (real *dev_dustdens)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {	
        real volume = _get_grid_volume(idx);
        dev_dustdens[idx] /= volume;
    }
}

// =========================================================================================================================

#endif // SAVE_DENS
