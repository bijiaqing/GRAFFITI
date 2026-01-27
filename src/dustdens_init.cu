#ifdef SAVE_DENS

#include "cudust_kern.cuh"

// =========================================================================================================================
// Kernel: dustdens_init
// Purpose: Initialize dust density grid to zero
// Dependencies: None (simple initialization)
// =========================================================================================================================

__global__
void dustdens_init (real *dev_dustdens)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    	
    if (idx < N_GRD)
    {
        dev_dustdens[idx] = 0.0;
    }
}

// =========================================================================================================================

#endif // SAVE_DENS
