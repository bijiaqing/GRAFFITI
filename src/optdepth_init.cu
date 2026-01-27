#if defined(TRANSPORT) && defined(RADIATION)

#include "cudust_kern.cuh"

// =========================================================================================================================
// Kernel: optdepth_init
// Purpose: Initialize optical depth grid to zero
// Dependencies: None (simple initialization)
// =========================================================================================================================

__global__
void optdepth_init (real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    	
    if (idx < N_GRD)
    {
        dev_optdepth[idx] = 0.0;
    }
}

// =========================================================================================================================

#endif // RADIATION
