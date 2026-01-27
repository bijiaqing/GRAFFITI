#if defined(TRANSPORT) && defined(RADIATION)

#include "cudust_kern.cuh"

// =========================================================================================================================
// Kernel: optdepth_csum
// Purpose: Compute cumulative sum of optical depth in the radial direction
// Dependencies: None
// =========================================================================================================================

__global__
void optdepth_csum (real *dev_optdepth) // cumulative sum in the radial direction
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < NG_XZ)
    {	
        int idx_x, idx_z, idx_cell;

        idx_x = idx % N_X;
        idx_z = (idx - idx_x) / N_X;

        // cumulative sum in Y direction
        // no race condition since each thread works on a unique Y row
        for (int i = 1; i < N_Y; i++)
        {
            idx_cell = idx_z*NG_XY + i*N_X + idx_x;
            dev_optdepth[idx_cell] += dev_optdepth[idx_cell - N_X];
        }
    }
}

// =========================================================================================================================

#endif // RADIATION
