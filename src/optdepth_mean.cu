#if defined(TRANSPORT) && defined(RADIATION)

#include "cudust_kern.cuh"

// =========================================================================================================================
// Kernel: optdepth_mean
// Purpose: Compute azimuthal average of optical depth (average in X direction)
// Dependencies: None
// =========================================================================================================================

__global__
void optdepth_mean (real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < NG_YZ)
    {	
        int idx_y, idx_z, idx_cell;

        idx_y = idx % N_Y;
        idx_z = (idx - idx_y) / N_Y;

        real optdepth_sum = 0.0;

        // summation in X direction
        // no race condition since each thread works on a unique X row
        for (int i = 0; i < N_X; i++)
        {
            idx_cell = idx_z*NG_XY + idx_y*N_X + i;
            optdepth_sum += dev_optdepth[idx_cell];
        }

        real optdepth_avg = optdepth_sum / N_X;

        for (int j = 0; j < N_X; j++)
        {
            idx_cell = idx_z*NG_XY + idx_y*N_X + j;
            dev_optdepth[idx_cell] = optdepth_avg;
        }
    }
}

// =========================================================================================================================

#endif // RADIATION
