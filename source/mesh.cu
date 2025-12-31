#include <cfloat>   // for DBL_MAX

#include "cudust.cuh"

// =========================================================================================================================

__global__
void optdepth_init (real *dev_optdepth)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    	
    if (idx >= 0 && idx < NUM_DIM)
    {
        dev_optdepth[idx] = 0.0;
    }
}

__global__
void dustdens_init (real *dev_dustdens)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    	
    if (idx >= 0 && idx < NUM_DIM)
    {
        dev_dustdens[idx] = 0.0;
    }
}

// =========================================================================================================================

__global__
void optdepth_enum (real *dev_optdepth, swarm *dev_particle)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        real par_azi = static_cast<real>(RES_AZI)*   (dev_particle[idx].position.x - AZI_MIN) /    (AZI_MAX - AZI_MIN);
        real par_rad = static_cast<real>(RES_RAD)*log(dev_particle[idx].position.y / RAD_MIN) / log(RAD_MAX / RAD_MIN);
        real par_col = static_cast<real>(RES_COL)*   (dev_particle[idx].position.z - COL_MIN) /    (COL_MAX - COL_MIN);

        bool inside_azi = par_azi >= 0.0 && par_azi < static_cast<real>(RES_AZI);
        bool inside_rad = par_rad >= 0.0 && par_rad < static_cast<real>(RES_RAD);
        bool inside_col = par_col >= 0.0 && par_col < static_cast<real>(RES_COL);

        if (inside_azi && inside_rad && inside_col)
        {
            interp result = linear_interp_cent(par_azi, par_rad, par_col);

            int  next_azi = result.next_azi;
            int  next_rad = result.next_rad;
            int  next_col = result.next_col;
            real frac_azi = result.frac_azi;
            real frac_rad = result.frac_rad;
            real frac_col = result.frac_col;

            int idx_cell = static_cast<int>(par_col)*NUM_COL + static_cast<int>(par_rad)*RES_AZI + static_cast<int>(par_azi);

            real grain_size = dev_particle[idx].grain_size;
            real grain_numr = dev_particle[idx].grain_numr;
            
            real weight = KAPPA_REF*SIZE_REF*RHO_DUST*grain_numr*grain_size*grain_size;

            atomicAdd(&dev_optdepth[idx_cell                                 ], (1.0-frac_azi)*(1.0-frac_rad)*(1.0-frac_col)*weight);
            atomicAdd(&dev_optdepth[idx_cell + next_azi                      ],      frac_azi *(1.0-frac_rad)*(1.0-frac_col)*weight);
            atomicAdd(&dev_optdepth[idx_cell            + next_rad           ], (1.0-frac_azi)*     frac_rad *(1.0-frac_col)*weight);
            atomicAdd(&dev_optdepth[idx_cell + next_azi + next_rad           ],      frac_azi *     frac_rad *(1.0-frac_col)*weight);
            atomicAdd(&dev_optdepth[idx_cell                       + next_col], (1.0-frac_azi)*(1.0-frac_rad)*     frac_col *weight);
            atomicAdd(&dev_optdepth[idx_cell + next_azi            + next_col],      frac_azi *(1.0-frac_rad)*     frac_col *weight);
            atomicAdd(&dev_optdepth[idx_cell            + next_rad + next_col], (1.0-frac_azi)*     frac_rad *     frac_col *weight);
            atomicAdd(&dev_optdepth[idx_cell + next_azi + next_rad + next_col],      frac_azi *     frac_rad *     frac_col *weight);
        }   
    } 
}

__global__
void dustdens_enum (real *dev_dustdens, swarm *dev_particle)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        real par_azi = static_cast<real>(RES_AZI)*   (dev_particle[idx].position.x - AZI_MIN) /    (AZI_MAX - AZI_MIN);
        real par_rad = static_cast<real>(RES_RAD)*log(dev_particle[idx].position.y / RAD_MIN) / log(RAD_MAX / RAD_MIN);
        real par_col = static_cast<real>(RES_COL)*   (dev_particle[idx].position.z - COL_MIN) /    (COL_MAX - COL_MIN);

        bool inside_azi = par_azi >= 0.0 && par_azi < static_cast<real>(RES_AZI);
        bool inside_rad = par_rad >= 0.0 && par_rad < static_cast<real>(RES_RAD);
        bool inside_col = par_col >= 0.0 && par_col < static_cast<real>(RES_COL);

        if (inside_azi && inside_rad && inside_col)
        {
            interp result = linear_interp_cent(par_azi, par_rad, par_col);

            int  next_azi = result.next_azi;
            int  next_rad = result.next_rad;
            int  next_col = result.next_col;
            real frac_azi = result.frac_azi;
            real frac_rad = result.frac_rad;
            real frac_col = result.frac_col;

            int idx_cell = static_cast<int>(par_col)*NUM_COL + static_cast<int>(par_rad)*RES_AZI + static_cast<int>(par_azi);

            real grain_size = dev_particle[idx].grain_size;
            real grain_numr = dev_particle[idx].grain_numr;

            real weight = RHO_DUST*grain_numr*grain_size*grain_size*grain_size;

            atomicAdd(&dev_dustdens[idx_cell                                 ], (1.0-frac_azi)*(1.0-frac_rad)*(1.0-frac_col)*weight);
            atomicAdd(&dev_dustdens[idx_cell + next_azi                      ],      frac_azi *(1.0-frac_rad)*(1.0-frac_col)*weight);
            atomicAdd(&dev_dustdens[idx_cell            + next_rad           ], (1.0-frac_azi)*     frac_rad *(1.0-frac_col)*weight);
            atomicAdd(&dev_dustdens[idx_cell + next_azi + next_rad           ],      frac_azi *     frac_rad *(1.0-frac_col)*weight);
            atomicAdd(&dev_dustdens[idx_cell                       + next_col], (1.0-frac_azi)*(1.0-frac_rad)*     frac_col *weight);
            atomicAdd(&dev_dustdens[idx_cell + next_azi            + next_col],      frac_azi *(1.0-frac_rad)*     frac_col *weight);
            atomicAdd(&dev_dustdens[idx_cell            + next_rad + next_col], (1.0-frac_azi)*     frac_rad *     frac_col *weight);
            atomicAdd(&dev_dustdens[idx_cell + next_azi + next_rad + next_col],      frac_azi *     frac_rad *     frac_col *weight);
        }   
    } 
}

// =========================================================================================================================

__device__
real _get_optdepth (real *dev_optdepth, real par_azi, real par_rad, real par_col)
{
    real optdepth = 0.0;

    bool inside_azi = par_azi >= 0.0 && par_azi < static_cast<real>(RES_AZI);
    bool inside_rad = par_rad >= 0.0 && par_rad < static_cast<real>(RES_RAD);
    bool inside_col = par_col >= 0.0 && par_col < static_cast<real>(RES_COL);

    if (inside_azi && inside_rad && inside_col)
    {
        interp result = linear_interp_stag(par_azi, par_rad, par_col);

        int  next_azi = result.next_azi;
        int  next_rad = result.next_rad;
        int  next_col = result.next_col;
        real frac_azi = result.frac_azi;
        real frac_rad = result.frac_rad;
        real frac_col = result.frac_col;

        int idx_cell = static_cast<int>(par_col)*NUM_COL + static_cast<int>(par_rad)*RES_AZI + static_cast<int>(par_azi);

        optdepth += dev_optdepth[idx_cell                                 ]*(1.0-frac_azi)*(1.0-frac_rad)*(1.0-frac_col);
        optdepth += dev_optdepth[idx_cell + next_azi                      ]*     frac_azi *(1.0-frac_rad)*(1.0-frac_col);
        optdepth += dev_optdepth[idx_cell            + next_rad           ]*(1.0-frac_azi)*     frac_rad *(1.0-frac_col);
        optdepth += dev_optdepth[idx_cell + next_azi + next_rad           ]*     frac_azi *     frac_rad *(1.0-frac_col);
        optdepth += dev_optdepth[idx_cell                       + next_col]*(1.0-frac_azi)*(1.0-frac_rad)*     frac_col ;
        optdepth += dev_optdepth[idx_cell + next_azi            + next_col]*     frac_azi *(1.0-frac_rad)*     frac_col ;
        optdepth += dev_optdepth[idx_cell            + next_rad + next_col]*(1.0-frac_azi)*     frac_rad *     frac_col ;
        optdepth += dev_optdepth[idx_cell + next_azi + next_rad + next_col]*     frac_azi *     frac_rad *     frac_col ;
    }
    else if (par_rad >= RES_RAD)
    {
        optdepth = DBL_MAX;
    }

    return optdepth;
}

// =========================================================================================================================

__global__
void optdepth_calc (real *dev_optdepth)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_DIM)
    {	
        int idx_azi, idx_rad, idx_col;

        idx_azi =  idx % RES_AZI;
        idx_rad = (idx % NUM_COL         - idx_azi) / RES_AZI;
        idx_col = (idx - idx_rad*RES_AZI - idx_azi) / NUM_COL;

        real d_azi, d_rad, d_col, rad_in, col_in, volume;

        d_azi =    (AZI_MAX - AZI_MIN)     / static_cast<real>(RES_AZI);
        d_rad = pow(RAD_MAX / RAD_MIN, 1.0 / static_cast<real>(RES_RAD));
        d_col =    (COL_MAX - COL_MIN)     / static_cast<real>(RES_COL);

        rad_in = RAD_MIN*pow(d_rad, static_cast<real>(idx_rad));
        col_in = COL_MIN+    d_col* static_cast<real>(idx_col);

        volume  = d_azi;
        volume *= rad_in*rad_in*rad_in*(d_rad*d_rad*d_rad - 1.0) / 3.0;
        volume *= cos(col_in) - cos(col_in + d_col);

        dev_optdepth[idx] /= volume;
        dev_optdepth[idx] *= (d_rad - 1.0)*rad_in; // prepare for radial integration
    }
}

__global__
void dustdens_calc (real *dev_dustdens)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_DIM)
    {	
        int idx_azi, idx_rad, idx_col;

        idx_azi =  idx % RES_AZI;
        idx_rad = (idx % NUM_COL         - idx_azi) / RES_AZI;
        idx_col = (idx - idx_rad*RES_AZI - idx_azi) / NUM_COL;

        real d_azi, d_rad, d_col, rad_in, col_in, volume;

        d_azi =    (AZI_MAX - AZI_MIN)     / static_cast<real>(RES_AZI);
        d_rad = pow(RAD_MAX / RAD_MIN, 1.0 / static_cast<real>(RES_RAD));
        d_col =    (COL_MAX - COL_MIN)     / static_cast<real>(RES_COL);

        rad_in = RAD_MIN*pow(d_rad, static_cast<real>(idx_rad));
        col_in = COL_MIN+    d_col* static_cast<real>(idx_col);

        volume  = d_azi;
        volume *= rad_in*rad_in*rad_in*(d_rad*d_rad*d_rad - 1.0) / 3.0;
        volume *= cos(col_in) - cos(col_in + d_col);

        dev_dustdens[idx] /= volume;
    }
}

// =========================================================================================================================

__global__
void optdepth_inte (real *dev_optdepth)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_RAD)
    {	
        int idx_azi, idx_col, idx_cell;

        idx_azi = idx % RES_AZI;
        idx_col = (idx - idx_azi) / RES_AZI;

        for (int i = 1; i < RES_RAD; i++)
        {
            idx_cell = idx_col*NUM_COL + i*RES_AZI + idx_azi;
            dev_optdepth[idx_cell] += dev_optdepth[idx_cell - RES_AZI];
        }
    }
}

__global__
void optdepth_mean (real *dev_optdepth)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_AZI)
    {	
        int idx_rad, idx_col, idx_cell;

        real optdepth_sum = 0.0;

        idx_rad = idx % RES_RAD;
        idx_col = (idx - idx_rad) / RES_RAD;

        for (int i = 0; i < RES_AZI; i++)
        {
            idx_cell = idx_col*NUM_COL + idx_rad*RES_AZI + i;
            optdepth_sum += dev_optdepth[idx_cell];
        }

        for (int j = 0; j < RES_AZI; j++)
        {
            idx_cell = idx_col*NUM_COL + idx_rad*RES_AZI + j;
            dev_optdepth[idx_cell] = optdepth_sum / RES_AZI;
        }
    }
}
