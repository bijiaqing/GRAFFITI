#include "cudust.cuh"

// =========================================================================================================================

__device__ 
interp _linear_interp_cent (real loc_x, real loc_y, real loc_z)
{
    // loc_x, loc_y, loc_z are the precise particle locations in the cell-index coordinates

    bool edge_x, edge_y, edge_z;    // whether the particle is too close to the domain boundaries
    real frac_x, frac_y, frac_z;    // the fraction of a particle NOT shared by the cell after interpolation
    real deci_x, deci_y, deci_z;    // the decimal value of particle locations in the cell-index coordinates
    real cent_x, cent_y, cent_z;    // the location of the center of the cell  in the cell-index coordinates
    int  next_x, next_y, next_z;    // the distance to the neighboring cell    in the cell-index coordinates

    deci_x = loc_x - floor(loc_x);
    deci_y = loc_y - floor(loc_y);
    deci_z = loc_z - floor(loc_z);

    real dy = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y)); // exponential base for logarithmic spacing

    cent_x = 0.5;
    cent_y = log(0.5*(1.0 + dy)) / log(dy); // dy^cent_y is the geometric center of the cell
    cent_z = 0.5;

    if (N_X == 1)                   // if there is only one cell in X
    {
        frac_x = 0;                 // the share for the current cell is '1.0 - frac_x'
        next_x = 0;                 // no other cells to share the particle
    }
    else
    {
        edge_x = loc_x < cent_x || loc_x > N_X - cent_x;

        if (not edge_x)             // still in the interior of the X domain
        {
            if (deci_x >= cent_x)   // share with the cell on the right
            {
                frac_x = deci_x - cent_x;
                next_x = 1;
            }
            else                    // share with the cell on the left
            {
                frac_x = cent_x - deci_x;
                next_x = -1;
            }
        }
        else                        // too close to the inner or the outer X boundary 
        {
            if (deci_x >= cent_x)   // too close to the outer X boundary
            {
                frac_x = deci_x - cent_x;
                next_x = 1 - N_X;   // share with the first cell of its row
            }
            else                    // too close to the inner X boundary
            {
                frac_x = cent_x - deci_x;
                next_x = N_X - 1;   // share with the last  cell of its row
            }
        }
    }

    if (N_Y == 1)                   // if there is only one cell in Y
    {
        frac_y = 0;                 // the share for the current cell is '1.0 - frac_y'
        next_y = 0;                 // no other cells to share the particle
    }
    else
    {
        edge_y = loc_y < cent_y || loc_y > N_Y - cent_y;

        if (not edge_y)             // still in the interior of the Y domain
        {
            if (deci_y >= cent_y)   // share with the cell on the right
            {
                frac_y = (pow(dy, deci_y - cent_y) - 1.0) / (dy - 1.0);
                next_y = N_X;       // the index distance to the next Y cell on the right is N_X
            }
            else                    // share with the cell on the left
            {
                frac_y = (pow(dy, deci_y - cent_y) - 1.0) / (1.0 / dy - 1.0);
                next_y = - N_X;
            }
        }
        else                        // at the Y domain boundaries, the current cell take it all like N_Y = 1
        {
            frac_y = 0;
            next_y = 0;
        }
    }


    if (N_Z == 1)                   // if there is only one cell in Z
    {
        frac_z = 0;                 // the share for the current cell is '1.0 - frac_z'
        next_z = 0;                 // no other cells to share the particle
    }
    else
    {
        edge_z = loc_z < cent_z || loc_z > N_Z - cent_z;
        
        if (not edge_z)             // still in the interior of the Z domain
        {
            if (deci_z >= cent_z)
            {
                frac_z = deci_z - cent_z;
                next_z = NG_XY;    // the index distance to the next Z cell on the right is N_X*N_Y = NG_XY
            }
            else
            {
                frac_z = cent_z - deci_z;
                next_z = - NG_XY;
            }
        }
        else                        // at the Z domain boundaries, the current cell take it all like N_Z = 1
        {
            frac_z = 0;
            next_z = 0;
        }
    }

    interp result;

    result.next_x = next_x;
    result.next_y = next_y;
    result.next_z = next_z;
    result.frac_x = frac_x;
    result.frac_y = frac_y;
    result.frac_z = frac_z;

    return result;
}

// =========================================================================================================================

__device__ 
interp _linear_interp_edge (real loc_x, real loc_y, real loc_z)
{
    // this function exists because the optical depth field is defined at the outer radial boundary of each cell
    // the particle needs to be interpolated based on the location of the radial cell edges to get the shadow
    // the only difference between '_linear_interp_edge' and '_linear_interp_cent' is the interpolation rule in Y

    bool edge_x, edge_y, edge_z;    // whether the particle is too close to the domain boundaries
    real frac_x, frac_y, frac_z;    // the fraction of a particle NOT shared by the cell after interpolation
    real deci_x, deci_y, deci_z;    // the decimal value of particle locations in the cell-index coordinates
    real cent_x, cent_y, cent_z;    // the location of the center of the cell  in the cell-index coordinates
    int  next_x, next_y, next_z;    // the distance to the neighboring cell    in the cell-index coordinates

    deci_x = loc_x - floor(loc_x);
    deci_y = loc_y - floor(loc_y);
    deci_z = loc_z - floor(loc_z);

    real dy = pow(Y_MAX / Y_MIN, 1.0 / static_cast<real>(N_Y)); // exponential base for logarithmic spacing

    cent_x = 0.5;
    cent_y = 1.0;
    cent_z = 0.5;

    if (N_X == 1)                   // if there is only one cell in X
    {
        frac_x = 0;                 // the share for the current cell is '1.0 - frac_x'
        next_x = 0;                 // no other cells to share the particle
    }
    else
    {
        edge_x = loc_x < cent_x || loc_x > N_X - cent_x;

        if (not edge_x)             // still in the interior of the X domain
        {
            if (deci_x >= cent_x)   // share with the cell on the right
            {
                frac_x = deci_x - cent_x;
                next_x = 1;
            }
            else                    // share with the cell on the left
            {
                frac_x = cent_x - deci_x;
                next_x = -1;
            }
        }
        else                        // too close to the inner or the outer X boundary 
        {
            if (deci_x >= cent_x)   // too close to the outer X boundary
            {
                frac_x = deci_x - cent_x;
                next_x = 1 - N_X;   // share with the first cell of its row
            }
            else                    // too close to the inner X boundary
            {
                frac_x = cent_x - deci_x;
                next_x = N_X - 1;   // share with the last  cell of its row
            }
        }
    }

    if (N_Y == 1)                   // if there is only one cell in Y
    {
        frac_y = 0;                 // the share for the current cell is '1.0 - frac_y'
        next_y = 0;                 // no other cells to share the particle
    }
    else
    {
        edge_y = loc_y < cent_y;

        if (not edge_y)
        {
            frac_y = (dy - pow(dy, deci_y)) / (dy - 1.0);
            next_y = - N_X;         // share with the cell on its left
        }
        else                        // if the particle is in the innermost radial layer
        {
            frac_y = (dy - pow(dy, deci_y)) / (dy - 1.0);
            next_y = 0;             // the particles are unfortunately 100% self-shadowed
        }
    }

    if (N_Z == 1)                   // if there is only one cell in Z
    {
        frac_z = 0;                 // the share for the current cell is '1.0 - frac_z'
        next_z = 0;                 // no other cells to share the particle
    }
    else
    {
        edge_z = loc_z < cent_z || loc_z > N_Z - cent_z;
        
        if (not edge_z)             // still in the interior of the Z domain
        {
            if (deci_z >= cent_z)
            {
                frac_z = deci_z - cent_z;
                next_z = NG_XY;    // the index distance to the next Z cell on the right is N_X*N_Y = NG_XY
            }
            else
            {
                frac_z = cent_z - deci_z;
                next_z = - NG_XY;
            }
        }
        else                        // at the Z domain boundaries, the current cell take it all like N_Z = 1
        {
            frac_z = 0;
            next_z = 0;
        }
    }

    interp result;

    result.next_x = next_x;
    result.next_y = next_y;
    result.next_z = next_z;
    result.frac_x = frac_x;
    result.frac_y = frac_y;
    result.frac_z = frac_z;

    return result;
}

// =========================================================================================================================

__device__
real _get_optdepth (real *dev_optdepth, real loc_x, real loc_y, real loc_z)
{
    real optdepth = 0.0;

    bool in_x = loc_x >= 0.0 && loc_x < static_cast<real>(N_X);
    bool in_y = loc_y >= 0.0 && loc_y < static_cast<real>(N_Y);
    bool in_z = loc_z >= 0.0 && loc_z < static_cast<real>(N_Z);

    if (in_x && in_y && in_z)
    {
        interp result = _linear_interp_edge(loc_x, loc_y, loc_z);

        int  next_x = result.next_x;
        int  next_y = result.next_y;
        int  next_z = result.next_z;
        real frac_x = result.frac_x;
        real frac_y = result.frac_y;
        real frac_z = result.frac_z;

        int idx_cell = static_cast<int>(loc_z)*NG_XY + static_cast<int>(loc_y)*N_X + static_cast<int>(loc_x);

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