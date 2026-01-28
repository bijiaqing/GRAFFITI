#if defined(TRANSPORT) && defined(DIFFUSION)

#include "cudust_kern.cuh"
#include "helpers_diskparam.cuh"

// =========================================================================================================================
// Kernel: diffusion_pos
// Purpose: Apply turbulent diffusion effects to particle positions (radial and vertical)
// Dependencies: cudust.cuh (for types and constants),
//               helpers_diskparam.cuh (for _get_hgas, _get_nu),
//               curand (for normal distribution random numbers)
// =========================================================================================================================

__global__
void diffusion_pos (swarm *dev_particle, curs *dev_rs_swarm, real dt)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_P)
    {
        real y = dev_particle[idx].position.y;
        real z = dev_particle[idx].position.z;

        real polar_R = y*sin(z);
        real h_gas = _get_hgas(polar_R);
        real nu = _get_nu(polar_R, h_gas);
        
        real coeff_R = nu / SC_R; // radial diffusion coefficient

        // assuming sigma_g ~ pow(R, IDX_P), then
        // (1) rho_g ~ sigma_g / H_gas ~ pow(R, IDX_P - 0.5*IDX_Q - 1.5)
        // (2) d(rhog)/dR/rhog = (IDX_P - 0.5*IDX_Q - 1.5) / R
        real avg_R1 = dt*coeff_R*(IDX_P - 0.5*IDX_Q - 1.5) / polar_R;
        real var_R1 = dt*coeff_R*2.0;

        #ifndef CONST_NU
        // given nu ~ alpha*H_gas*c_s ~ pow(R, IDX_Q + 1.5), then d(coeff_R)/dR = (IDX_Q + 1.5)*coeff_R / R
        real avg_R2 = dt*coeff_R*(IDX_Q + 1.5) / polar_R;
        real var_R2 = avg_R2*avg_R2;
        #else  // CONST_NU
        real avg_R2 = 0.0;
        real var_R2 = 0.0;
        #endif // NOT CONST_NU

        curs rs_swarm = dev_rs_swarm[idx]; // use a local state for less global memory traffic 

        real delta_R = (avg_R1 + avg_R2) + sqrt(var_R1 + var_R2)*curand_normal_double(&rs_swarm);

        if (N_Z == 1)
        {
            dev_particle[idx].position.y = polar_R + delta_R;
        }
        else // Y-direction must exist, include vertical diffusion
        {
            real polar_Z = y*cos(z);
            real coeff_Z = nu / SC_Z;    // vertical diffusion coefficient

            // assuming rho_g ~ exp(-Z^2 / (2*H_gas^2)), then
            // (1) d(rhog)/dZ/rhog = -Z / H_gas^2
            real avg_Z1 = dt*coeff_Z*(-polar_Z / (h_gas*h_gas*polar_R*polar_R));
            real var_Z1 = dt*coeff_Z*2.0;

            // assuming both nu and alpha are constant in Z-direction
            real avg_Z2 = 0.0;
            real var_Z2 = 0.0;

            real delta_Z = (avg_Z1 + avg_Z2) + sqrt(var_Z1 + var_Z2)*curand_normal_double(&rs_swarm);

            dev_particle[idx].position.y = sqrt((polar_R + delta_R)*(polar_R + delta_R) + (polar_Z + delta_Z)*(polar_Z + delta_Z));
            dev_particle[idx].position.z = atan2(polar_R + delta_R, polar_Z + delta_Z);
        }

        dev_rs_swarm[idx] = rs_swarm; // update the global state, otherwise, always the same number
    }
}

// =========================================================================================================================

#endif // DIFFUSION
