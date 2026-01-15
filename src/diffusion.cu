#include "cudust.cuh"

#ifdef DIFFUSION

// =========================================================================================================================

__global__
void pos_diffusion (swarm *dev_particle, curs *dev_rs_swarm, real dt)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        real y = dev_particle[idx].position.y;
        real z = dev_particle[idx].position.z;

        real polar_R = y*sin(z);
        real hgas_sq = ASPR_0*ASPR_0*pow(polar_R, IDX_Q + 1.0);
        
        #ifndef CONST_NU
        real nu = ALPHA*hgas_sq*sqrt(G*M_S*polar_R);     // kinematic viscosity
        #else
        real nu = NU;
        #endif // CONST_NU
        
        real coeff_R = nu / SCHMIDT_R;                  // radial diffusion coefficient

        // assuming sigma_g ~ pow(R, IDX_P), then
        // (1) rho_g ~ sigma_g / H_gas ~ pow(R, IDX_P - 0.5*IDX_Q - 1.5)
        // (2) d(rhog)/dR/rhog = (IDX_P - 0.5*IDX_Q - 1.5) / R
        real avg_R1 = dt*coeff_R*(IDX_P - 0.5*IDX_Q - 1.5) / polar_R;
        real var_R1 = dt*coeff_R*2.0;

        #ifndef CONST_NU
        // given nu ~ alpha*H_gas*c_s ~ pow(R, IDX_Q + 1.5), then d(coeff_R)/dR = (IDX_Q + 1.5)*coeff_R / R
        real avg_R2 = dt*coeff_R*(IDX_Q + 1.5) / polar_R;
        real var_R2 = avg_R2*avg_R2;
        #else
        real avg_R2 = 0.0;
        real var_R2 = 0.0;
        #endif // CONST_NU

        curs rs_swarm = dev_rs_swarm[idx];          // use a local state for less global memory traffic 

        real delta_R = (avg_R1 + avg_R2) + sqrt(var_R1 + var_R2)*curand_normal_double(&rs_swarm);

        if (N_Z == 1)
        {
            dev_particle[idx].position.y = polar_R + delta_R;
        }
        else // Y-direction must exist, include vertical diffusion
        {
            real polar_Z = y*cos(z);
            real coeff_Z = nu / SCHMIDT_Z;              // vertical diffusion coefficient

            // assuming rho_g ~ exp(-Z^2 / (2*H_gas^2)), then
            // (1) d(rhog)/dZ/rhog = -Z / H_gas^2
            real avg_Z1 = dt*coeff_Z*(-polar_Z / (hgas_sq*polar_R*polar_R));
            real var_Z1 = dt*coeff_Z*2.0;

            real avg_Z2 = 0.0; // since coeff_Z is constant in Z-direction
            real var_Z2 = 0.0;

            real delta_Z = (avg_Z1 + avg_Z2) + sqrt(var_Z1 + var_Z2)*curand_normal_double(&rs_swarm);

            dev_particle[idx].position.y = sqrt((polar_R + delta_R)*(polar_R + delta_R) + (polar_Z + delta_Z)*(polar_Z + delta_Z));
            dev_particle[idx].position.z = atan2(polar_R + delta_R, polar_Z + delta_Z);
        }

        dev_rs_swarm[idx] = rs_swarm;               // update the global state, otherwise, always the same number
    }
}

#endif // DIFFUSION

// =========================================================================================================================