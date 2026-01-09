#include "cudust.cuh"

// =========================================================================================================================

__global__
void pos_diffusion (swarm *dev_particle, curandState *dev_rngs_par, real dt)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        real y = dev_particle[idx].position.y;
        real z = dev_particle[idx].position.z;

        real bigR = y*sin(z);
        real bigZ = y*cos(z);

        real h_sqr = h_0*h_0*pow(bigR, IDX_TEMP + 1.0);
        
        real nu = ALPHA*h_sqr*sqrt(G*M_S*bigR);     // kinematic viscosity
        real D_R = nu / SCHMIDT_R;                  // radial diffusion coefficient
        real D_Z = nu / SCHMIDT_Z;                  // vertical diffusion coefficient
        
        // ( d(rhog)/dR ) / rhog = p / R = (IDX_SURF - 0.5*IDX_TEMP - 1.5) / R
        // ( d(diff)/dR ) = (2f + 0.5)*diff / R = (IDX_TEMP + 1.5)*diff / R
        real delta_avg_y = (IDX_SURF + 0.5*IDX_TEMP)*dt*D_R / bigR;
        real sigma_sqr_y = 2.0*D_R*dt + (IDX_TEMP + 1.5)*(IDX_TEMP + 1.5)*D_R*D_R*dt*dt / bigR / bigR;
        
        // ( d(rhog)/dZ ) / rhog = -z / H_g^2
        // ( d(diff)/dZ ) = 0
        real delta_avg_z = -bigZ*dt*D_Z / (h_sqr*bigR*bigR);
        real sigma_sqr_z = 2.0*D_Z*dt;
        
        curandState rngs_par = dev_rngs_par[idx];   // use a local state for less global memory traffic 

        real delta_R = delta_avg_y + sqrt(sigma_sqr_y)*curand_normal_double(&rngs_par);    // in polar R
        real delta_Z = delta_avg_z + sqrt(sigma_sqr_z)*curand_normal_double(&rngs_par);    // in polar Z

        dev_rngs_par[idx] = rngs_par;               // update the global state, otherwise, always the same number

        if (N_Z == 1)
        {
            dev_particle[idx].position.y = bigR + delta_R;
        }
        else // Y-direction must exist
        {
            dev_particle[idx].position.y = sqrt((bigR + delta_R)*(bigR + delta_R) + (bigZ + delta_Z)*(bigZ + delta_Z));
            dev_particle[idx].position.z = atan2(bigR + delta_R, bigZ + delta_Z);
        }
    }
}
