#include "cudust.cuh"

// =========================================================================================================================

__global__ 
void ssa_substep_1 (swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        real dt = DT_DYNAMICS;

        real x_i  = dev_particle[idx].position.x;
        real y_i  = dev_particle[idx].position.y;
        real z_i  = dev_particle[idx].position.z;
        
        real lx_i = dev_particle[idx].velocity.x;
        real vy_i = dev_particle[idx].velocity.y;
        real lz_i = dev_particle[idx].velocity.z;

        real y_1 = y_i + 0.5*vy_i*dt;
        real z_1 = z_i + 0.5*lz_i*dt / y_i / y_1;
        real x_1 = x_i + 0.5*lx_i*dt / y_i / y_1 / sin(z_i) / sin(z_1);

        if (N_X == 1) 
        {
            x_1 = 0.5*(X_MIN + X_MAX);
        }
        else
        {
            // keep x within [X_MIN, X_MAX)
            while (x_1 >= X_MAX) x_1 -= X_MAX - X_MIN;
            while (x_1 <  X_MIN) x_1 += X_MAX - X_MIN;
        }

        if (y_1 < Y_MIN)    
        {
            // throw it to the end of the radial domain
            y_1 = Y_MAX;
            z_1 = 0.5*M_PI;
            dev_particle[idx].velocity.x = sqrt(G*M_S*Y_MAX);
            dev_particle[idx].velocity.y = 0.0;
            dev_particle[idx].velocity.z = 0.0;
        }

        if (N_Z == 1)
        {
            z_1 = 0.5*M_PI;
        }
        else if (z_1 < Z_MIN || z_1 >= Z_MAX)
        {
            // z_1 = ( z_1 < Z_MIN ) ? (2.0*Z_MIN - z_1) : (2.0*Z_MAX - z_1); // reflect at the boundary
            // z_1 = 0.5*M_PI; // throw it to the mid-plane
            
            // throw it to the end of the radial domain
            y_1 = Y_MAX;
            z_1 = 0.5*M_PI;
            dev_particle[idx].velocity.x = sqrt(G*M_S*Y_MAX);
            dev_particle[idx].velocity.y = 0.0;
            dev_particle[idx].velocity.z = 0.0;
        }

        dev_particle[idx].position.x = x_1;
        dev_particle[idx].position.y = y_1;
        dev_particle[idx].position.z = z_1;
    }
}

// =========================================================================================================================

__global__
void ssa_substep_2 (swarm *dev_particle, real *dev_optdepth)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        real dt = DT_DYNAMICS;

        real x_1  = dev_particle[idx].position.x;
        real y_1  = dev_particle[idx].position.y;
        real z_1  = dev_particle[idx].position.z;
        
        real lx_i = dev_particle[idx].velocity.x;
        real vy_i = dev_particle[idx].velocity.y;
        real lz_i = dev_particle[idx].velocity.z;
        
        real s    = dev_particle[idx].par_size;

        real bigR_1 = y_1*sin(z_1);
        real bigZ_1 = y_1*cos(z_1);

        real h_sqr = h_0*h_0*pow(bigR_1, IDX_TEMP + 1.0);

        // get the velocity of gas in the hydrostatic equilibrium state
        real eta_1; // eta_1 is normally a negative value
        eta_1  = (IDX_SURF + 0.5*IDX_TEMP - 1.5)*h_sqr; // (p + q)h^2
        eta_1 += IDX_TEMP*(1.0 - bigR_1 / y_1);
        
        real lxg_1 = sqrt(G*M_S*bigR_1)*sqrt(1.0 + eta_1);
        real vyg_1 = 0.0;
        real lzg_1 = 0.0;

        // calculate the stopping time and the dimensionless time
        real ts_1;
        ts_1  = ST_0*(s / S_0) / sqrt(G*M_S / R_0 / R_0 / R_0); // for reference size grain at the reference radius
        ts_1 *= pow(bigR_1, -IDX_SURF + 1.5);                   // correct for radial gas density and sound speed
        ts_1 *= exp(-bigZ_1*bigZ_1 / (2.0*h_sqr*bigR_1*bigR_1));   // correct for vertical gas density
        
        real tau_1 = dt / ts_1;

        // retrieve the optical depth of the particle based on its position and calculate beta
        real loc_x = (N_X > 1) ? (static_cast<real>(N_X)*   (x_1 - X_MIN) /    (X_MAX - X_MIN)) : 0.0;
        real loc_y = (N_Y > 1) ? (static_cast<real>(N_Y)*log(y_1 / Y_MIN) / log(Y_MAX / Y_MIN)) : 0.0;
        real loc_z = (N_Z > 1) ? (static_cast<real>(N_Z)*   (z_1 - Z_MIN) /    (Z_MAX - Z_MIN)) : 0.0;

        real optdepth = _get_optdepth(dev_optdepth, loc_x, loc_y, loc_z);
        real beta_1 = BETA_0*exp(-optdepth) / (s / S_0);

        // calculate the forces and torques (using the updated position but outdated velocity)
        real Tx_1 =  0.0;
        real Fy_1 = -(1.0 - beta_1)*G*M_S / y_1 / y_1;
        real Tz_1 =  0.0;

        // calculate the centrifugal forces (using the updated position but outdated velocity)
        real Fcy_1 = lx_i*lx_i / bigR_1 / bigR_1 / y_1 + lz_i*lz_i / y_1 / y_1 / y_1;
        real Tcz_1 = lx_i*lx_i / bigR_1 / bigR_1 / sin(z_1) * cos(z_1);

        // calculate the updated velocities
        real lx_1 = lx_i + ((Tx_1        )*ts_1 + lxg_1 - lx_i)*(1.0 - exp(-0.5*tau_1));
        real vy_1 = vy_i + ((Fy_1 + Fcy_1)*ts_1 + vyg_1 - vy_i)*(1.0 - exp(-0.5*tau_1));
        real lz_1 = lz_i + ((Tz_1 + Tcz_1)*ts_1 + lzg_1 - lz_i)*(1.0 - exp(-0.5*tau_1));

        // calculate the forces and torques (using the updated position and velocity)
        real Tx_2 =  0.0;
        real Fy_2 = -(1.0 - beta_1)*G*M_S / y_1 / y_1;
        real Tz_2 =  0.0;

        // calculate the centrifugal forces (using the updated position and velocity)
        real Fcy_2 = lx_1*lx_1 / bigR_1 / bigR_1 / y_1 + lz_1*lz_1 / y_1 / y_1 / y_1;
        real Tcz_2 = lx_1*lx_1 / bigR_1 / bigR_1 / sin(z_1) * cos(z_1);

        // calculate the next-step velocity
        real lx_j = lx_i + ((Tx_2        )*ts_1 + lxg_1 - lx_i)*(1.0 - exp(-tau_1));
        real vy_j = vy_i + ((Fy_2 + Fcy_2)*ts_1 + vyg_1 - vy_i)*(1.0 - exp(-tau_1));
        real lz_j = lz_i + ((Tz_2 + Tcz_2)*ts_1 + lzg_1 - lz_i)*(1.0 - exp(-tau_1));

        // calculate the next-step position (the sequence matters!!)
        real y_j = y_1 + 0.5*vy_j*dt;
        real z_j = z_1 + 0.5*lz_j*dt / y_1 / y_j;
        real x_j = x_1 + 0.5*lx_j*dt / y_1 / y_j / sin(z_1) / sin(z_j);

        if (N_X == 1) 
        {
            x_j = 0.5*(X_MIN + X_MAX);
        }
        else
        {
            // keep x within [X_MIN, X_MAX)
            while (x_j >= X_MAX) x_j -= X_MAX - X_MIN;
            while (x_j <  X_MIN) x_j += X_MAX - X_MIN;
        }

        if (y_j < Y_MIN)    // throw it to the end of the radial domain
        {
            y_j  = Y_MAX;
            z_j  = 0.5*M_PI;
            lx_j = sqrt(G*M_S*Y_MAX);
            vy_j = 0.0;
            lz_j = 0.0;
        }

        if (N_Z == 1)
        {
            z_j = 0.5*M_PI;
        }
        else if (z_j < Z_MIN || z_j >= Z_MAX)
        {
            // z_j = ( z_j < Z_MIN ) ? (2.0*Z_MIN - z_j) : (2.0*Z_MAX - z_j); // reflect at the boundary
            // z_j = 0.5*M_PI; // throw it to the mid-plane
            
            // throw it to the end of the radial domain
            y_j  = Y_MAX;
            z_j  = 0.5*M_PI;
            lx_j = sqrt(G*M_S*Y_MAX);
            vy_j = 0.0;
            lz_j = 0.0;
        }

        dev_particle[idx].position.x = x_j;
        dev_particle[idx].position.y = y_j;
        dev_particle[idx].position.z = z_j;
        
        dev_particle[idx].velocity.x = lx_j;
        dev_particle[idx].velocity.y = vy_j;
        dev_particle[idx].velocity.z = lz_j;
    }
}

// =========================================================================================================================

__global__
void pos_diffusion (swarm *dev_particle, curandState *dev_rngs_par)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        real dt = DT_DYNAMICS;
        
        real y = dev_particle[idx].position.y;
        real z = dev_particle[idx].position.z;

        real bigR = y*sin(z);
        real bigZ = y*cos(z);

        real h_sqr = h_0*h_0*pow(bigR, IDX_TEMP + 1.0);

        real D_diff = ALPHA*h_sqr*sqrt(G*M_S*bigR);
        
        // ( d(rhog)/dR ) / rhog = p / R = (IDX_SURF - 0.5*IDX_TEMP - 1.5) / R
        // ( d(diff)/dR ) = (2f + 0.5)*diff / R = (IDX_TEMP + 1.5)*diff / R
        real delta_avg_y = (IDX_SURF + 0.5*IDX_TEMP)*dt*D_diff / bigR;
        real sigma_sqr_y = 2.0*D_diff*dt + (IDX_TEMP + 1.5)*(IDX_TEMP + 1.5)*D_diff*D_diff*dt*dt / bigR / bigR;
        
        // ( d(rhog)/dZ ) / rhog = -z / H_g^2
        // ( d(diff)/dZ ) = 0
        real delta_avg_z = -bigZ*dt*D_diff / (h_sqr*bigR*bigR);
        real sigma_sqr_z = 2.0*D_diff*dt;
        
        curandState rngs_par = dev_rngs_par[idx];   // use a local state for less global memory traffic 

        real delta_R = delta_avg_y + sqrt(sigma_sqr_y)*curand_normal_double(&rngs_par);    // in polar R
        real delta_Z = delta_avg_z + sqrt(sigma_sqr_z)*curand_normal_double(&rngs_par);    // in polar Z

        dev_rngs_par[idx] = rngs_par;               // update the global state, otherwise, always the same number

        if (N_Z == 1)
        {
            dev_particle[idx].position.y = bigR + delta_R;
        }
        else
        {
            dev_particle[idx].position.y = sqrt((bigR + delta_R)*(bigR + delta_R) + (bigZ + delta_Z)*(bigZ + delta_Z));
            dev_particle[idx].position.z = atan2(bigR + delta_R, bigZ + delta_Z);
        }
    }
}