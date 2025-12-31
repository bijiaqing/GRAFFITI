#include "cudust.cuh"

// =========================================================================================================================

__global__ 
void ssa_substep_1 (swarm *dev_particle, real *dev_timestep)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx >= 0 && idx < NUM_PAR)
    {
        real azi_i, l_azi_i, azi_1;
        real rad_i, v_rad_i, rad_1;
        real col_i, l_col_i, col_1;

        real dt = *dev_timestep;

        azi_i   = dev_particle[idx].position.x;
        rad_i   = dev_particle[idx].position.y;
        col_i   = dev_particle[idx].position.z;
        l_azi_i = dev_particle[idx].velocity.x;
        v_rad_i = dev_particle[idx].velocity.y;
        l_col_i = dev_particle[idx].velocity.z;

        rad_1 = rad_i + 0.5*v_rad_i*dt;
        col_1 = col_i + 0.5*l_col_i*dt / rad_i / rad_1;
        azi_1 = azi_i + 0.5*l_azi_i*dt / rad_i / rad_1 / sin(col_i) / sin(col_1);

        while (azi_1 >= AZI_MAX) azi_1 -= 2.0*M_PI;
        while (azi_1 <  AZI_MIN) azi_1 += 2.0*M_PI;

        if (rad_1 < RAD_MIN) 
        {
            rad_1 = RAD_MAX;
            dev_particle[idx].velocity.x = sqrt(G*M_STAR*RAD_MAX);
            dev_particle[idx].velocity.y = 0.0;
            dev_particle[idx].velocity.z = 0.0;
        }

        dev_particle[idx].position.x = azi_1;
        dev_particle[idx].position.y = rad_1;
        dev_particle[idx].position.z = col_1;
    }
}

// =========================================================================================================================

__global__
void ssa_substep_2 (swarm *dev_particle, real *dev_optdepth, real *dev_timestep)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx >= 0 && idx < NUM_PAR)
    {
        real l_azi_i, azi_1, l_azi_1, lg_azi_1, torq_azi_1, torq_azi_2, azi_j, l_azi_j, par_azi;
        real v_rad_i, rad_1,          vg_rad_1,             forc_rad_2, rad_j, v_rad_j, par_rad;
        real l_col_i, col_1, l_col_1, lg_col_1, torq_col_1, torq_col_2, col_j, l_col_j, par_col;
        real size, optdepth, ts_1, tau_1, eta_1, beta_1;

        azi_1   = dev_particle[idx].position.x;
        rad_1   = dev_particle[idx].position.y;
        col_1   = dev_particle[idx].position.z;
        l_azi_i = dev_particle[idx].velocity.x;
        v_rad_i = dev_particle[idx].velocity.y;
        l_col_i = dev_particle[idx].velocity.z;
        size    = dev_particle[idx].grain_size;

        real dt = *dev_timestep;

        real bigR_1 = rad_1*sin(col_1);
        real bigZ_1 = rad_1*cos(col_1);

        // get the velocity of gas in the hydrostatic equilibrium state
        eta_1 = (IDX_SIGMAG + IDX_TEMP - 1.0)*H_REF*H_REF*pow(bigR_1 / RAD_REF, IDX_TEMP + 1.0) + IDX_TEMP*(1.0 - bigR_1 / rad_1); 
        lg_azi_1 = sqrt(G*M_STAR*bigR_1)*sqrt(1.0 + eta_1);
        vg_rad_1 = 0.0;
        lg_col_1 = 0.0;

        // calculate the stopping time and the dimensionless time
        ts_1  = ST_REF*(size / SIZE_REF) / sqrt(G*M_STAR / RAD_REF / RAD_REF / RAD_REF);
        ts_1 *= pow(bigR_1 / RAD_REF, - 0.5*IDX_TEMP - IDX_SIGMAG + 1.0); // correct for radial gas density and sound speed
        ts_1 *= exp(bigZ_1*bigZ_1 / (2.0*H_REF*H_REF*bigR_1*bigR_1*pow(bigR_1 / RAD_REF, IDX_TEMP + 1.0))); // correct for vertical gas density
        tau_1 = dt / ts_1;

        // retrieve the optical depth of the particle based on its position and calculate beta
        par_azi  = static_cast<real>(RES_AZI)*   (azi_1 - AZI_MIN) /    (AZI_MAX - AZI_MIN);
        par_rad  = static_cast<real>(RES_RAD)*log(rad_1 / RAD_MIN) / log(RAD_MAX / RAD_MIN);
        par_col  = static_cast<real>(RES_COL)*   (col_1 - COL_MIN) /    (COL_MAX - COL_MIN);
        optdepth = _get_optdepth(dev_optdepth, par_azi, par_rad, par_col);
        beta_1   = 1.0 - BETA_REF*exp(-optdepth) / (size / SIZE_REF);

        // calculate the forces and torques (using the updated position but outdated velocity)
        torq_azi_1 =  0.0;
        // forc_rad_1 = -beta_1*G*M_STAR / rad_1 / rad_1;
        torq_col_1 =  0.0;

        // calculate the centrifugal forces (using the updated position but outdated velocity)
        // real ctfg_rad_1 = l_azi_i*l_azi_i / bigR_1 / bigR_1 / rad_1 + l_col_i*l_col_i / rad_1 / rad_1 / rad_1;
        real ctfg_col_1 = l_azi_i*l_azi_i / bigR_1 / bigR_1 / sin(col_1) * cos(col_1);

        // calculate the updated velocities
        l_azi_1 = l_azi_i + ((torq_azi_1             )*ts_1 + lg_azi_1 - l_azi_i)*(1.0 - exp(-0.5*tau_1));
        // v_rad_1 = v_rad_i + ((forc_rad_1 + ctfg_rad_1)*ts_1 + vg_rad_1 - v_rad_i)*(1.0 - exp(-0.5*tau_1));
        l_col_1 = l_col_i + ((torq_col_1 + ctfg_col_1)*ts_1 + lg_col_1 - l_col_i)*(1.0 - exp(-0.5*tau_1));

        // calculate the forces and torques (using the updated position and velocity)
        torq_azi_2 =  0.0;
        forc_rad_2 = -beta_1*G*M_STAR / rad_1 / rad_1;
        torq_col_2 =  0.0;

        // calculate the centrifugal forces (using the updated position and velocity)
        real ctfg_rad_2 = l_azi_1*l_azi_1 / bigR_1 / bigR_1 / rad_1 + l_col_1*l_col_1 / rad_1 / rad_1 / rad_1;
        real ctfg_col_2 = l_azi_1*l_azi_1 / bigR_1 / bigR_1 / sin(col_1) * cos(col_1);

        // calculate the next-step velocity
        l_azi_j = l_azi_i + ((torq_azi_2             )*ts_1 + lg_azi_1 - l_azi_i)*(1.0 - exp(-tau_1));
        v_rad_j = v_rad_i + ((forc_rad_2 + ctfg_rad_2)*ts_1 + vg_rad_1 - v_rad_i)*(1.0 - exp(-tau_1));
        l_col_j = l_col_i + ((torq_col_2 + ctfg_col_2)*ts_1 + lg_col_1 - l_col_i)*(1.0 - exp(-tau_1));

        // calculate the next-step position (the sequence matters!!)
        rad_j = rad_1 + 0.5*v_rad_j*dt;
        col_j = col_1 + 0.5*l_col_j*dt / rad_1 / rad_j;
        azi_j = azi_1 + 0.5*l_azi_j*dt / rad_1 / rad_j / sin(col_1) / sin(col_j);

        while (azi_j >= AZI_MAX) azi_j -= 2.0*M_PI;
        while (azi_j <  AZI_MIN) azi_j += 2.0*M_PI;

        if (rad_j < RAD_MIN) 
        {
            rad_j   = RAD_MAX;
            l_azi_j = sqrt(G*M_STAR*RAD_MAX);
            v_rad_j = 0.0;
            l_col_j = 0.0;
        }

        dev_particle[idx].position.x = azi_j;
        dev_particle[idx].position.y = rad_j;
        dev_particle[idx].position.z = col_j;

        dev_particle[idx].velocity.x = l_azi_j;
        dev_particle[idx].velocity.y = v_rad_j;
        dev_particle[idx].velocity.z = l_col_j;
    }
}
