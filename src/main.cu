#include <chrono>           // for std::chrono::system_clock
#include <iomanip>          // for std::setw, std::setfill
#include <iostream>         // for std::cout, std::endl
#include <sstream>          // for std::stringstream
#include <sys/stat.h>       // for mkdir

#ifdef COLLISION
#include <thrust/device_ptr.h>  // for thrust::device_ptr
#include <thrust/extrema.h>     // for thrust::max_element
#endif // COLLISION

#include "cudust_kern.cuh"
#include "cudust_host.cuh"

std::mt19937 rand_generator;

// =========================================================================================================================

int main (int argc, char **argv)
{
    int idx_resume;
    real clock_sim = 0.0;   // total simulation time
    real clock_out = 0.0;   // time accumulated toward the next output

    #ifdef TRANSPORT
    int count_dyn;          // how many dynamics timesteps in one output timestep
    real dt_dyn;            // dynamical timestep
    #endif // TRANSPORT
    
    #ifdef COLLISION
    int count_col;          // how many collision calculations in one dynamics timestep
    real clock_dyn = 0.0;   // time accumulated toward the next dynamics timestep
    real dt_col;            // collisional timestep
    #endif // COLLISION
    
    std::string fname;
    std::uniform_real_distribution <real> random(0.0, 1.0); // distribution in [0, 1)

    swarm *particle, *dev_particle;
    cudaMallocHost((void**)&particle, sizeof(swarm)*N_P);
    cudaMalloc((void**)&dev_particle, sizeof(swarm)*N_P);
    
    real *dustdens, *dev_dustdens;
    cudaMallocHost((void**)&dustdens, sizeof(real)*N_G);
    cudaMalloc((void**)&dev_dustdens, sizeof(real)*N_G);

    #if defined(TRANSPORT) && defined(RADIATION)
    real *optdepth, *dev_optdepth;
    cudaMallocHost((void**)&optdepth, sizeof(real)*N_G);
    cudaMalloc((void**)&dev_optdepth, sizeof(real)*N_G);
    #endif // RADIATION

    #ifdef COLLISION
    real max_rate; // Simple variable for max collision rate (thrust approach)

    bbox *dev_boundbox;
    cudaMalloc((void**)&dev_boundbox, sizeof(bbox));

    tree *dev_col_tree;
    cudaMalloc((void**)&dev_col_tree, sizeof(tree)*N_P);

    curs *dev_rs_grids;
    cudaMalloc((void**)&dev_rs_grids, sizeof(curs)*N_G);
    
    real *dev_col_rand, *dev_col_expt;
    cudaMalloc((void**)&dev_col_rand, sizeof(real)*N_G);
    cudaMalloc((void**)&dev_col_expt, sizeof(real)*N_G);

    real *dev_col_rate;
    cudaMalloc((void**)&dev_col_rate, sizeof(real)*N_G);

    int  *dev_col_flag;
    cudaMalloc((void**)&dev_col_flag, sizeof(int) *N_G);
    #endif // COLLISION

    #if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
    curs *dev_rs_swarm;
    cudaMalloc((void**)&dev_rs_swarm, sizeof(curs)*N_P);
    #endif // COLLISION or DIFFUSION

    if (argc <= 1) // no flag, fresh start
	{
        idx_resume = 0;

        real *random_x, *dev_random_x;
        real *random_y, *dev_random_y;
        real *random_z, *dev_random_z;
        real *random_s, *dev_random_s;
        
        cudaMallocHost((void**)&random_x, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_x, sizeof(real)*N_P);
        cudaMallocHost((void**)&random_y, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_y, sizeof(real)*N_P);
        cudaMallocHost((void**)&random_z, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_z, sizeof(real)*N_P);
        cudaMallocHost((void**)&random_s, sizeof(real)*N_P); cudaMalloc((void**)&dev_random_s, sizeof(real)*N_P);

        rand_generator.seed(0); // or use rand_generator.seed(std::time(NULL));

        real idx_rho_g = IDX_P - 0.5*IDX_Q - 1.5;   // volumetric gas density power-law index
        real idx_swarm = -0.5;                      // all swarms have an equal mass

        #if defined(TRANSPORT) && defined(RADIATION)
        // all swarms have an equal surface area, see _get_grain_number in initialize.cu
        idx_swarm = -1.5; 
        #endif // TRANSPORT && RADIATION

        rand_uniform(random_x, N_P, INIT_XMIN, INIT_XMAX);
        // rand_convpow(random_y, N_P, INIT_YMIN, INIT_YMAX, idx_rho_g, 0.05*R_0, N_Y);
        rand_uniform(random_y, N_P, INIT_YMIN, INIT_YMAX);
        rand_uniform(random_z, N_P, INIT_ZMIN, INIT_ZMAX);
        rand_pow_law(random_s, N_P, INIT_SMIN, INIT_SMAX, idx_swarm);
        // rand_4_linear(random_s, N_P);

        cudaMemcpy(dev_random_x, random_x, sizeof(real)*N_P, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_y, random_y, sizeof(real)*N_P, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_z, random_z, sizeof(real)*N_P, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_s, random_s, sizeof(real)*N_P, cudaMemcpyHostToDevice);

        particle_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_random_x, dev_random_y, dev_random_z, dev_random_s);

        cudaFreeHost(random_x); cudaFree(dev_random_x);
        cudaFreeHost(random_y); cudaFree(dev_random_y);
        cudaFreeHost(random_z); cudaFree(dev_random_z);
        cudaFreeHost(random_s); cudaFree(dev_random_s);
        
        #if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
        rs_swarm_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_rs_swarm);
        #endif // COLLISION or DIFFUSION
        
        #ifdef COLLISION
        rs_grids_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_rs_grids);
        #endif // COLLISION

        mkdir(PATH_OUT.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        save_variable(PATH_OUT + "variables.txt");

        #if defined(TRANSPORT) && defined(RADIATION)
        optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);                   // set all optical depth data to zero
        optdepth_scat <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);     // add particle contribution to cells
        optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);                   // calculate the optical thickness of each cell
        optdepth_csum <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);                   // integrate in Y direction to get optical depth
        optdepth_mean <<< NB_X, THREADS_PER_BLOCK >>> (dev_optdepth);                   // do azimuthal averaging for initial condition
        cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_G, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "optdepth_" + frame_num(idx_resume) + ".dat";
        save_binary(fname, optdepth, N_G);
        #endif // RADIATION

        #ifdef SAVE_DENS
        dustdens_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);                   // this process cannot be merged with the above
        dustdens_scat <<< NB_P, THREADS_PER_BLOCK >>> (dev_dustdens, dev_particle);     // because the weight in density and that in optical thickness
        dustdens_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);                   // of each particle may differ
        cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_G, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "dustdens_" + frame_num(idx_resume) + ".dat";
        save_binary(fname, dustdens, N_G);
        #endif // SAVE_DENS

        cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_P, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "particle_" + frame_num(idx_resume) + ".dat";
        save_binary(fname, particle, N_P);

        msg_output(0);
    }
    else
    {
        std::stringstream convert{argv[1]}; // read resume file number from input argument
        if (!(convert >> idx_resume)) // set idx_resume by input argument, exit if failed
        {
            std::cerr << "Error: Invalid resume file number: " << argv[1] << std::endl;
            return 1;
        }

        fname = PATH_OUT + "particle_" + frame_num(idx_resume) + ".dat";
        if (!load_binary(fname, particle, N_P)) // read particle data from file, exit if failed
        {
            std::cerr << "Error: Failed to load file: " << fname << std::endl;
            return 1;
        }
        cudaMemcpy(dev_particle, particle, sizeof(swarm)*N_P,  cudaMemcpyHostToDevice);

        #if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
        rs_swarm_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_rs_swarm);
        #endif // COLLISION or DIFFUSION
        
        #ifdef COLLISION
        rs_grids_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_rs_grids);
        #endif // COLLISION

        msg_output(idx_resume);
    }

    #if defined(COLLISION) && !defined(TRANSPORT) // only need to build KD-tree once and can use it forever
    col_tree_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_col_tree, dev_particle);
    cukd::buildTree <tree, tree_traits> (dev_col_tree, N_P, dev_boundbox);
    #endif // COLLISION but NO TRANSPORT

    #if !defined(TRANSPORT) && !defined(COLLISION)
    std::cerr << "Error: No evolution module is enabled. Please enable TRANSPORT and/or COLLISION." << std::endl;
    return 1;
    #endif // NO TRANSPORT and NO COLLISION

    for (int idx_file = 1 + idx_resume; idx_file <= SAVE_MAX; idx_file++) // main evolution loop
    {
        #ifdef LOGOUTPUT
        real dt_out = DT_OUT*pow(static_cast<real>(LOG_BASE), static_cast<real>(idx_file - 1));
        #else  // LINEAR
        real dt_out = DT_OUT;
        #endif // LOGOUTPUT
        
        clock_out = 0.0; // reset output clock
        
        #ifdef TRANSPORT
        count_dyn = 0; // reset dynamical step counter
        #endif // TRANSPORT

        PRINT_TITLE_TO_SCREEN();
        
        do // begin of while (clock_out < dt_out)
        {
            #ifdef TRANSPORT
            dt_dyn = fmin(DT_DYN, dt_out - clock_out);
            
            if (dt_dyn < DT_MIN)
            {
                break; // jump to file save if dt_dyn is too small
            }

            #ifdef RADIATION
            ssa_substep_1 <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dt_dyn);
            optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
            optdepth_scat <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);
            optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
            optdepth_csum <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);
            ssa_substep_2 <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_optdepth, dt_dyn);
            #else  // NO RADIATION
            ssa_transport <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dt_dyn);
            #endif // RADIATION
            
            #ifdef DIFFUSION
            diffusion_pos <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_rs_swarm, dt_dyn);
            #endif // DIFFUSION

            cudaDeviceSynchronize();

            #ifndef COLLISION // only update simulation clock here if NO COLLISION
            clock_sim += dt_dyn;
            #endif // NO COLLISION

            clock_out += dt_dyn;
            count_dyn ++;

            PRINT_VALUE_TO_SCREEN();
            #endif // TRANSPORT
            
            #ifdef COLLISION
            count_col = 0;

            #ifdef TRANSPORT
            clock_dyn = 0.0;
            
            // need to rebuild KD-tree for every dynamical timestep
            col_tree_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_col_tree, dev_particle);
            cukd::buildTree <tree, tree_traits> (dev_col_tree, N_P, dev_boundbox);
            #endif // TRANSPORT
        
            do // begin of while (clock_dyn < dt_dyn) or while (clock_out < dt_out)
            {
                col_rate_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_col_rate, dev_col_expt, dev_col_rand);
                col_rate_calc <<< NB_P, THREADS_PER_BLOCK >>> (dev_col_rate, dev_particle, dev_col_tree, dev_boundbox);
                
                // use thrust to find maximum collision rate on device
                thrust::device_ptr <const real> dev_ptr(dev_col_rate);
                thrust::device_ptr <const real> max_ptr = thrust::max_element(dev_ptr, dev_ptr + N_G);
                max_rate = *max_ptr; // dereference triggers single-value cudaMemcpy from device to host
                
                // Compute collisional timestep
                dt_col = 1.0 / max_rate;

                if (dt_col < DT_MIN)
                {
                    std::cerr << "Error: Minimum collisional timestep reached due to high collision rate." << std::endl;
                    return 1;
                }
                
                #ifdef TRANSPORT
                dt_col = fmin(dt_col, dt_dyn - clock_dyn);
                #else  // NO TRANSPORT
                dt_col = fmin(dt_col, dt_out - clock_out);
                #endif // TRANSPORT

                if (dt_col < DT_MIN)
                {
                    break; // jump to transport, or file save, if dt_col is too small
                }

                col_flag_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_col_flag, dev_rs_grids, dev_col_rand, dev_col_rate, dt_col);
                col_proc_exec <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_rs_swarm, dev_col_expt, dev_col_rand, dev_col_flag, dev_col_tree, dev_boundbox);

                cudaDeviceSynchronize();

                #ifdef TRANSPORT
                clock_dyn += dt_col;
                #else  // NO TRANSPORT
                clock_out += dt_col;
                #endif // TRANSPORT
                
                clock_sim += dt_col;
                count_col ++;

                PRINT_VALUE_TO_SCREEN();
            }
            #ifdef TRANSPORT
            while (clock_dyn < dt_dyn);
            #else  // NO TRANSPORT
            while (clock_out < dt_out);
            #endif // TRANSPORT
            #endif // COLLISION

            // prevent infinite loop when COLLISION but NO TRANSPORT
            #if defined(COLLISION) && !defined(TRANSPORT)
            if (dt_out - clock_out < DT_MIN) 
            {
                break; // jump to file save if remaining time is too small
            }
            #endif // COLLISION but NO TRANSPORT

        } while (clock_out < dt_out);

        #if defined(TRANSPORT) && defined(RADIATION)
        optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_scat <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);
        optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_csum <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);
        cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_G, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "optdepth_" + frame_num(idx_file) + ".dat";
        save_binary(fname, optdepth, N_G);
        #endif // RADIATION
    
        #ifdef SAVE_DENS
        dustdens_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);
        dustdens_scat <<< NB_P, THREADS_PER_BLOCK >>> (dev_dustdens, dev_particle);
        dustdens_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);
        cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_G, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "dustdens_" + frame_num(idx_file) + ".dat";
        save_binary(fname, dustdens, N_G);
        #endif // SAVE_DENS

        cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_P, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "particle_" + frame_num(idx_file) + ".dat";
        save_binary(fname, particle, N_P);

        msg_output(idx_file);
    }
 
    return 0;
}

// =========================================================================================================================
