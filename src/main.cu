#include <chrono>           // for std::chrono::system_clock
#include <iomanip>          // for std::setw, std::setfill
#include <sstream>          // for std::stringstream
#include <iostream>         // for std::cout, std::endl
#include <sys/stat.h>       // for mkdir

#ifdef COLLISION
#include <thrust/device_ptr.h>  // for thrust::device_ptr
#include <thrust/extrema.h>     // for thrust::max_element
#endif // COLLISION

#include "cudust.cuh"

std::mt19937 rand_generator;

// =========================================================================================================================

int main (int argc, char **argv)
{
    int idx_resume;
    real timer_sim = 0.0;   // total simulation time

    int count_dyn;          // how many dynamics timesteps in one output timestep
    real timer_out = 0.0;   // time accumulated toward the next output
    real dt_dyn;            // dynamical timestep

    #ifdef COLLISION
    int count_col;          // how many collision calculations in one dynamics timestep
    real timer_dyn = 0.0;   // time accumulated toward the next dynamics timestep
    real dt_col;            // collisional timestep
    #endif // COLLISION
    
    std::string fname;
    std::uniform_real_distribution <real> random(0.0, 1.0); // distribution in [0, 1)

    swarm *particle, *dev_particle;
    cudaMallocHost((void**)&particle, sizeof(swarm)*N_PAR);
    cudaMalloc((void**)&dev_particle, sizeof(swarm)*N_PAR);
    
    real *dustdens, *dev_dustdens;
    cudaMallocHost((void**)&dustdens, sizeof(real)*N_GRD);
    cudaMalloc((void**)&dev_dustdens, sizeof(real)*N_GRD);

    #ifdef RADIATION
    real *optdepth, *dev_optdepth;
    cudaMallocHost((void**)&optdepth, sizeof(real)*N_GRD);
    cudaMalloc((void**)&dev_optdepth, sizeof(real)*N_GRD);
    #endif // RADIATION

    #ifdef COLLISION
    real max_rate; // Simple variable for max collision rate (Thrust approach)

    bbox *dev_boundbox;
    cudaMalloc((void**)&dev_boundbox, sizeof(bbox));

    tree *dev_treenode;
    cudaMalloc((void**)&dev_treenode, sizeof(tree)*N_PAR);

    curs *dev_rs_grids;
    cudaMalloc((void**)&dev_rs_grids, sizeof(curs)*N_GRD);
    
    real *dev_col_rand, *dev_col_expt;
    cudaMalloc((void**)&dev_col_rand, sizeof(real)*N_GRD);
    cudaMalloc((void**)&dev_col_expt, sizeof(real)*N_GRD);

    real *dev_col_rate;
    cudaMalloc((void**)&dev_col_rate, sizeof(real)*N_GRD);

    int  *dev_col_flag;
    cudaMalloc((void**)&dev_col_flag, sizeof(int) *N_GRD);
    #endif // COLLISION

    #if defined(COLLISION) || defined(DIFFUSION)
    curs *dev_rs_swarm;
    cudaMalloc((void**)&dev_rs_swarm, sizeof(curs)*N_PAR);
    #endif // COLLISION or DIFFUSION

    // auto start = std::chrono::system_clock::now();
    // cudaDeviceSynchronize();
    // auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    if (argc <= 1) // no flag, fresh start
	{
        idx_resume = 0;

        real *random_x, *dev_random_x;
        real *random_y, *dev_random_y;
        real *random_z, *dev_random_z;
        real *random_s, *dev_random_s;
        
        cudaMallocHost((void**)&random_x, sizeof(real)*N_PAR); cudaMalloc((void**)&dev_random_x, sizeof(real)*N_PAR);
        cudaMallocHost((void**)&random_y, sizeof(real)*N_PAR); cudaMalloc((void**)&dev_random_y, sizeof(real)*N_PAR);
        cudaMallocHost((void**)&random_z, sizeof(real)*N_PAR); cudaMalloc((void**)&dev_random_z, sizeof(real)*N_PAR);
        cudaMallocHost((void**)&random_s, sizeof(real)*N_PAR); cudaMalloc((void**)&dev_random_s, sizeof(real)*N_PAR);

        rand_generator.seed(0); // or use rand_generator.seed(std::time(NULL));

        real idx_rho_g = IDX_P - 0.5*IDX_Q - 1.5;   // volumetric gas density power-law index
        real idx_swarm = -0.5;                      // all swarms have an equal mass

        #ifdef RADIATION // all swarms have an equal surface area, see _get_grain_number in initialize.cu
        idx_swarm = -1.5; 
        #endif // RADIATION

        rand_uniform(random_x, N_PAR, INIT_XMIN, INIT_XMAX);
        rand_convpow(random_y, N_PAR, INIT_YMIN, INIT_YMAX, idx_rho_g, 0.05*R_0, N_Y);
        rand_uniform(random_z, N_PAR, INIT_ZMIN, INIT_ZMAX);
        rand_pow_law(random_s, N_PAR, INIT_SMIN, INIT_SMAX, idx_swarm);

        cudaMemcpy(dev_random_x, random_x, sizeof(real)*N_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_y, random_y, sizeof(real)*N_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_z, random_z, sizeof(real)*N_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_s, random_s, sizeof(real)*N_PAR, cudaMemcpyHostToDevice);

        particle_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_random_x, dev_random_y, dev_random_z, dev_random_s);

        cudaFreeHost(random_x); cudaFree(dev_random_x);
        cudaFreeHost(random_y); cudaFree(dev_random_y);
        cudaFreeHost(random_z); cudaFree(dev_random_z);
        cudaFreeHost(random_s); cudaFree(dev_random_s);
        
        #if defined(COLLISION) || defined(DIFFUSION)
        rs_swarm_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_rs_swarm);
        #endif // COLLISION or DIFFUSION
        
        #ifdef COLLISION
        rs_grids_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_rs_grids);
        #endif // COLLISION

        mkdir(PATH_OUT.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        save_variable(PATH_OUT + "variables.txt");

        #ifdef RADIATION
        optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);                   // set all optical depth data to zero
        optdepth_scat <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);     // add particle contribution to cells
        optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);                   // calculate the optical thickness of each cell
        optdepth_csum <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);                   // integrate in Y direction to get optical depth
        optdepth_mean <<< NB_X, THREADS_PER_BLOCK >>> (dev_optdepth);                   // do azimuthal averaging for initial condition
        
        cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "optdepth_" + frame_num(idx_resume) + ".dat";
        save_binary(fname, optdepth, N_GRD);
        #endif // RADIATION

        dustdens_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);                   // this process cannot be merged with the above
        dustdens_scat <<< NB_P, THREADS_PER_BLOCK >>> (dev_dustdens, dev_particle);     // because the weight in density and that in optical thickness
        dustdens_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);                   // of each particle may differ

        cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "dustdens_" + frame_num(idx_resume) + ".dat";
        save_binary(fname, dustdens, N_GRD);

        cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_PAR, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "particle_" + frame_num(idx_resume) + ".dat";
        save_binary(fname, particle, N_PAR);

        log_output(0);
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
        if (!load_binary(fname, particle, N_PAR)) // read particle data from file, exit if failed
        {
            std::cerr << "Error: Failed to load file: " << fname << std::endl;
            return 1;
        }
        cudaMemcpy(dev_particle, particle, sizeof(swarm)*N_PAR,  cudaMemcpyHostToDevice);

        #if defined(COLLISION) || defined(DIFFUSION)
        rs_swarm_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_rs_swarm);
        #endif // COLLISION or DIFFUSION
        
        #ifdef COLLISION
        rs_grids_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_rs_grids);
        #endif // COLLISION

        log_output(idx_resume);
    }

    for (int idx_file = 1 + idx_resume; idx_file <= SAVE_MAX; idx_file++)    // main evolution loop
    {
        timer_out = 0.0;
        count_dyn = 0;

        std::cout << std::setfill(' ')
            << std::setw(10) << "idx_file"  << " "
            << std::setw(10) << "timer_sim" << " "
            << std::setw(10) << "timer_out" << " "
            << std::setw(10) << "count_dyn" << " "
            << std::setw(10) << "dt_dyn"    << " "
            #ifdef COLLISION
            << std::setw(10) << "timer_dyn" << " "
            << std::setw(10) << "count_col" << " "
            << std::setw(10) << "dt_col"    << " "
            #endif // COLLISION
            << std::endl;

        while (timer_out < DT_OUT) // evolve particle until one output timestep
        {
            // Compute dynamics timestep: min of DT_DYN and remaining time to file save
            dt_dyn = fmin(DT_DYN, DT_OUT - timer_out);
            
            #ifdef COLLISION
            timer_dyn = 0.0;
            count_col = 0;
            
            treenode_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_treenode, dev_particle);
            cukd::buildTree <tree, tree_traits> (dev_treenode, N_PAR, dev_boundbox);

            while (timer_dyn < dt_dyn)
            {
                col_rate_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_col_rate, dev_col_expt, dev_col_rand);
                col_rate_calc <<< NB_P, THREADS_PER_BLOCK >>> (dev_col_rate, dev_particle, dev_treenode, dev_boundbox);
                
                // use thrust to find maximum collision rate
                thrust::device_ptr <const real> dev_ptr (dev_col_rate);
                thrust::device_ptr <const real> max_ptr = thrust::max_element (dev_ptr, dev_ptr + N_GRD);
                max_rate = *max_ptr; // Dereference triggers single-value cudaMemcpy from device to host
                
                // Compute collisional timestep
                dt_col = 1.0 / max_rate;
                
                // Use minimum of all constraints
                dt_col = fmin(dt_col, dt_dyn);
                dt_col = fmin(dt_col, dt_dyn - timer_dyn);

                col_flag_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_col_flag, dev_rs_grids, dev_col_rand, dev_col_rate, dt_col);
                run_collision <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_rs_swarm, dev_col_expt, dev_col_rand, 
                    dev_col_flag, dev_treenode, dev_boundbox);

                cudaDeviceSynchronize();

                timer_dyn += dt_col;
                count_col ++;

                std::cout 
                    << std::setfill(' ')
                    << std::defaultfloat
                    << std::setw(10) << idx_file    << " "
                    << std::scientific << std::setprecision(3)
                    << std::setw(10) << timer_sim   << " "
                    << std::defaultfloat
                    << std::setw(10) << count_dyn   << " "
                    << std::scientific << std::setprecision(3)
                    << std::setw(10) << timer_out   << " "
                    << std::setw(10) << dt_dyn      << " "
                    << std::defaultfloat
                    << std::setw(10) << count_col   << " "
                    << std::scientific << std::setprecision(3)
                    << std::setw(10) << timer_dyn   << " "
                    << std::setw(10) << dt_col      << " "
                    << std::endl;
            }
            #endif // COLLISION

            #ifdef RADIATION
            ssa_substep_1 <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dt_dyn);
            optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
            optdepth_scat <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);
            optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
            optdepth_csum <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);
            ssa_substep_2 <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_optdepth, dt_dyn);
            #else
            ssa_integrate <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dt_dyn);
            #endif // RADIATION
            
            #ifdef DIFFUSION
            pos_diffusion <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_rs_swarm, dt_dyn);
            #endif // DIFFUSION

            cudaDeviceSynchronize();

            timer_sim += dt_dyn;
            timer_out += dt_dyn;

            count_dyn ++;

            std::cout 
                << std::setfill(' ')
                << std::defaultfloat
                << std::setw(10) << idx_file    << " "
                << std::scientific << std::setprecision(3)
                << std::setw(10) << timer_sim   << " "
                << std::defaultfloat
                << std::setw(10) << count_dyn   << " "
                << std::scientific << std::setprecision(3)
                << std::setw(10) << timer_out   << " "
                << std::setw(10) << dt_dyn      << " "
                #ifdef COLLISION
                << std::defaultfloat
                << std::setw(10) << count_col   << " "
                << std::scientific << std::setprecision(3)
                << std::setw(10) << timer_dyn   << " "
                << std::setw(10) << dt_col      << " "
                #endif // COLLISION
                << std::endl;
        }   
    
        // calculate dustdens grids for each output
        dustdens_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);
        dustdens_scat <<< NB_P, THREADS_PER_BLOCK >>> (dev_dustdens, dev_particle);
        dustdens_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);

        cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "dustdens_" + frame_num(idx_file) + ".dat";
        save_binary(fname, dustdens, N_GRD);

        #ifdef RADIATION
        optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_scat <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);
        optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_csum <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);

        cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "optdepth_" + frame_num(idx_file) + ".dat";
        save_binary(fname, optdepth, N_GRD);
        #endif // RADIATION

        if (idx_file % SAVE_PAR == 0)
        {
            cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_PAR, cudaMemcpyDeviceToHost);
            fname = PATH_OUT + "particle_" + frame_num(idx_file) + ".dat";
            save_binary(fname, particle, N_PAR);
        }

        log_output(idx_file);
    }
 
    return 0;
}
