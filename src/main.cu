#include <ctime>            // for std::time_t, std::time, std::ctime
#include <chrono>           // for std::chrono::system_clock
#include <iomanip>          // for std::setw, std::setfill
#include <sstream>          // for std::stringstream
#include <iostream>         // for std::cout, std::endl
#include <sys/stat.h>       // for mkdir

#include "cudust.cuh"
#include "curand_kernel.h"

std::mt19937 rand_generator;

int main (int argc, char **argv)
{
    int resume;
    int count_col; // how many collision calculations in one dynamics timestep
    int count_dyn; // how many dynamics timesteps in one output timestep
    
    real timer_sim = 0.0;
    real timer_dyn = 0.0;
    real timer_out = 0.0;
    
    real dt_dyn, dt_col;
    
    std::string fname;
    std::uniform_real_distribution <real> random(0.0, 1.0); // distribution in [0, 1)

    swarm *particle, *dev_particle;
    cudaMallocHost((void**)&particle, sizeof(swarm)*N_PAR);
    cudaMalloc((void**)&dev_particle, sizeof(swarm)*N_PAR);
    
    real *dustdens, *dev_dustdens;
    cudaMallocHost((void**)&dustdens, sizeof(real)*N_GRD);
    cudaMalloc((void**)&dev_dustdens, sizeof(real)*N_GRD);

    real *optdepth, *dev_optdepth;
    cudaMallocHost((void**)&optdepth, sizeof(real)*N_GRD);
    cudaMalloc((void**)&dev_optdepth, sizeof(real)*N_GRD);

    real *dev_col_rand, *dev_col_real;
    cudaMalloc((void**)&dev_col_rand, sizeof(real)*N_GRD);
    cudaMalloc((void**)&dev_col_real, sizeof(real)*N_GRD);

    real *dev_col_rate;
    cudaMalloc((void**)&dev_col_rate, sizeof(real)*N_GRD);

    tree *dev_treenode;
    cudaMalloc((void**)&dev_treenode, sizeof(tree)*N_PAR);

    int *dev_col_flag;
    cudaMalloc((void**)&dev_col_flag, sizeof(int)*N_GRD);

    float *max_rate, *dev_max_rate;
    cudaMallocHost((void**)&max_rate, sizeof(float));
    cudaMalloc((void**)&dev_max_rate, sizeof(float));

    boxf *dev_boundbox;
    cudaMalloc((void**)&dev_boundbox, sizeof(boxf));

    curandState *dev_rngs_par, *dev_rngs_grd;
    cudaMalloc((void**)&dev_rngs_par, sizeof(curandState)*N_PAR);
    cudaMalloc((void**)&dev_rngs_grd, sizeof(curandState)*N_GRD);

    // auto start = std::chrono::system_clock::now();
    // cudaDeviceSynchronize();
    // auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    if (argc <= 1) // no flag, fresh start
	{
        resume = 0;

        // for initialization
        real *random_x, *dev_random_x;
        real *random_y, *dev_random_y;
        real *random_z, *dev_random_z;
        real *random_s, *dev_random_s;
        
        cudaMallocHost((void**)&random_x, sizeof(real)*N_PAR); cudaMalloc((void**)&dev_random_x, sizeof(real)*N_PAR);
        cudaMallocHost((void**)&random_y, sizeof(real)*N_PAR); cudaMalloc((void**)&dev_random_y, sizeof(real)*N_PAR);
        cudaMallocHost((void**)&random_z, sizeof(real)*N_PAR); cudaMalloc((void**)&dev_random_z, sizeof(real)*N_PAR);
        cudaMallocHost((void**)&random_s, sizeof(real)*N_PAR); cudaMalloc((void**)&dev_random_s, sizeof(real)*N_PAR);

        rand_generator.seed(0); // or use rand_generator.seed(std::time(NULL));

        rand_uniform(random_x, N_PAR, INIT_XMIN, INIT_XMAX);
        rand_convpow(random_y, N_PAR, INIT_YMIN, INIT_YMAX, IDX_SURF - 1.0, 0.05*R_0, N_Y);
        rand_uniform(random_z, N_PAR, INIT_ZMIN, INIT_ZMAX);
        rand_pow_law(random_s, N_PAR, INIT_SMIN, INIT_SMAX, -1.5); // pow_idx is explained in init.cu

        cudaMemcpy(dev_random_x, random_x, sizeof(real)*N_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_y, random_y, sizeof(real)*N_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_z, random_z, sizeof(real)*N_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_random_s, random_s, sizeof(real)*N_PAR, cudaMemcpyHostToDevice);

        particle_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_random_x, dev_random_y, dev_random_z, dev_random_s);

        cudaFreeHost(random_x); cudaFree(dev_random_x);
        cudaFreeHost(random_y); cudaFree(dev_random_y);
        cudaFreeHost(random_z); cudaFree(dev_random_z);
        cudaFreeHost(random_s); cudaFree(dev_random_s);

        optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);                   // set all optical depth data to zero
        optdepth_enum <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);     // add particle contribution to cells
        optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);                   // calculate the optical thickness of each cell
        optdepth_intg <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);                   // integrate in Y direction to get optical depth
        optdepth_mean <<< NB_X, THREADS_PER_BLOCK >>> (dev_optdepth);                   // do azimuthal averaging for initial condition
        
        dustdens_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);                   // this process cannot be merged with the above
        dustdens_enum <<< NB_P, THREADS_PER_BLOCK >>> (dev_dustdens, dev_particle);     // because the weight in density and that in optical thickness
        dustdens_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);                   // of each particle may differ
        
        rngs_par_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_rngs_par);
        rngs_grd_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_rngs_grd);

        mkdir(PATH_OUT.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        save_variable(PATH_OUT + "variables.txt");

        cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "dustdens_" + frame_num(resume) + ".dat";
        save_binary(fname, dustdens, N_GRD);

        cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "optdepth_" + frame_num(resume) + ".dat";
        save_binary(fname, optdepth, N_GRD);

        cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_PAR, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "particle_" + frame_num(resume) + ".dat";
        save_binary(fname, particle, N_PAR);

        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout   << std::endl
                    << std::setw(3) << std::setfill('0') << 0 << "/" 
                    << std::setw(3) << std::setfill('0') << SAVE_MAX << " finished on " << std::ctime(&end_time)
                    << std::endl;
    }
    else
    {
        std::stringstream convert{argv[1]};     // set up a stringstream variable named convert, initialized with the input from argv[1]
        if (!(convert >> resume)) resume = -1;  // do the conversion, if conversion fails, set resume to a default value

        fname = PATH_OUT + "particle_" + frame_num(resume) + ".dat";
        load_binary(fname, particle, N_PAR);
        cudaMemcpy(dev_particle, particle, sizeof(swarm)*N_PAR,  cudaMemcpyHostToDevice);

        optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_enum <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);
        optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_intg <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);

        rngs_par_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_rngs_par);
        rngs_grd_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_rngs_grd);

        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout   << std::endl 
                    << std::setfill('0')
                    << std::setw(3) << resume << "/"
                    << std::setw(3) << SAVE_MAX
                    << " finished on " << std::ctime(&end_time)
                    << std::endl;
    }

    for (int idx_file = 1 + resume; idx_file <= SAVE_MAX; idx_file++)    // main evolution loop
    {
        timer_out = 0.0;
        count_dyn = 0;
        
        while (timer_out < DT_OUT)                                          // evolve particle until one output timestep
        {
            timer_dyn = 0.0;
            count_col = 0;
            
            // Compute dynamics timestep: min of DT_DYN and remaining time to file save
            dt_dyn = fmin(DT_DYN, DT_OUT - timer_out);
            
            // Update tree for collision calculations
            // treenode_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_treenode);
            // cukd::buildTree <tree, tree_traits> (dev_treenode, N_PAR, dev_boundbox);
            
            std::cout   << std::setfill(' ')
                        << std::setw(10) << "idx_file"  << " "
                        << std::setw(10) << "count_col" << " "
                        << std::setw(10) << "count_dyn" << " "
                        << std::setw(10) << "dt_dyn"    << " "
                        << std::setw(10) << "timer_out" << " "
                        << std::setw(10) << "timer_sim" << std::endl;

            // Collision loop: evolve until timer_dyn reaches dt_dyn
            // while (timer_dyn < dt_dyn)
            // {
            //     col_rate_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_col_rate, dev_col_rand, dev_col_real, dev_max_rate);
            //     col_rate_calc <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_treenode, dev_col_rate, dev_boundbox);
            //     col_rate_peak <<< NB_A, THREADS_PER_BLOCK >>> (dev_col_rate, dev_max_rate);
                
            //     cudaMemcpy(max_rate, dev_max_rate, sizeof(float), cudaMemcpyDeviceToHost);
                
            //     // Compute collisional timestep
            //     dt_col = 1.0 / static_cast<real>(*max_rate);
                
            //     // Use minimum of all constraints
            //     dt_col = fmin(dt_col, dt_dyn);
            //     dt_col = fmin(dt_col, dt_dyn - timer_dyn);

            //     col_flag_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_col_rate, dev_col_rand, dev_col_flag, dev_rngs_grd, dt_col);
            //     particle_evol <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_treenode, dev_col_flag, dev_col_rand, 
            //                                                    dev_col_real, dev_boundbox, dev_rngs_par);

            //     cudaDeviceSynchronize();

            //     timer_dyn += dt_col;
            //     count_col ++;
            // }

            // Dynamics integration with dt_dyn timestep
            ssa_substep_1 <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dt_dyn);
            optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
            optdepth_enum <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);
            optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
            optdepth_intg <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);
            ssa_substep_2 <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_optdepth, dt_dyn);
            // pos_diffusion <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_rngs_par, dt_dyn);

            cudaDeviceSynchronize();

            timer_sim += dt_dyn;
            timer_out += dt_dyn;

            count_dyn ++;

            std::cout   << std::setfill(' ')
                        << std::setw(10) << idx_file    << " "
                        << std::setw(10) << count_col   << " " 
                        << std::setw(10) << count_dyn   << " "
                        << std::scientific << std::setprecision(3) 
                        << std::setw(10) << dt_dyn      << " "
                        << std::setw(10) << timer_out   << " "
                        << std::setw(10) << timer_sim   << std::endl;
        }
    
        // calculate dustdens grids for each output
        dustdens_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);
        dustdens_enum <<< NB_P, THREADS_PER_BLOCK >>> (dev_dustdens, dev_particle);
        dustdens_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);

        cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "dustdens_" + frame_num(idx_file) + ".dat";
        save_binary(fname, dustdens, N_GRD);

        // calculate optical depth grids for each output
        optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_enum <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);
        optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_intg <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);

        cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_OUT + "optdepth_" + frame_num(idx_file) + ".dat";
        save_binary(fname, optdepth, N_GRD);

        if (idx_file % SAVE_PAR == 0)
        {
            cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_PAR, cudaMemcpyDeviceToHost);
            fname = PATH_OUT + "particle_" + frame_num(idx_file) + ".dat";
            save_binary(fname, particle, N_PAR);
        }

        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout   << std::endl << std::setfill('0') 
                    << std::setw(3) << idx_file << "/" 
                    << std::setw(3) << SAVE_MAX 
                    << " finished on " << std::ctime(&end_time)
                    << std::endl;
    }
 
    return 0;
}
