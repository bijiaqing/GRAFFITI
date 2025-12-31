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
    
    real timer = 0.0;
    real dynamics_timer = 0.0;
    real filesave_timer = 0.0;
    
    std::string fname;
    std::ofstream ofile;
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

    real *timestep, *dev_timestep;
    cudaMallocHost((void**)&timestep, sizeof(real));
    cudaMalloc((void**)&dev_timestep, sizeof(real));

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

        velocity_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle);                   // initialize particle velocity after optical depth calculation
        
        dustdens_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);                   // this process cannot be merged with the above
        dustdens_enum <<< NB_P, THREADS_PER_BLOCK >>> (dev_dustdens, dev_particle);     // because the weight in density and that in optical thickness
        dustdens_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);                   // of each particle may differ
        
        rngs_par_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_rngs_par);
        rngs_grd_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_rngs_grd);

        mkdir(PATH_FILESAVE.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        open_txt_file(ofile, PATH_FILESAVE + "variables.txt");
        save_variable(ofile);

        cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_FILESAVE + "dustdens_" + frame_num(resume) + ".bin";
        open_bin_file(ofile, fname);
        save_bin_file(ofile, dustdens, N_GRD);

        cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_FILESAVE + "optdepth_" + frame_num(resume) + ".bin";
        open_bin_file(ofile, fname);
        save_bin_file(ofile, optdepth, N_GRD);

        cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_PAR, cudaMemcpyDeviceToHost);
        fname = PATH_FILESAVE + "particle_" + frame_num(resume) + ".par";
        open_bin_file(ofile, fname);
        save_bin_file(ofile, particle, N_PAR);

        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::setw(3) << std::setfill('0') << 0 << "/" << std::setw(3) << std::setfill('0') << FILENUM_MAX << " finished on " << std::ctime(&end_time);
    }
    else
    {
        std::stringstream convert{argv[1]};     // set up a stringstream variable named convert, initialized with the input from argv[1]
        if (!(convert >> resume)) resume = -1;  // do the conversion, if conversion fails, set resume to a default value

        std::ifstream ifile;
        fname = PATH_FILESAVE + "particle_" + frame_num(resume) + ".bin";
        load_bin_file(ifile, fname);
        read_bin_file(ifile, particle, N_PAR);
        cudaMemcpy(dev_particle, particle, sizeof(swarm)*N_PAR,  cudaMemcpyHostToDevice);

        optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_enum <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);
        optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_intg <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);

        rngs_par_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_rngs_par);
        rngs_grd_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_rngs_grd);

        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::setw(3) << std::setfill('0') << resume << "/" << std::setw(3) << std::setfill('0') << FILENUM_MAX << " finished on " << std::ctime(&end_time);
    }

    for (int idx_file = 1 + resume; idx_file <= FILENUM_MAX; idx_file++)    // main evolution loop
    {
        while (filesave_timer < DT_FILESAVE)                                // evolve particle dynamics until one output timestep
        {
            treenode_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_treenode);
            cukd::buildTree <tree, tree_traits> (dev_treenode, N_PAR, dev_boundbox);    // takes 250 ms!!
            
            while (dynamics_timer < DT_DYNAMICS)                            // evolve grain collision until one dynamical timestep
            {
                col_rate_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_col_rate, dev_col_rand, dev_col_real, dev_max_rate);
                col_rate_calc <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_treenode, dev_col_rate, dev_boundbox);
                col_rate_peak <<< NB_A, THREADS_PER_BLOCK >>> (dev_col_rate, dev_max_rate);
                
                cudaMemcpy(max_rate, dev_max_rate, sizeof(float), cudaMemcpyDeviceToHost);
                
                *timestep = 1.0 / static_cast<real>(*max_rate);
                
                if (*timestep > DT_DYNAMICS)                  *timestep = DT_DYNAMICS;
                if (*timestep > DT_DYNAMICS - dynamics_timer) *timestep = DT_DYNAMICS - dynamics_timer;
                if (*timestep > DT_FILESAVE - filesave_timer) *timestep = DT_FILESAVE - filesave_timer;

                cudaMemcpy(dev_timestep, timestep, sizeof(real), cudaMemcpyHostToDevice);

                col_flag_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_col_rate, dev_col_rand, dev_col_flag, dev_timestep, dev_rngs_grd);
                particle_evol <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_treenode, dev_col_flag, dev_col_rand, dev_col_real, dev_boundbox, dev_rngs_par);

                cudaDeviceSynchronize();  // Ensure collisions are processed before next iteration

                timer          += *timestep;
                dynamics_timer += *timestep;
                filesave_timer += *timestep;
            }

            dynamics_timer = 0.0;

            ssa_substep_1 <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle);
            optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
            optdepth_enum <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);
            optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
            optdepth_intg <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);
            ssa_substep_2 <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_optdepth);
            pos_diffusion <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_rngs_par);

            cudaDeviceSynchronize();  // Ensure positions updated before next iteration

            std::cout << std::setprecision(6) << std::scientific << *timestep << ' ' << filesave_timer << ' ' << timer << std::endl;
        }
    
        filesave_timer = 0.0;
    
        // calculate dustdens grids for each output
        dustdens_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);
        dustdens_enum <<< NB_P, THREADS_PER_BLOCK >>> (dev_dustdens, dev_particle);
        dustdens_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_dustdens);

        cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_FILESAVE + "dustdens_" + frame_num(idx_file) + ".bin";
        open_bin_file(ofile, fname); save_bin_file(ofile, dustdens, N_GRD);

        // calculate optical depth grids for each output
        optdepth_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_enum <<< NB_P, THREADS_PER_BLOCK >>> (dev_optdepth, dev_particle);
        optdepth_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_optdepth);
        optdepth_intg <<< NB_Y, THREADS_PER_BLOCK >>> (dev_optdepth);

        cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*N_GRD, cudaMemcpyDeviceToHost);
        fname = PATH_FILESAVE + "optdepth_" + frame_num(idx_file) + ".bin";
        open_bin_file(ofile, fname); save_bin_file(ofile, optdepth, N_GRD);

        if (idx_file % SWARM_EVERY == 0)
        {
            cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_PAR, cudaMemcpyDeviceToHost);
            fname = PATH_FILESAVE + "particle_" + frame_num(idx_file) + ".par";
            open_bin_file(ofile, fname); save_bin_file(ofile, particle, N_PAR);
        }

        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::setw(3) << std::setfill('0') << idx_file << "/" << std::setw(3) << std::setfill('0') << FILENUM_MAX << " finished on " << std::ctime(&end_time);
    }
 
    return 0;
}
