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
    real output_timer = 0.0;
    
    std::string fname;
    std::ofstream ofile;
    std::uniform_real_distribution <real> random(0.0, 1.0); // distribution in [0, 1)

    swarm *particle, *dev_particle;
    cudaMallocHost((void**)&particle, sizeof(swarm)*NUM_PAR);
    cudaMalloc((void**)&dev_particle, sizeof(swarm)*NUM_PAR);
    
    real *dustdens, *dev_dustdens;
    cudaMallocHost((void**)&dustdens, sizeof(real)*NUM_DIM);
    cudaMalloc((void**)&dev_dustdens, sizeof(real)*NUM_DIM);

    real *optdepth, *dev_optdepth;
    cudaMallocHost((void**)&optdepth, sizeof(real)*NUM_DIM);
    cudaMalloc((void**)&dev_optdepth, sizeof(real)*NUM_DIM);

    real *dev_col_rand, *dev_col_real;
    cudaMalloc((void**)&dev_col_rand, sizeof(real)*NUM_DIM);
    cudaMalloc((void**)&dev_col_real, sizeof(real)*NUM_DIM);

    real *dev_col_rate;
    cudaMalloc((void**)&dev_col_rate, sizeof(real)*NUM_DIM);

    tree *dev_treenode;
    cudaMalloc((void**)&dev_treenode, sizeof(tree)*NUM_PAR);

    int *dev_col_flag;
    cudaMalloc((void**)&dev_col_flag, sizeof(int)*NUM_DIM);

    float *max_rate, *dev_max_rate;
    cudaMallocHost((void**)&max_rate, sizeof(float));
    cudaMalloc((void**)&dev_max_rate, sizeof(float));

    real *timestep, *dev_timestep;
    cudaMallocHost((void**)&timestep, sizeof(real));
    cudaMalloc((void**)&dev_timestep, sizeof(real));

    boxf *dev_boundbox;
    cudaMalloc((void**)&dev_boundbox, sizeof(boxf));

    curandState *dev_rngs_par, *dev_rngs_dim;
    cudaMalloc((void**)&dev_rngs_par, sizeof(curandState)*NUM_PAR);
    cudaMalloc((void**)&dev_rngs_dim, sizeof(curandState)*NUM_DIM);

    if (argc <= 1) // no flag, start from the initial condition
	{
        resume = 0;

        real *prof_azi, *dev_prof_azi;
        real *prof_rad, *dev_prof_rad;
        real *prof_col, *dev_prof_col;
        real *prof_siz, *dev_prof_siz;
        cudaMallocHost((void**)&prof_azi, sizeof(real)*NUM_PAR);
        cudaMalloc((void**)&dev_prof_azi, sizeof(real)*NUM_PAR);
        cudaMallocHost((void**)&prof_rad, sizeof(real)*NUM_PAR);
        cudaMalloc((void**)&dev_prof_rad, sizeof(real)*NUM_PAR);
        cudaMallocHost((void**)&prof_col, sizeof(real)*NUM_PAR);
        cudaMalloc((void**)&dev_prof_col, sizeof(real)*NUM_PAR);
        cudaMallocHost((void**)&prof_siz, sizeof(real)*NUM_PAR);
        cudaMalloc((void**)&dev_prof_siz, sizeof(real)*NUM_PAR);

        rand_generator.seed(0); // or use rand_generator.seed(std::time(NULL));

        rand_uniform  (prof_azi, NUM_PAR, AZI_INIT_MIN,  AZI_INIT_MAX);
        rand_conv_pow (prof_rad, NUM_PAR, RAD_INIT_MIN,  RAD_INIT_MAX, IDX_SIGMAG - 1.0, SMOOTH_RAD, RES_RAD);
        rand_uniform  (prof_col, NUM_PAR, COL_INIT_MIN,  COL_INIT_MAX);
        rand_powerlaw (prof_siz, NUM_PAR, SIZE_INIT_MIN, SIZE_INIT_MAX, -1.5); // pow_idx = -1.5 is explained in init.cu

        cudaMemcpy(dev_prof_azi, prof_azi, sizeof(real)*NUM_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_prof_rad, prof_rad, sizeof(real)*NUM_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_prof_col, prof_col, sizeof(real)*NUM_PAR, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_prof_siz, prof_siz, sizeof(real)*NUM_PAR, cudaMemcpyHostToDevice);

        particle_init <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_prof_azi, dev_prof_rad, dev_prof_col, dev_prof_siz);

        cudaFreeHost(prof_azi); cudaFree(dev_prof_azi);
        cudaFreeHost(prof_rad); cudaFree(dev_prof_rad);
        cudaFreeHost(prof_col); cudaFree(dev_prof_col);
        cudaFreeHost(prof_siz); cudaFree(dev_prof_siz);

        // auto start = std::chrono::system_clock::now();
        // cudaDeviceSynchronize();
        // auto end = std::chrono::system_clock::now();
        // std::chrono::duration<double> elapsed_seconds = end - start;
        // std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

        optdepth_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
        optdepth_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_optdepth, dev_particle);
        optdepth_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
        optdepth_inte <<<BLOCKNUM_RAD, THREADS_PER_BLOCK>>> (dev_optdepth);
        optdepth_mean <<<BLOCKNUM_AZI, THREADS_PER_BLOCK>>> (dev_optdepth);
        
        dustdens_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_dustdens);
        dustdens_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_dustdens, dev_particle);
        dustdens_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_dustdens);
        
        velocity_init <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_optdepth);
        rngs_par_init <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_rngs_par);
        rngs_dim_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_rngs_dim);

        mkdir(OUTPUT_PATH.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        open_txt_file(ofile, OUTPUT_PATH + "variables.txt");
        save_variable(ofile);

        // cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*NUM_DIM, cudaMemcpyDeviceToHost);
        // fname = OUTPUT_PATH + "dustdens_" + frame_num(resume) + ".bin";
        // open_bin_file(ofile, fname);
        // save_bin_file(ofile, dustdens, NUM_DIM);

        // cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*NUM_DIM, cudaMemcpyDeviceToHost);
        // fname = OUTPUT_PATH + "optdepth_" + frame_num(resume) + ".bin";
        // open_bin_file(ofile, fname);
        // save_bin_file(ofile, optdepth, NUM_DIM);

        cudaMemcpy(particle, dev_particle, sizeof(swarm)*NUM_PAR, cudaMemcpyDeviceToHost);
        fname = OUTPUT_PATH + "particle_" + frame_num(resume) + ".par";
        open_bin_file(ofile, fname);
        save_bin_file(ofile, particle, NUM_PAR);

        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::setw(3) << std::setfill('0') << 0 << "/" << std::setw(3) << std::setfill('0') << OUTPUT_NUM << " finished on " << std::ctime(&end_time);
    }
    else
    {
        std::stringstream convert{argv[1]};     // set up a stringstream variable named convert, initialized with the input from argv[1]
        if (!(convert >> resume)) resume = -1;  // do the conversion, if conversion fails, set resume to a default value

        std::ifstream ifile;
        fname = OUTPUT_PATH + "particle_" + frame_num(resume) + ".bin";
        load_bin_file(ifile, fname);
        read_bin_file(ifile, particle, NUM_PAR);
        cudaMemcpy(dev_particle, particle, sizeof(swarm)*NUM_PAR,  cudaMemcpyHostToDevice);

        optdepth_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
        optdepth_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_optdepth, dev_particle);
        optdepth_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
        optdepth_inte <<<BLOCKNUM_RAD, THREADS_PER_BLOCK>>> (dev_optdepth);

        rngs_par_init <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_rngs_par);
        rngs_dim_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_rngs_dim);

        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::setw(3) << std::setfill('0') << resume << "/" << std::setw(3) << std::setfill('0') << OUTPUT_NUM << " finished on " << std::ctime(&end_time);
    }

    // for test use
    treenode_init <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_treenode);
    cukd::buildTree <tree, tree_traits> (dev_treenode, NUM_PAR, dev_boundbox);

    for (int i = 1 + resume; i <= OUTPUT_NUM; i++)
    {
        while (output_timer < OUTPUT_INT)
        {
            // treenode_init <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_treenode);
            // cukd::buildTree <tree, tree_traits> (dev_treenode, NUM_PAR, dev_boundbox);                                                  // 250 ms!!
            
            col_rate_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_col_rate, dev_col_rand, dev_col_real, dev_max_rate);
            col_rate_calc <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_treenode, dev_col_rate, dev_boundbox);
            col_rate_peak <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_col_rate, dev_max_rate);
            
            cudaMemcpy(max_rate, dev_max_rate, sizeof(float), cudaMemcpyDeviceToHost);
            *timestep = 1.0 / static_cast<real>(*max_rate);
            // *timestep = -std::log(1.0 - random(rand_generator)) / static_cast<real>(*max_rate);
            // if (*timestep > DT_MAX) *timestep = DT_MAX;
            if (*timestep > OUTPUT_INT - output_timer) *timestep = OUTPUT_INT - output_timer;
            cudaMemcpy(dev_timestep, timestep, sizeof(real), cudaMemcpyHostToDevice);

            col_flag_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_col_rate, dev_col_rand, dev_col_flag, dev_timestep, dev_rngs_dim);
            par_evol_calc <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_treenode, dev_col_flag, dev_col_rand, dev_col_real, dev_boundbox, dev_rngs_par);

            // ssa_substep_1 <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_timestep);
            // optdepth_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
            // optdepth_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_optdepth, dev_particle);
            // optdepth_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
            // optdepth_inte <<<BLOCKNUM_RAD, THREADS_PER_BLOCK>>> (dev_optdepth);
            // ssa_substep_2 <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_particle, dev_optdepth, dev_timestep);

            timer        += *timestep;
            output_timer += *timestep;

            std::cout << std::setprecision(6) << std::scientific << *timestep << ' ' << output_timer << ' ' << timer << std::endl;
        }
    
        output_timer = 0.0;
    
        // // calculate dustdens grids for each output
        // dustdens_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_dustdens);
        // dustdens_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_dustdens, dev_particle);
        // dustdens_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_dustdens);

        // cudaMemcpy(dustdens, dev_dustdens, sizeof(real)*NUM_DIM, cudaMemcpyDeviceToHost);
        // fname = OUTPUT_PATH + "dustdens_" + frame_num(i) + ".bin";
        // open_bin_file(ofile, fname);
        // save_bin_file(ofile, dustdens, NUM_DIM);

        // // calculate optical depth grids for each output
        // optdepth_init <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
        // optdepth_enum <<<BLOCKNUM_PAR, THREADS_PER_BLOCK>>> (dev_optdepth, dev_particle);
        // optdepth_calc <<<BLOCKNUM_DIM, THREADS_PER_BLOCK>>> (dev_optdepth);
        // optdepth_inte <<<BLOCKNUM_RAD, THREADS_PER_BLOCK>>> (dev_optdepth);

        // cudaMemcpy(optdepth, dev_optdepth, sizeof(real)*NUM_DIM, cudaMemcpyDeviceToHost);
        // fname = OUTPUT_PATH + "optdepth_" + frame_num(i) + ".bin";
        // open_bin_file(ofile, fname);
        // save_bin_file(ofile, optdepth, NUM_DIM);

        if (i % OUTPUT_PAR == 0)
        {
            cudaMemcpy(particle, dev_particle, sizeof(swarm)*NUM_PAR, cudaMemcpyDeviceToHost);
            fname = OUTPUT_PATH + "particle_" + frame_num(i) + ".par";
            open_bin_file(ofile, fname);
            save_bin_file(ofile, particle, NUM_PAR);
        }

        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::setw(3) << std::setfill('0') << i << "/" << std::setw(3) << std::setfill('0') << OUTPUT_NUM << " finished on " << std::ctime(&end_time);
    }
 
    return 0;
}
