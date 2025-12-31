#include <ctime>    // for std::time_t, std::time, std::ctime
#include <chrono>   // for std::chrono::system_clock
#include <cmath>    // for M_PI
#include <random>   // for std::mt19937
#include <string>   // for std::string
#include <iomanip>
#include <iostream> // for std::cout, std::endl

#include "cukd/knn.h"
#include "cukd/builder.h"
#include "curand_kernel.h"

using real  = double;
using real2 = double2;
using real3 = double3;

std::mt19937 rand_generator;

const int N_PAR = 1e+07;
const int RES_X = 100;
const int RES_Y = 100;
const int RES_Z = 100;
const int KNN_SIZE = 25;

const real X_MIN = 0.0;
const real X_MAX = 1.0;
const real Y_MIN = 0.0;
const real Y_MAX = 1.0;
const real Z_MIN = 0.0;
const real Z_MAX = 1.0;

const int  FILENUM_MAX = 100;
const real DT_FILESAVE = 0.1;
const real DT_DYNAMICS = 1.0e-03;

const real M_D = 1.0e27;
// const real BOOST_Q = 0.25;
const real S_0 = 1.0;
const real LAMBDA_0 = 1.0e-19; // this is the lambda(qj, qk) in beutel & dullemond 2023
const real MAX_DIST = 0.1;

const int THREADS_PER_BLOCK = 32;
const int N_GRD = RES_X*RES_Y*RES_Z;
const int NB_P = N_PAR / THREADS_PER_BLOCK + 1;
const int NB_A = N_GRD / THREADS_PER_BLOCK + 1;

const std::string PATH_FILESAVE = "outputs/";

struct swarm
{
    real3 position; // x = azi, y = rad, z = col
    real  dustsize; // size of the individual grain
    real  numgrain; // number of grains in the swarm
    real  collrate; // total collision rate for particle i
};

struct tree
{
    float3 xyz;     // xyz position of a tree node (particle), no higher than single precision allowed
    int index_old;  // the index of partile before being shuffled by the tree builder
    int split_dim;  // an 1-byte param for the k-d tree build
};

struct tree_traits
{
    using point_t = float3;
    enum { has_explicit_dim = true };
    
    static inline __host__ __device__ const point_t &get_point (const tree &node) { return node.xyz; }
    static inline __host__ __device__ float get_coord (const tree &node, int dim) { return cukd::get_coord(node.xyz, dim); }
    static inline __host__ __device__ int get_dim (const tree &node) { return node.split_dim; }
    static inline __host__ __device__ void set_dim (tree &node, int dim) { node.split_dim = dim; }
};

__host__
void rand_uniform (real *profile, int number, real p_min, real p_max)
{
    std::uniform_real_distribution <real> random(0.0, 1.0);

    for (int i = 0; i < number; i++)
    {
        profile[i] = p_min + (p_max - p_min)*random(rand_generator);
    }
}

__global__ 
void particle_init (swarm *dev_particle, real *dev_profile_x, real *dev_profile_y, real *dev_profile_z)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        dev_particle[idx].position.x = dev_profile_x[idx];
        dev_particle[idx].position.y = dev_profile_y[idx];
        dev_particle[idx].position.z = dev_profile_z[idx];

        // int idx_z = static_cast<int>((idx) / 1.0e+5);
        // int idx_y = static_cast<int>((idx - 1.0e+5*idx_z) / 1.0e3);
        // int idx_x = static_cast<int>((idx - 1.0e+5*idx_z - 1.0e3*idx_y) / 1.0e1);
        // int idx_i = idx % 10;
        
        // dev_particle[idx].position.x = X_MIN + (idx_x + 0.1*idx_i + 0.05)*(X_MAX - X_MIN) / RES_X / RES_X;
        // dev_particle[idx].position.y = Y_MIN + (idx_y + 0.1*idx_i + 0.05)*(Y_MAX - Y_MIN) / RES_Y / RES_Y;
        // dev_particle[idx].position.z = Z_MIN + (idx_z + 0.1*idx_i + 0.05)*(Z_MAX - Z_MIN) / RES_Z / RES_Z;

        real size = S_0;

        dev_particle[idx].dustsize = size;
        dev_particle[idx].numgrain = M_D / N_PAR / size / size / size; // 1e27 * 1e-7 = 1e20
        dev_particle[idx].collrate = 0.0;
    }
}

__host__
std::string frame_num (int number, std::size_t length)
{
    std::string str = std::to_string(number);

    if (str.length() < length)
    {
        str.insert(0, length - str.length(), '0');
    }

    return str;
}

__host__
void open_bin_file (std::ofstream &bin_file, std::string file_name) 
{
    bin_file.open(file_name.c_str(), std::ios::out | std::ios::binary);
}

__host__
void save_bin_file (std::ofstream &bin_file, swarm *data, int number) 
{
    bin_file.write((char*)data, sizeof(swarm)*number);
    bin_file.close();
}

__global__
void treenode_init (swarm *dev_particle, tree *dev_treenode)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        dev_treenode[idx].xyz.x = static_cast<float>(dev_particle[idx].position.x);
        dev_treenode[idx].xyz.y = static_cast<float>(dev_particle[idx].position.y);
        dev_treenode[idx].xyz.z = static_cast<float>(dev_particle[idx].position.z);
        dev_treenode[idx].index_old = idx;
    }
}

__global__
void parstate_init (curandState *dev_rngs_par, unsigned long long seed)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        curand_init(seed, idx, 0, &dev_rngs_par[idx]);
    }
}

__global__
void grdstate_init (curandState *dev_rngs_grd, unsigned long long seed)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {
        curand_init(seed, idx, 0, &dev_rngs_grd[idx]);
    }
}

__global__
void collrate_init (real *dev_collrate, real *dev_collrand, real *dev_collreal, int *dev_collrate_max)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {
        dev_collrate[idx] = 0.0;
        dev_collrand[idx] = 0.0;
        dev_collreal[idx] = 0.0;

        if (idx == 0) *dev_collrate_max = 0.0;
    }
}

__global__
void collrate_calc (swarm *dev_particle, tree *dev_treenode, real *dev_collrate, const cukd::box_t<float3> *dev_boundbox)
{
    int idx_tree = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx_tree >= 0 && idx_tree < N_PAR)
    {
        using candidatelist = cukd::HeapCandidateList<KNN_SIZE>;
        candidatelist query_result(static_cast<float>(MAX_DIST));
        cukd::cct::knn <candidatelist, tree, tree_traits> (query_result, dev_treenode[idx_tree].xyz, *dev_boundbox, dev_treenode, N_PAR);

        real collrate_ij = 0.0; // collision rate between particle i and j
        real collrate_i  = 0.0; // total collision rate for particle i

        int idx_old_1 = dev_treenode[idx_tree].index_old;
        int idx_old_2, idx_tmp;

        for(int j = 0; j < KNN_SIZE; j++)
        {
            collrate_ij = 0.0;
            idx_tmp = query_result.returnIndex(j);

            if (idx_tmp != -1)
            {
                idx_old_2 = dev_treenode[idx_tmp].index_old;
                collrate_ij = LAMBDA_0*dev_particle[idx_old_2].numgrain;
            }

            collrate_i += collrate_ij;
        }

        int idx_x = static_cast<int>(RES_X*(dev_particle[idx_old_1].position.x - X_MIN) / (X_MAX - X_MIN));
        int idx_y = static_cast<int>(RES_Y*(dev_particle[idx_old_1].position.y - Y_MIN) / (Y_MAX - Y_MIN));
        int idx_z = static_cast<int>(RES_Z*(dev_particle[idx_old_1].position.z - Z_MIN) / (Z_MAX - Z_MIN));

        dev_particle[idx_old_1].collrate = collrate_i;
        atomicAdd(&dev_collrate[idx_z*RES_Y*RES_X + idx_y*RES_X + idx_x], collrate_i);
    }
}

__global__
void collrate_peak (real *dev_collrate, int *dev_collrate_max)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {
        atomicMax(dev_collrate_max, static_cast<int>(dev_collrate[idx]));
    }
}

__global__
void collflag_calc (real *dev_collrate, real *dev_collrand, int *dev_collflag, real *dev_timestep, curandState *dev_rngs_grd)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_GRD)
    {
        real rand_collide = curand_uniform_double(&dev_rngs_grd[idx]); // (0,1]
        real real_collide = (*dev_timestep)*dev_collrate[idx];

        if (real_collide >= rand_collide)
        {
            dev_collflag[idx] = 1;
            dev_collrand[idx] = dev_collrate[idx]*curand_uniform_double(&dev_rngs_grd[idx]);
        }
        else
        {
            dev_collflag[idx] = 0;
        }
    }
}

__global__
void dustcoag_calc (swarm *dev_particle, tree *dev_treenode, int *dev_collflag, real *dev_collrand, real *dev_collreal,
    const cukd::box_t<float3> *dev_boundbox, curandState *dev_rngs_par)
{
    int idx_tree = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx_tree >= 0 && idx_tree < N_PAR)
    {
        int idx_old_1 = dev_treenode[idx_tree].index_old;
        
        int idx_x = static_cast<int>(RES_X*(dev_particle[idx_old_1].position.x - X_MIN) / (X_MAX - X_MIN));
        int idx_y = static_cast<int>(RES_Y*(dev_particle[idx_old_1].position.y - Y_MIN) / (Y_MAX - Y_MIN));
        int idx_z = static_cast<int>(RES_Z*(dev_particle[idx_old_1].position.z - Z_MIN) / (Z_MAX - Z_MIN));

        int idx_cell = idx_z*RES_Y*RES_X + idx_y*RES_X + idx_x;
        
        if (dev_collflag[idx_cell] == 1)
        {
            real collrate_i = dev_particle[idx_old_1].collrate;
            real rand_collide = dev_collrand[idx_cell];
            real real_collide = atomicAdd(&dev_collreal[idx_cell], collrate_i);

            if (real_collide < rand_collide && real_collide + collrate_i >= rand_collide)
            {
                using candidatelist = cukd::HeapCandidateList<KNN_SIZE>;
                candidatelist query_result(static_cast<float>(MAX_DIST));
                cukd::cct::knn <candidatelist, tree, tree_traits> (query_result, dev_treenode[idx_tree].xyz, *dev_boundbox, dev_treenode, N_PAR);

                int idx_old_2, idx_tmp, idx_knn = 0;

                real rand_collide_ij = collrate_i*curand_uniform_double(&dev_rngs_par[idx_old_1]);
                real real_collide_ij = 0.0;

                while (real_collide_ij < rand_collide_ij && idx_knn < KNN_SIZE)
                {
                    idx_tmp = query_result.returnIndex(idx_knn);

                    if (idx_tmp != -1)
                    {
                        idx_old_2 = dev_treenode[idx_tmp].index_old;
                        real_collide_ij += LAMBDA_0*dev_particle[idx_old_2].numgrain;
                    }

                    idx_knn++;
                }

                // collide with idx_old_2 and merge
                real size_1 = dev_particle[idx_old_1].dustsize;
                real size_2 = dev_particle[idx_old_2].dustsize;
                real size_3 = cbrt(size_1*size_1*size_1 + size_2*size_2*size_2);

                dev_particle[idx_old_1].dustsize  = size_3;
                dev_particle[idx_old_1].numgrain *= (size_1/size_3)*(size_1/size_3)*(size_1/size_3);
            }
        }
    }
}

int main ()
{
    real timer = 0.0;
    real filesave_timer = 0.0;

    std::string fname;
    std::ofstream ofile;
    std::uniform_real_distribution <real> random(0.0, 1.0); // distribution in [0, 1)

    swarm *particle, *dev_particle;
    cudaMallocHost((void**)&particle, sizeof(swarm)*N_PAR);
    cudaMalloc((void**)&dev_particle, sizeof(swarm)*N_PAR);

    tree *dev_treenode;
    cudaMalloc((void**)&dev_treenode, sizeof(tree)*N_PAR);

    real *timestep, *dev_timestep;
    cudaMallocHost((void**)&timestep, sizeof(real));
    cudaMalloc((void**)&dev_timestep, sizeof(real));

    int *collrate_max, *dev_collrate_max;
    cudaMallocHost((void**)&collrate_max, sizeof(int));
    cudaMalloc((void**)&dev_collrate_max, sizeof(int));

    int *dev_collflag;
    cudaMalloc((void**)&dev_collflag, sizeof(int)*N_GRD);

    real *dev_collrand;
    real *dev_collreal;
    cudaMalloc((void**)&dev_collrand, sizeof(real)*N_GRD);
    cudaMalloc((void**)&dev_collreal, sizeof(real)*N_GRD);

    real *collrate, *dev_collrate;
    cudaMallocHost((void**)&collrate, sizeof(real)*N_GRD);
    cudaMalloc((void**)&dev_collrate, sizeof(real)*N_GRD);

    curandState *dev_rngs_par, *dev_rngs_grd;
    cudaMalloc((void**)&dev_rngs_par, sizeof(curandState)*N_PAR);
    cudaMalloc((void**)&dev_rngs_grd, sizeof(curandState)*N_GRD);

    cukd::box_t<float3> *dev_boundbox;
    cudaMalloc((void**)&dev_boundbox, sizeof(cukd::box_t<float3>));

    real *profile_x, *dev_profile_x;
    real *profile_y, *dev_profile_y;
    real *profile_z, *dev_profile_z;
    cudaMallocHost((void**)&profile_x, sizeof(real)*N_PAR);
    cudaMalloc((void**)&dev_profile_x, sizeof(real)*N_PAR);
    cudaMallocHost((void**)&profile_y, sizeof(real)*N_PAR);
    cudaMalloc((void**)&dev_profile_y, sizeof(real)*N_PAR);
    cudaMallocHost((void**)&profile_z, sizeof(real)*N_PAR);
    cudaMalloc((void**)&dev_profile_z, sizeof(real)*N_PAR);

    rand_generator.seed(1);

    rand_uniform (profile_x, N_PAR, X_MIN, X_MAX);
    rand_uniform (profile_y, N_PAR, Y_MIN, Y_MAX);
    rand_uniform (profile_z, N_PAR, Z_MIN, Z_MAX);

    cudaMemcpy(dev_profile_x, profile_x, sizeof(real)*N_PAR, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_profile_y, profile_y, sizeof(real)*N_PAR, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_profile_z, profile_z, sizeof(real)*N_PAR, cudaMemcpyHostToDevice);

    particle_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_profile_x, dev_profile_y, dev_profile_z);

    cudaFreeHost(profile_x); cudaFree(dev_profile_x);
    cudaFreeHost(profile_y); cudaFree(dev_profile_y);
    cudaFreeHost(profile_z); cudaFree(dev_profile_z);

    cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_PAR, cudaMemcpyDeviceToHost);
    fname = PATH_FILESAVE + "particle_" + frame_num(0, 5) + ".par";
    open_bin_file(ofile, fname);
    save_bin_file(ofile, particle, N_PAR);

    treenode_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_treenode);
    cukd::buildTree <tree, tree_traits> (dev_treenode, N_PAR, dev_boundbox);

    parstate_init <<< NB_P, THREADS_PER_BLOCK >>> (dev_rngs_par);
    grdstate_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_rngs_grd);

    std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << std::setw(3) << std::setfill('0') << 0 << "/" << std::setw(3) << std::setfill('0') << FILENUM_MAX << " finished on " << std::ctime(&end_time);

    for (int i = 1; i <= FILENUM_MAX; i++)
    {
        while (filesave_timer < DT_FILESAVE)
        {
            collrate_init <<< NB_A, THREADS_PER_BLOCK >>> (dev_collrate, dev_collrand, dev_collreal, dev_collrate_max);
            collrate_calc <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_treenode, dev_collrate, dev_boundbox);
            collrate_peak <<< NB_A, THREADS_PER_BLOCK >>> (dev_collrate, dev_collrate_max);

            cudaMemcpy(collrate_max, dev_collrate_max, sizeof(int), cudaMemcpyDeviceToHost);
            // *timestep = -std::log(1.0 - random(rand_generator)) / static_cast<real>(*collrate_max);
            *timestep = 1.0 / static_cast<real>(*collrate_max);
            // if (*timestep > DT_DYNAMICS) *timestep = DT_DYNAMICS;
            if (*timestep > DT_FILESAVE - filesave_timer) *timestep = DT_FILESAVE - filesave_timer;
            cudaMemcpy(dev_timestep, timestep, sizeof(real), cudaMemcpyHostToDevice);
            
            collflag_calc <<< NB_A, THREADS_PER_BLOCK >>> (dev_collrate, dev_collrand, dev_collflag, dev_timestep, dev_rngs_grd);
            dustcoag_calc <<< NB_P, THREADS_PER_BLOCK >>> (dev_particle, dev_treenode, dev_collflag, dev_collrand, dev_collreal, dev_boundbox, dev_rngs_par);

            timer += *timestep;
            filesave_timer += *timestep;
            
            // std::cout << std::setprecision(6) << std::scientific << *timestep << ' ' << filesave_timer << ' ' << timer << std::endl;
        }
    
        filesave_timer = 0.0;
        
        cudaMemcpy(particle, dev_particle, sizeof(swarm)*N_PAR, cudaMemcpyDeviceToHost);
        fname = PATH_FILESAVE + "particle_" + frame_num(i, 5) + ".par";
        open_bin_file(ofile, fname);
        save_bin_file(ofile, particle, N_PAR);

        std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::setw(3) << std::setfill('0') << i << "/" << std::setw(3) << std::setfill('0') << FILENUM_MAX << " finished on " << std::ctime(&end_time);
    }
    
    return 0;
}