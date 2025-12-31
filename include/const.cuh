#ifndef CONST_CUH
#define CONST_CUH

#include <cmath>                            // for M_PI
#include <string>                           // for std::string

#include "cukd/builder.h"                   // for cukd::get_coord

using real  = double;
using real3 = double3;
using boxf  = cukd::box_t<float3>;

// =========================================================================================================================
// code units

const real  G           = 1.0;
const real  M_S         = 1.0;
const real  R_0         = 1.0;

// =========================================================================================================================
// gas paramters

const real  NU          = 0.0;              // the kinematic viscosity parameter of the gas
const real  ALPHA       = 1.0e-4;           // the Shakura-Sunayev viscosity parameter of the gas

const real  h_0         = 0.05;             // the reference aspect ratio of the gas disk
const real  IDX_TEMP    = -0.4;             // the power-law index of the gas temperature profile
const real  IDX_SURF    = -1.5;             // the power-law index of the gas surface density profile

// =========================================================================================================================
// dust parameters for dynamics

const real  S_0         = 1.0;              // the reference grain size of the dust
const real  ST_0        = 1.0e-04;          // the reference Stokes number of the dust
const real  BETA_0      = 1.0e+01;          // the reference ratio between the radiation pressure and the gravity
const real  KAPPA_0     = 1.0;              // the reference gray opacity of the dust

// =========================================================================================================================
// dust parameters for coagulation

const real  M_D         = 1.0e30;           // the total dust mass in the disk
const real  RHO_0       = 1.0;              // the reference internal density of the dust
const real  LAMBDA_0    = 1.0e-20;          // the reference collision rate of the dust
const real  V_FRAG      = 1.0e-04;          // the fragmentation velocity for dust collision

const int   KNN_SIZE = 100;                 // the maximum number   for neighbor search 
const float MAX_DIST = 0.05;                // the maximum distance for neighbor search

// =========================================================================================================================
// mesh domain size and resolution

const int   N_PAR         = 1e+07;            // total number of super-particles in the model

const int   N_X         = 1;
const real  X_MIN       = M_PI;
const real  X_MAX       = M_PI;

const int   N_Y         = 100;
const real  Y_MIN       = 1.0;
const real  Y_MAX       = 3.0;

const int   N_Z         = 1;
const real  Z_MIN       = 0.5*M_PI;
const real  Z_MAX       = 0.5*M_PI;

// =========================================================================================================================
// dust initialization parameters

const real INIT_XMIN    = X_MIN;
const real INIT_XMAX    = X_MAX;

const real INIT_YMIN    = Y_MIN;
const real INIT_YMAX    = Y_MAX;

const real INIT_ZMIN    = Z_MIN;
const real INIT_ZMAX    = Z_MAX;

const real INIT_SMIN    = 0.01;
const real INIT_SMAX    = 10.0;

// =========================================================================================================================
// time step and output parameters

const int  FILENUM_MAX = 100;               // total number of outputs for mesh fields
const int  SWARM_EVERY = 1;                 // save particle data every X mesh outputs

const real DT_FILESAVE = 0.2*M_PI;          // time interval between adjascent outputs
const real DT_DYNAMICS = 2.0*M_PI/static_cast<real>(N_X);

const std::string PATH_FILESAVE = "outputs/";

// =========================================================================================================================
// structures

struct swarm
{
    real3   position;                       // x = azimuth [radian], y = radius [R_0], z = colattitude [radian]
    real3   velocity;                       // velocity for rad but specific angular momentum for azi and col
    real    par_size;                       // size of an individual dust grain in the swarm
    real    par_numr;                       // number of individual dust grains in the swarm
    real    col_rate;                       // total collision rate for the particle i
};

struct tree
{
    float3  cartesian;                      // xyz position of a tree node
    int     index_old;                      // index of the partile before index shuffling by the KD-tree builder
    int     split_dim;                      // parameter for the KD-tree builder
};

struct tree_traits
{
    using point_t = float3;
    enum { has_explicit_dim = true };
    
    static inline __host__ __device__ const point_t &get_point (const tree &node) { return node.cartesian; }
    static inline __host__ __device__ float get_coord (const tree &node, int dim) { return cukd::get_coord(node.cartesian, dim); }
    static inline __host__ __device__ int get_dim (const tree &node) { return node.split_dim; }
    static inline __host__ __device__ void set_dim (tree &node, int dim) { node.split_dim = dim; }
};

struct interp
{
    int  next_x, next_y, next_z;
    real frac_x, frac_y, frac_z;
};

enum GridFieldType
{ 
    OPTDEPTH,   // optical depth field
    DUSTDENS,   // dust density field
};

// =========================================================================================================================
// cuda numerical parameters

const int THREADS_PER_BLOCK = 32;

// number of grid cells
const int N_GRD = N_X*N_Y*N_Z;
const int NG_XY = N_X*N_Y;
const int NG_XZ = N_X*N_Z;
const int NG_YZ = N_Y*N_Z;

const int NB_P = N_PAR / THREADS_PER_BLOCK + 1; // number of blocks for swarm-level parallelization
const int NB_A = N_GRD / THREADS_PER_BLOCK + 1; // number of blocks for cell-level  parallelization
const int NB_X = NG_YZ / THREADS_PER_BLOCK + 1; // number of blocks for X-direction parallelization
const int NB_Y = NG_XZ / THREADS_PER_BLOCK + 1; // number of blocks for Y-direction parallelization

// =========================================================================================================================

#endif
