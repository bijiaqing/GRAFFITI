#ifndef CONST_CUH
#define CONST_CUH

#include <cmath>                            // for M_PI
#include <string>                           // for std::string

#if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
#include "curand_kernel.h"                  // for curandState
using curs = curandState;
#endif // COLLISION or DIFFUSION

#ifdef COLLISION
#include "cukd/builder.h"   // for cukd::get_coord, cukd::box_t
using bbox = cukd::box_t<float3>;           // axis-aligned bounding box type for KD-tree
#endif // COLLISION

using real  = double;                       // code real type
using real3 = double3;                      // double3 is a built-in CUDA type

// =========================================================================================================================
// code units

const real  G           = 1.0;              // gravitational constant
const real  M_S         = 1.0;              // mass of the central star
const real  R_0         = 1.0;              // reference radius of the disk
const real  S_0         = 1.0;              // the reference grain size of the dust, decoupled from R_0 for flexibility

// =========================================================================================================================
// mesh domain size and resolution

const int   N_PAR       = 1e+07;            // total number of super-particles in the model

const int   N_X         = 50;               // number of grid cells in X direction (azimuth)
const real  X_MIN       = -M_PI;            // minimum X boundary (azimuth)
const real  X_MAX       = +M_PI;            // maximum X boundary (azimuth)

const int   N_Y         = 50;               // number of grid cells in Y direction (radius)
const real  Y_MIN       = 0.5;              // minimum Y boundary (radius)
const real  Y_MAX       = 1.5;              // maximum Y boundary (radius)

const int   N_Z         = 50;               // number of grid cells in Z direction (colattitude)
const real  Z_MIN       = 0.5*M_PI - 0.2;   // minimum Z boundary (colattitude)
const real  Z_MAX       = 0.5*M_PI + 0.2;   // maximum Z boundary (colattitude)

// =========================================================================================================================
// gas parameters

const real  SIGMA_0     = 1.0e-02;          // the reference gas surface density at R_0, used for collision rate calculation
const real  ASPR_0      = 0.05;             // the reference aspect ratio of the gas disk
const real  IDX_P       = -1.0;             // the radial power-law index of the gas surface density profile
const real  IDX_Q       = -0.4;             // the radial power-law index of the gas temperature profile (vertically isothermal)

#if defined(COLLISION) || (defined(TRANSPORT) && defined(DIFFUSION))
#ifndef CONST_NU
const real  ALPHA       = 1.0e-02;          // the Shakura-Sunayev viscosity parameter of the gas
#else  // CONST_NU
const real  NU          = 1.0e-04;          // the kinematic viscosity parameter of the gas
#endif // NOT CONST_NU
#endif // COLLISION or DIFFUSION

#ifdef COLLISION
#ifndef CODE_UNIT
const real  M_MOL       = 2.3*1.66054e-24;  // mean molecular weight of the gas in grams
const real  X_SEC       = 2.0e-15;          // the cross section of H2 gas in cm^2
#else  // CODE_UNIT
const real  RE_0        = 1.0e+08;          // reference Reynolds number at R_0
#endif // NOT CODE_UNIT
#endif // COLLISION

// =========================================================================================================================
// dust parameters for dynamics

const real  ST_0        = 1.0e-01;          // the reference Stokes number of dust with the reference size

const real  M_D         = 1.0e30;           // the total dust mass in the disk, decoupled from M_S for flexibility
const real  RHO_0       = 1.0;              // the reference internal density of the dust

#if defined(TRANSPORT) && defined(RADIATION)
const real  BETA_0      = 1.0e+01;          // the reference ratio between the radiation pressure and the gravity
const real  KAPPA_0     = 3.0e-28;          // the reference gray opacity of the dust
#endif // RADIATION

#if defined(TRANSPORT) && defined(DIFFUSION)
const real  SC_R        = 1.0e+10;          // the Schmidt number for radial   diffusion
#endif // DIFFUSION

#if (defined(TRANSPORT) && defined(DIFFUSION)) || defined(COLLISION)
const real  SC_Z        = 1.0;              // the Schmidt number for vertical diffusion
#endif // DIFFUSION or COLLISION

#ifdef COLLISION
const int   COAG_KERNEL = 2;                // coagulation kernels: 0 = constant, 1 = linear, 2 = product, 3 = custom
const int   N_K         = 200;              // the maximum number for KNN neighbor search 

const real  LAMBDA_0    = N_X*N_Y*N_Z/N_K;  // the reference collision rate of the dust
const real  V_FRAG      = 1.0;              // the fragmentation velocity for dust collision
#endif // COLLISION

// =========================================================================================================================
// dust initialization parameters

const real INIT_XMIN    = X_MIN;            // minimum X boundary for particle initialization
const real INIT_XMAX    = X_MAX;            // maximum X boundary for particle initialization

const real INIT_YMIN    = Y_MIN;            // minimum Y boundary for particle initialization
const real INIT_YMAX    = Y_MAX;            // maximum Y boundary for particle initialization

const real INIT_ZMIN    = Z_MIN;            // minimum Z boundary for particle initialization
const real INIT_ZMAX    = Z_MAX;            // maximum Z boundary for particle initialization

const real INIT_SMIN    = 1.0e+00;          // minimum grain size for particle initialization
const real INIT_SMAX    = 1.0e+00;          // maximum grain size for particle initialization

// =========================================================================================================================
// time step and output parameters

const int  SAVE_MAX     = 20;              // total number of outputs for mesh fields

const real DT_OUT       = 0.1;

#ifdef TRANSPORT
const real DT_DYN       = 0.1;
#endif // TRANSPORT

#ifdef LOGOUTPUT
const int  LOG_BASE     = 2;               // save particle data at t = DT_OUT*LOG_BASE^N, N = 0, 1, 2, ...
#else  // LINEAR
const int  LIN_BASE     = 1;                // save particle data at t = DT_OUT*LIN_BASE*N, N = 0, 1, 2, ...
#endif // LOGOUTPUT

const real DT_MIN       = 1.0e-14;         // calculation steps with smaller dt will be skipped

const std::string PATH_OUT = "./out/";

// =========================================================================================================================
// structures

struct swarm                                // for particle swarm
{
    real3   position;                       // x = azimuth [radian], y = radius [R_0], z = colattitude [radian]
    real3   velocity;                       // velocity for rad but specific angular momentum for azi and col
    real    par_size;                       // size of an individual dust grain in the swarm
    real    par_numr;                       // number of individual dust grains in the swarm

    #ifdef COLLISION
    real    col_rate;                       // total collision rate for the particle i
    real    max_dist;                       // maximum distance to the KNN neighbors
    #endif // COLLISION
};

#ifdef COLLISION
struct tree                                 // KD-tree node structure for cukd::builder
{
    float3  cartesian;                      // xyz position of a tree node in Cartesian coordinate
    int     index_old;                      // index of the particle before index shuffling by the KD-tree builder
    int     split_dim;                      // splitting dimension of the tree node
};

struct tree_traits                          // traits for cukd::builder
{
    using point_t = float3;
    enum { has_explicit_dim = true };
    
    // getter and setter functions
    static inline __host__ __device__ const point_t &get_point (const tree &node) { return node.cartesian; }
    static inline __host__ __device__ float get_coord (const tree &node, int dim) { return cukd::get_coord(node.cartesian, dim); }
    static inline __host__ __device__ int get_dim (const tree &node) { return node.split_dim; }
    static inline __host__ __device__ void set_dim (tree &node, int dim) { node.split_dim = dim; }
};
#endif // COLLISION

// =========================================================================================================================
// cuda numerical parameters

const int THREADS_PER_BLOCK = 32;               // number of threads per block

const int N_GRD = N_X*N_Y*N_Z;                  // total number of grid cells
const int NG_XY = N_X*N_Y;                      // number of grid cells in X-Y plane
const int NG_XZ = N_X*N_Z;                      // number of grid cells in X-Z plane
const int NG_YZ = N_Y*N_Z;                      // number of grid cells in Y-Z plane

const int NB_P = N_PAR / THREADS_PER_BLOCK + 1; // number of blocks for swarm-level parallelization
const int NB_A = N_GRD / THREADS_PER_BLOCK + 1; // number of blocks for cell-level  parallelization
const int NB_X = NG_YZ / THREADS_PER_BLOCK + 1; // number of blocks for X-direction parallelization
const int NB_Y = NG_XZ / THREADS_PER_BLOCK + 1; // number of blocks for Y-direction parallelization

// =========================================================================================================================

#endif // NOT CONST_CUH
