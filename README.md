# GRAFFITI

![GRAFFITI Banner](doc/banner.png)

**GPU-paRallelized numericAl Framework For dust evolution In asTrophysical dIsks**

GRAFFITI is a high-performance CUDA-based simulation framework for modeling dust particle evolution in astrophysical disks. It leverages GPU parallelization to simulate millions of dust particles using a super-particle approach, including dynamics, collisions, diffusion, and radiative processes.

## Features

- **Particle-Based Simulation**: Tracks up to 10⁷ super-particles representing dust evolution
- **KD-Tree Neighbor Search**: Efficient collision detection using header-only `cukd` library
- **Modular Physics**:
  - Transport (advection)
  - Radiation pressure
  - Turbulent diffusion
  - Collision and coagulation/fragmentation
- **Grid-Based Fields**: Optical depth and dust density calculations on Eulerian grids
- **Flexible Configuration**: Compilation flags for enabling/disabling physics modules
- **Resume Capability**: Save and resume simulations from checkpoints

## Requirements

- **NVIDIA CUDA Toolkit** (tested with CUDA 11+)
- **GPU** with compute capability ≥ 8.0 (sm_80)
- **libcurand** (included with CUDA)
- **Standard C++ compiler** with C++17 support

## Quick Start

### Build

GRAFFITI uses a model-based build system. You must specify a `MODEL` when building:

```bash
# Example: Build the linear_kernel_test model
make MODEL=linear_kernel_test
```

The model directory (`mod/<MODEL>/`) must contain:
- `flags.mk`: Compilation flags for physics modules
- `const.cuh`: Model-specific constants (optional override)
- Source files that override defaults from `src/`

### Run

```bash
# Fresh start
./mod/<MODEL>/graffiti

# Resume from saved frame
./mod/<MODEL>/graffiti <frame-number>
```

Output files are written to `out/<MODEL>/`.

### Clean

```bash
# Clean specific model
make clean MODEL=linear_kernel_test

# Clean all models
make clean
```

## Configuration

### Physics Modules

Control simulation features via compilation flags in `mod/<MODEL>/flags.mk`:

| Flag | Description |
|------|-------------|
| `-DTRANSPORT` | Enable particle advection (required for RADIATION and DIFFUSION) |
| `-DRADIATION` | Enable radiation pressure (requires TRANSPORT) |
| `-DDIFFUSION` | Enable turbulent diffusion (requires TRANSPORT) |
| `-DCOLLISION` | Enable particle collisions (independent of TRANSPORT) |
| `-DCONST_NU` | Use constant kinematic viscosity instead of α-parameter |
| `-DCONST_ST` | Use constant Stokes number |
| `-DLOGOUTPUT` | Enable logarithmic time output spacing |

**Important**: When `TRANSPORT` is OFF, both `RADIATION` and `DIFFUSION` are inactive regardless of their settings. `COLLISION` is independent and can be enabled even when `TRANSPORT` is OFF.

### Simulation Parameters

Edit model-specific constants in `mod/<MODEL>/const.cuh` or the default `inc/const.cuh`:

**Domain & Resolution**:
```c
const int   N_P      = 1e+07;    // Number of super-particles
const int   N_X      = 100;      // Grid cells (azimuthal)
const int   N_Y      = 100;      // Grid cells (radial)
const int   N_Z      = 1;        // Grid cells (vertical)
```

**Physical Parameters**:
```c
const real  SIGMA_0  = 1.0e-02;  // Gas surface density at R_0
const real  ASPR_0   = 0.05;     // Disk aspect ratio
const real  ST_0     = 1.0e-03;  // Reference Stokes number
const real  ALPHA    = 1.0e-04;  // Shakura-Sunyaev α parameter
```

**Collision Parameters** (when `COLLISION` enabled):
```c
const int   COAG_KERNEL = 0;     // Kernel type: 0=const, 1=linear, 2=product, 3=custom
const int   N_K         = 10;    // Max KNN neighbors
const real  V_FRAG      = 1.0;   // Fragmentation velocity
```

**Output Control**:
```c
const real  DT_OUT   = 1.0e-02;  // Output time interval
```

For logarithmic output (with `-DLOGOUTPUT`):
```c
const real  LOG_BASE = 1.1;      // Outputs at t = DT_OUT × LOG_BASE^N
```

### GPU Architecture

Change target GPU in `Makefile`:
```makefile
NVCC += -arch=sm_80  # Change to match your GPU (e.g., sm_70, sm_86)
```

## Project Structure

```
graffiti/
├── inc/                      # Header files
│   ├── const.cuh            # Default simulation constants
│   ├── graffiti_host.cuh    # Host function declarations
│   ├── graffiti_kern.cuh    # Kernel declarations
│   ├── helpers_*.cuh        # Device helper functions
│   └── cukd/                # KD-tree library (header-only)
├── src/                      # Default source implementations
│   ├── graffiti_main.cu     # Main program & evolution loop
│   ├── collision.cu         # Collision kernels
│   ├── diffusion_*.cu       # Diffusion kernels
│   ├── particle_init.cu     # Initialization
│   └── ...
├── mod/                      # Model configurations
│   └── <MODEL>/             # Specific model setup
│       ├── flags.mk         # Compilation flags
│       ├── const.cuh        # Model constants (optional)
│       └── *.cu             # Model-specific overrides (optional)
├── obj/<MODEL>/             # Build artifacts (ignored)
├── out/<MODEL>/             # Simulation output (ignored)
├── doc/                     # Documentation & images
└── Makefile                 # Build system
```

## Code Organization

### Key Files

- **`src/graffiti_main.cu`**: Program entry point, main evolution loop, tree building, and I/O orchestration
- **`inc/const.cuh`**: Single source for simulation constants and structures (e.g., `struct swarm`)
- **`src/collision.cu`**: Collision rate calculation and handling (`col_rate_calc`, `run_collision`)
- **`src/*_init.cu`**: Initialization routines for particles, grids, and RNG states
- **`inc/cukd/`**: KD-tree builders with different memory/speed tradeoffs

### Coding Conventions

- **Declarations**: Kernels in `inc/graffiti_kern.cuh`, host helpers in `inc/graffiti_host.cuh`, device helpers in `inc/helpers_*.cuh`
- **Implementations**: In corresponding `src/*.cu` files
- **Memory**: Host uses `cudaMallocHost` (pinned), device uses `cudaMalloc`
- **Grid Indexing**: Cells computed from positions; respect `N_X/N_Y/N_Z` constants
- **RNG**: Host uses `std::mt19937`, device uses `curandState` per-thread
- **Device Helpers**: Must be `__forceinline__` to avoid linker errors

## Performance Notes

- Default `N_P = 1e7` requires substantial GPU memory. For testing, reduce `N_P` and `N_Y` in constants
- **KD-tree builders** (in `inc/cukd/builder.h`):
  - `builder_thrust`: Fastest, requires extra temp memory
  - `builder_inplace`: Memory-efficient, slower build
- Use `THREADS_PER_BLOCK` in `const.cuh` to tune kernel launch parameters

## Debugging Tips

1. **Reduce problem size**: Set `N_P = 1e5` and `N_Y = 10` for quick testing
2. **Check boundaries**: Infinite loop bugs can occur with `N_X=1` when `X_MIN==X_MAX`
3. **Use CUDA tools**: `cuda-memcheck` or `nsight-compute` for crashes
4. **Resume testing**: Ensure saved particle data exists before running with `<frame>`
5. **Preprocessor vs runtime**: Use `#define` for constants in `#if` conditions, not `const int`

## Output Files

Binary output files saved to `out/<MODEL>/`:
- Particle data: positions, velocities, sizes, masses
- Grid fields: optical depth, dust density
- Frame naming controlled by `open_bin_file()` and `save_bin_file()` in `inc/graffiti_host.cuh`

## Citation

If you use GRAFFITI in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

For questions, issues, or contributions, please contact [contact information to be added] or open an issue on the repository.

---

**Note**: This is a research code under active development. Performance characteristics and API may change between versions.
