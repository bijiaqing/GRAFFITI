Purpose
-------
This file provides focused, repo-specific guidance to AI coding agents working on cuDust (CUDA dust-coagulation/simulation code). Use it to understand the big-picture architecture, where to change runtime parameters, build/run commands, and important coding patterns to preserve.

**Big picture**
- The repository implements a particle-based dust evolution simulation written in CUDA/C++: host orchestration in `src/main.cu`, many device kernels in `src/*.cu`, and kernel/helper declarations in `inc/cudust_kern.cuh` and `inc/cudust_host.cuh`.
- Particles are represented by `struct swarm` (see `inc/const.cuh`) and live on device memory (`dev_particle`). Grid fields (optical depth, dust density) live in arrays like `dev_optdepth` and `dev_dustdens`.
- KD-tree neighbor search and tree building come from the vendored header-only `inc/cukd/` library; main tree build happens in `src/main.cu` via `cukd::buildTree<tree, tree_traits>(...)`.

**Key files & examples**
- `inc/const.cuh`: single place for simulation constants and structures (e.g. `N_PAR`, `N_X/N_Y/N_Z`, `THREADS_PER_BLOCK`, `PATH_OUT`). Change problem size or resolution here. **Important:** use `#define` for constants that appear in preprocessor conditionals (`#if`), not `const int`.
- `inc/cudust_kern.cuh`: kernel declarations (device code).
- `inc/cudust_host.cuh`: host-side helper functions (frame naming, file IO, random profiles, logarithmic output check).
- `inc/helpers_*.cuh`: device helper functions organized by domain (collision, diffusion, disk parameters, grid fields, transport).
- `src/main.cu`: program entrypoint and main evolution loop; shows orchestration of: tree build, `col_rate_calc`, `run_collision`, `ssa_substep_*`, grid updates and file saves.
- `src/gridfields.cu`: grid-related kernels (e.g., `optdepth_init`, `optdepth_calc`).
- `src/collision.cu`: collision rate computation and collision-handling kernels (`col_rate_calc`, `run_collision`).
- `inc/cukd/*`: KD-tree builders; choose builder variant via the `CUKD_BUILDER_*` defines in `cukd/builder.h` (thrust/bitonic/inplace) — significant perf/memory tradeoffs.

**Build & run (practical commands)**
- Build: run `make` at repo root. The Makefile uses `nvcc -arch=sm_70` by default.
- Change target GPU arch: edit `Makefile` `CC` variable to use the appropriate `-arch=sm_XX` for your GPU.
- Run: `./cuDust` (fresh start) or `./cuDust <frame-number>` to resume from saved particle data (see `src/main.cu` resume logic).
- Output files are written to `./out/` (default defined by `PATH_OUT` in `inc/const.cuh`).
- Conditional compilation: use `-DTRANSPORT`, `-DRADIATION`, `-DDIFFUSION`, `-DCOLLISION`, `-DCONST_NU`, `-DCONST_ST`, `-DLOGOUTPUT` flags to enable optional features (see `Makefile`).
- **Physics module flags**: `-DTRANSPORT` controls particle movement; when OFF, `-DRADIATION` and `-DDIFFUSION` are inactive regardless of their settings. `-DCOLLISION` is independent and can be ON even when TRANSPORT is OFF. See `.github/transport-flag-implementation.md` for details.

**Project-specific conventions & patterns**
- Declarations vs implementations: declare kernels in `inc/cudust_kern.cuh`, host helpers in `inc/cudust_host.cuh`, device helpers in `inc/helpers_*.cuh`; implement them in `src/*.cu`. Keep that separation.
- Host/device memory pattern: code uses `cudaMallocHost` for pinned host buffers and `cudaMalloc` for device buffers (see `src/main.cu`); follow these allocations for performance-critical host-device transfers.
- Grid indexing: grid cell indices computed from particle positions (see `src/collision.cu` and `src/integrator.cu`); treat `N_X/N_Y/N_Z` and `NG_XY` constants carefully when modifying resolution.
- RNGs: uses both host `std::mt19937` (for initialization) and `curand` device RNGs (`curandState`) for per-thread randomness; do not mix semantics — look at `rs_swarm_init` and `rs_grids_init` in `src/initialize.cu`.
- **Preprocessor vs runtime conditionals**: Use `#define` for constants checked in `#if` directives (preprocessor evaluates at compile-time). Using `const int` in `#if` conditions will fail — preprocessor cannot evaluate runtime constants. This can cause infinite loops or wrong branches being compiled. When in doubt, use runtime `if` statements instead of `#if`.
- **Device function inlining**: Device helper functions in headers must be `__forceinline__` to avoid multiple definition linker errors when included in multiple `.cu` files.
- **Variable scope with preprocessor blocks**: Declare variables before `#ifdef`/`#ifndef` blocks if they need to be visible outside; only assign values inside the conditional blocks.

**Performance & safety notes**
- Default `N_PAR = 1e7` (in `inc/const.cuh`) is huge. For local dev rely on a small test size (reduce `N_PAR`, `N_Y`) before running on a cluster/GPU with large memory.
- KD-tree builder selection impacts both memory and build/runtime speed; `builder_thrust` is fastest but requires extra temporary memory; `builder_inplace` is memory-sparing but slower. See `inc/cukd/builder.h` comments.

**Where to change behavior**
- To change simulation size / resolution: edit `inc/const.cuh` (e.g., `N_PAR`, `N_Y`, `N_X`, `THREADS_PER_BLOCK`). Rebuild after changes.
- To tune kernel launch sizes: either change `THREADS_PER_BLOCK` or compute blocks in `const.cuh` NB_P / NB_A logic.
- To change file output behavior look for `open_bin_file`, `save_bin_file`, `save_variable` declarations in `inc/cudust_host.cuh` and their implementations in `src/fileoutput.cu`.
- To enable logarithmic time output: add `-DLOGOUTPUT` flag to Makefile, set `LOG_BASE` in `inc/const.cuh`; particle data will be saved at t = DT_OUT × LOG_BASE^N for N = 0, 1, 2, ...

**Debugging tips specific to this repo**
- Reduce `N_PAR` and `N_GRD` to a tiny value to reproduce logic quickly; many kernels assume thread loops up to `N_PAR` or `N_GRD` using `idx < N_PAR` guards.
- Use `cuda-memcheck` / `nsight` when encountering crashes; common issues are out-of-bounds grid indexing because of log-based mapping in `src/collision.cu` or wrong `THREADS_PER_BLOCK`.
- To reproduce the resume path, run `./cuDust <frame>` and ensure saved particle data exists (written by prior runs). See `src/main.cu` load logic.
- **Infinite loop bugs**: When `N_X==1` and `X_MIN==X_MAX`, loops like `while(x >= X_MAX) x -= X_MAX-X_MIN` become infinite (subtracting zero). Check boundary handling in `inc/helpers_transport.cuh`.

**External dependencies**
- NVIDIA CUDA toolkit (nvcc), libcurand (curand_kernel.h). Standard C++ STL used for host-side I/O and RNG.

If any section is unclear or you'd like me to include more code examples (line references or expanded explanation for a particular kernel), tell me which area to expand and I'll update this file.
