Purpose
-------
This file provides focused, repo-specific guidance to AI coding agents working on cuDust (CUDA dust-coagulation/simulation code). Use it to understand the big-picture architecture, where to change runtime parameters, build/run commands, and important coding patterns to preserve.

**Big picture**
- The repository implements a particle-based dust evolution simulation written in CUDA/C++: host orchestration in `source/main.cu`, many device kernels in `source/*.cu`, and kernel declarations in `include/cudust.cuh`.
- Particles are represented by `struct swarm` (see `include/const.cuh`) and live on device memory (`dev_particle`). Grid fields (optical depth, dust density) live in arrays like `dev_optdepth` and `dev_dustdens`.
- KD-tree neighbor search and tree building come from the vendored header-only `include/cukd/` library; main tree build happens in `source/main.cu` via `cukd::buildTree<tree, tree_traits>(...)`.

**Key files & examples**
- `include/const.cuh`: single place for simulation constants and structures (e.g. `N_PAR`, `N_X/N_Y/N_Z`, `THREADS_PER_BLOCK`, `PATH_FILESAVE`). Change problem size or resolution here.
- `include/cudust.cuh`: kernel declarations and host helpers (frame naming, file IO, random profiles).
- `source/main.cu`: program entrypoint and main evolution loop; shows orchestration of: tree build, `col_rate_calc`, `run_collision`, `ssa_substep_*`, grid updates and file saves.
- `source/mesh.cu`: grid-related kernels (e.g., `optdepth_init`, `optdepth_calc`).
- `source/collision.cu`: collision rate computation and collision-handling kernels (`col_rate_calc`, `run_collision`).
- `include/cukd/*`: KD-tree builders; choose builder variant via the `CUKD_BUILDER_*` defines in `cukd/builder.h` (thrust/bitonic/inplace) — significant perf/memory tradeoffs.

**Build & run (practical commands)**
- Build: run `make` at repo root. The Makefile uses `nvcc -arch=sm_70` by default.
- Change target GPU arch: edit `Makefile` `CC` variable to use the appropriate `-arch=sm_XX` for your GPU.
- Run: `./cuDust` (fresh start) or `./cuDust <frame-number>` to resume from `outputs/particle_<frame>.par` (see `main.cu` resume logic).
- Output files are written to `outputs/` (default defined by `PATH_FILESAVE` in `include/const.cuh`).

**Project-specific conventions & patterns**
- Declarations vs implementations: declare kernels and host helpers in `include/cudust.cuh`; implement them in `source/*.cu`. Keep that separation.
- Host/device memory pattern: code uses `cudaMallocHost` for pinned host buffers and `cudaMalloc` for device buffers (see `main.cu`); follow these allocations for performance-critical host-device transfers.
- Grid indexing: grid cell indices computed from particle positions (see `collision.cu` and `integrator.cu`); treat `N_X/N_Y/N_Z` and `NG_XY` constants carefully when modifying resolution.
- RNGs: uses both host `std::mt19937` (for initialization) and `curand` device RNGs (`curandState`) for per-thread randomness; do not mix semantics — look at `rs_swarm_init` and `rs_grids_init` in `initialize.cu`.

**Performance & safety notes**
- Default `N_PAR = 1e7` (in `include/const.cuh`) is huge. For local dev rely on a small test size (reduce `N_PAR`, `N_Y`) before running on a cluster/GPU with large memory.
- KD-tree builder selection impacts both memory and build/runtime speed; `builder_thrust` is fastest but requires extra temporary memory; `builder_inplace` is memory-sparing but slower. See `include/cukd/builder.h` comments.

**Where to change behavior**
- To change simulation size / resolution: edit `include/const.cuh` (e.g., `N_PAR`, `N_Y`, `N_X`, `THREADS_PER_BLOCK`). Rebuild after changes.
- To tune kernel launch sizes: either change `THREADS_PER_BLOCK` or compute blocks in `const.cuh` NB_P / NB_A logic.
- To change file output behavior look for `open_bin_file`, `save_bin_file`, `save_variable` declarations in `include/cudust.cuh` and their implementations in `source/outputs.cu`.

**Debugging tips specific to this repo**
- Reduce `N_PAR` and `N_GRD` to a tiny value to reproduce logic quickly; many kernels assume thread loops up to `N_PAR` or `N_GRD` using `idx < N_PAR` guards.
- Use `cuda-memcheck` / `nsight` when encountering crashes; common issues are out-of-bounds grid indexing because of log-based mapping in `collision.cu` or wrong `THREADS_PER_BLOCK`.
- To reproduce the resume path, run `./cuDust <frame>` and ensure `outputs/particle_<frame>.par` exists (written by prior runs). See `main.cu` load logic.

**External dependencies**
- NVIDIA CUDA toolkit (nvcc), libcurand (curand_kernel.h). Standard C++ STL used for host-side I/O and RNG.

If any section is unclear or you'd like me to include more code examples (line references or expanded explanation for a particular kernel), tell me which area to expand and I'll update this file.
