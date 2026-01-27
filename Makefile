INC_DIR = ./inc
OBJ_DIR = ./obj
SRC_DIR = ./src
OUT_DIR = ./out

NVCC = nvcc
# sm_80 is for A100 GPU
NVCC += -arch=sm_80
# use fast math operations, which are less precise but faster
NVCC += --use_fast_math
# suppress some warnings
# 177: variable was declared but never referenced
# 550: variable was set but never used
NVCC += --diag-suppress 177,550

# =========================================================================================================================
# Compile-time feature flags
# Uncomment lines below to enable specific physics modules

# CODE_UNIT: Use code units instead of cgs units
NVCC += -DCODE_UNIT

# SAVE_DENS: Save dust density field to output files
# NVCC += -DSAVE_DENS

# LOGOUTPUT: Use logarithmic output intervals
# NVCC += -DLOGOUTPUT

# COLLISION: Enable dust collision and coagulation/fragmentation
NVCC += -DCOLLISION

# TRANSPORT: Enable particle transport (position/velocity evolution)
# NOTE: If TRANSPORT is off, RADIATION and DIFFUSION are inactive regardless of their flags
# NVCC += -DTRANSPORT

# RADIATION: Enable radiation pressure calculations (optical depth, beta)
# NVCC += -DRADIATION

# DIFFUSION: Enable turbulent diffusion of dust particles
# NVCC += -DDIFFUSION

# CONST_NU: Use constant kinematic viscosity NU instead of alpha-viscosity
# NVCC += -DCONST_NU

# CONST_ST: Use constant Stokes number instead of constant physical size for dust particles
# NVCC += -DCONST_ST

# =========================================================================================================================

EXEC = cuDust

_INC = const.cuh             \
       cudust_host.cuh       \
       cudust_kern.cuh       \
       helpers_collision.cuh \
       helpers_diskparam.cuh \
       helpers_gridfield.cuh \
       helpers_transport.cuh

_OBJ = col_flag_calc.o \
       col_proc_exec.o \
       col_rate_calc.o \
       col_rate_init.o \
       diffusion_pos.o \
       diffusion_vel.o \
       dustdens_calc.o \
       dustdens_init.o \
       dustdens_scat.o \
       main.o          \
       optdepth_calc.o \
       optdepth_csum.o \
       optdepth_init.o \
       optdepth_mean.o \
       optdepth_scat.o \
       particle_init.o \
       rs_grids_init.o \
       rs_swarm_init.o \
       ssa_substep_1.o \
       ssa_substep_2.o \
       ssa_transport.o \
       col_tree_init.o

INC = $(patsubst %, $(INC_DIR)/%, $(_INC))
OBJ = $(patsubst %, $(OBJ_DIR)/%, $(_OBJ))

.PHONY: all clean cleanall

all: $(EXEC)

$(EXEC): $(OBJ)
	@printf "%-12s %-25s %s\n" "Linking" "$@" "from $(words $(OBJ)) objects"
	@$(NVCC) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(INC) | $(OBJ_DIR)
	@printf "%-12s %-25s -> %s\n" "Compiling" "$<" "$@"
	@$(NVCC) --device-c -o $@ $< -I $(INC_DIR)

$(OBJ_DIR):
	@printf "%-12s %s\n" "Creating" "$@"
	@mkdir -p $(OBJ_DIR)

clean:
	@printf "%-12s %s\n" "Cleaning" "$(EXEC)"
	@rm -f $(EXEC)
	@printf "%-12s %s\n" "Cleaning" "object files"
	@rm -rf $(OBJ_DIR)/*

cleanall: clean
	@printf "%-12s %s\n" "Cleaning" "output files"
	@rm -rf $(OUT_DIR)/*