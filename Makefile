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

# RADIATION: Enable radiation pressure calculations (optical depth, beta)
# NVCC += -DRADIATION

# DIFFUSION: Enable turbulent diffusion of dust particles
NVCC += -DDIFFUSION

# COLLISION: Enable dust collision and coagulation/fragmentation
# NVCC += -DCOLLISION

# CONST_NU: Use constant kinematic viscosity NU instead of alpha-viscosity
# NVCC += -DCONST_NU

# CONST_ST: Use constant Stokes number instead of constant physical size for dust particles
# NVCC += -DCONST_ST

# =========================================================================================================================

EXEC = cuDust

_INC = const.cuh cudust.cuh
INC = $(patsubst %, $(INC_DIR)/%, $(_INC))

_OBJ = collision.o diffusion.o fileoutput.o gridfields.o helperfunc.o initializer.o integrator.o interpolator.o main.o randprofile.o
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