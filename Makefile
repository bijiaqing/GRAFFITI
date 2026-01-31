# Project root directory (absolute path where Makefile is located)
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

INC_DIR = $(ROOT_DIR)/inc
MOD_DIR = $(ROOT_DIR)/mod
OBJ_DIR = $(ROOT_DIR)/obj
OUT_DIR = $(ROOT_DIR)/out
SRC_DIR = $(ROOT_DIR)/src

# Allow clean targets to work without MODEL
ifneq ($(MAKECMDGOALS),clean)
ifndef MODEL
$(error MODEL is not defined. Usage: make MODEL=model_name)
endif
endif

ifdef MODEL
MOD_DIR := $(MOD_DIR)/$(MODEL)
OUT_DIR := $(OUT_DIR)/$(MODEL)
OBJ_DIR := $(OBJ_DIR)/$(MODEL)  # Model-specific to avoid stale objects
endif # MODEL argument exists

# NVCC base configuration (platform-specific flags)
NVCC  = nvcc
NVCC += -arch=sm_80
NVCC += --use_fast_math
NVCC += --diag-suppress 177,550

ifdef MODEL
# Check if model directory exists, error if not
ifeq ($(wildcard $(MOD_DIR)),)
$(error Model directory $(MOD_DIR) does not exist. Please create it first.)
endif # MOD_DIR exists

# Include model-specific compilation flags and/or extra files if flags.mk exists
-include $(MOD_DIR)/flags.mk

$(info Using model setup: $(MODEL) from $(MOD_DIR))

EXEC = $(MOD_DIR)/cuDust
endif # MODEL argument exists

_INC =						\
	const.cuh				\
	cudust_host.cuh			\
	cudust_kern.cuh			\
	helpers_collision.cuh	\
	helpers_diffusion.cuh	\
	helpers_interpval.cuh	\
	helpers_paramgrid.cuh	\
	helpers_paramphys.cuh	\
	helpers_scatfield.cuh	\
	helpers_transport.cuh

_OBJ =				\
	col_flag_calc.o	\
	col_proc_exec.o	\
	col_rate_calc.o	\
	col_rate_init.o	\
	col_tree_init.o	\
	diffusion_pos.o	\
	diffusion_vel.o	\
	dustdens_calc.o	\
	dustdens_init.o	\
	dustdens_scat.o	\
	main.o			\
	optdepth_calc.o	\
	optdepth_csum.o	\
	optdepth_init.o	\
	optdepth_mean.o	\
	optdepth_scat.o	\
	particle_init.o	\
	rs_grids_init.o	\
	rs_swarm_init.o	\
	ssa_substep_1.o	\
	ssa_substep_2.o	\
	ssa_transport.o

# Append model-specific files if defined in flags.mk
_INC += $(_INC_MOD)
_OBJ += $(_OBJ_MOD)

INC = $(patsubst %, $(INC_DIR)/%, $(_INC))
OBJ = $(patsubst %, $(OBJ_DIR)/%, $(_OBJ))

# Function to find source file: use setup override if exists, otherwise use src/
find-source = $(firstword $(wildcard $(MOD_DIR)/$(notdir $(1))) $(1))

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJ)
	@printf "%-12s %-20s %s\n" "Linking" "$@" "from $(words $(OBJ)) objects"
	@$(NVCC) -o $@ $^

$(OBJ_DIR)/%.o: $(INC) | $(OBJ_DIR)
	$(eval SOURCE := $(call find-source,$(SRC_DIR)/$*.cu))
	$(eval PREREQ := $(SOURCE))
	@printf "%-12s %-35s -> %s\n" "Compiling" "$(patsubst $(ROOT_DIR)/%,%,$(SOURCE))" "$@"
	@$(NVCC) --device-c -o $@ $(SOURCE) -I $(INC_DIR) \
		$(if $(wildcard $(MOD_DIR)),-I $(MOD_DIR),) \
		-DPATH_OUT=\"$(abspath $(OUT_DIR))/\" \
		-MMD -MP -MF $(OBJ_DIR)/$*.d

# Declare actual source dependency (Make will check timestamp of correct file)
$(OBJ_DIR)/%.o: $$(call find-source,$$(SRC_DIR)/$$*.cu)

# Include auto-generated dependency files (tracks all headers each .cu includes)
-include $(OBJ:.o=.d)

$(OBJ_DIR):
	@printf "%-12s %s\n" "Creating" "$@"
	@mkdir -p $(OBJ_DIR)

clean:
ifdef MODEL
	@printf "%-12s %s\n" "Cleaning" "$(EXEC)"
	@rm -f $(EXEC)
	@printf "%-12s %s\n" "Cleaning" "object files"
	@rm -rf $(OBJ_DIR)
else
	@printf "%-12s %s\n" "Cleaning" "all object files"
	@rm -rf $(OBJ_DIR)/*
endif