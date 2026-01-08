IDIR = ./include
ODIR = ./object
SDIR = ./source

NVCC = nvcc
# sm_80 is for A100 GPU
NVCC += -arch=sm_80
# use fast math operations, which are less precise but faster
NVCC += --use_fast_math
# suppress some warnings
# 177: variable was declared but never referenced
# 550: variable was set but never used
NVCC += --diag-suppress 177,550

EXEC = cuDust

_DEP = const.cuh cudust.cuh
DEP = $(patsubst %, $(IDIR)/%, $(_DEP))

_OBJ = collision.o initialize.o integrator.o interpolate.o main.o mesh.o profiles.o 
OBJ = $(patsubst %, $(ODIR)/%, $(_OBJ))

.PHONY: all clean cleanall

all: $(EXEC)

$(EXEC): $(OBJ)
	@printf "%-12s %-25s %s\n" "Linking" "$@" "from $(words $(OBJ)) objects"
	@$(NVCC) -o $@ $^

$(ODIR)/%.o: $(SDIR)/%.cu $(DEP) | $(ODIR)
	@printf "%-12s %-25s -> %s\n" "Compiling" "$<" "$@"
	@$(NVCC) --device-c -o $@ $< -I $(IDIR)

$(ODIR):
	@printf "%-12s %s\n" "Creating" "$@"
	@mkdir -p $(ODIR)

clean:
	@printf "%-12s %s\n" "Cleaning" "$(EXEC) and object files"
	@rm -f $(EXEC) $(OBJ)

cleanall: clean
	@printf "%-12s %s\n" "Removing" "outputs directory"
	@rm -rf outputs/