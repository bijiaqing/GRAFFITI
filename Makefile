IDIR = ./include
ODIR = ./object
SDIR = ./source

EXEC = cuDust

CC = nvcc -arch=sm_80
CFLAGS = -I $(IDIR)

_DEPS = const.cuh cudust.cuh
DEPS = $(patsubst %, $(IDIR)/%, $(_DEPS))

_OBJ = collision.o initialize.o integrator.o interpolate.o main.o mesh.o profiles.o 
OBJ = $(patsubst %, $(ODIR)/%, $(_OBJ))

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) -o $@ $^

$(ODIR)/%.o: $(SDIR)/%.cu $(DEPS) | $(ODIR)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR):
	mkdir -p $(ODIR)

clean:
	rm -f $(EXEC) $(OBJ)