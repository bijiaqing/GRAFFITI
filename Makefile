IDIR = ./include
ODIR = ./object
SDIR = ./source

CC = nvcc -arch=sm_70
CFLAGS = -I $(IDIR)

_DEPS = const.cuh cudust.cuh
DEPS = $(patsubst %, $(IDIR)/%, $(_DEPS))

_OBJ = collision.o initialize.o integrator.o interpolate.o main.o mesh.o outputs.o profiles.o 
OBJ = $(patsubst %, $(ODIR)/%, $(_OBJ))

all: cuDust

cuDust: $(OBJ)
	$(CC) -o $@ $^

$(ODIR)/collision.o: $(SDIR)/collision.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/initialize.o: $(SDIR)/initialize.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/integrator.o: $(SDIR)/integrator.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/interpolate.o: $(SDIR)/interpolate.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/main.o: $(SDIR)/main.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/mesh.o: $(SDIR)/mesh.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/outputs.o: $(SDIR)/outputs.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/profiles.o: $(SDIR)/profiles.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

clean:
	rm cuDust outputs/* $(OBJ)

