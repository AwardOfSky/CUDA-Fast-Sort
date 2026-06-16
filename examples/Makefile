NVCC      := nvcc
ARCH      := sm_86
STD       := c++17
OPT       := -O3

ROOT      := ..
INCLUDES  := -I$(ROOT)/monolithic

ifeq ($(OS),Windows_NT)
    EXE_EXT := .exe
else
    EXE_EXT :=
endif

EXAMPLE_FULL_EXE  := example_full$(EXE_EXT)
EXAMPLE_BASIC_EXE := example_basic$(EXE_EXT)

EXAMPLE_FULL_SRC  := example_full.cu
EXAMPLE_BASIC_SRC := example_basic.cu

NVCCFLAGS := $(OPT) -std=$(STD) -arch=$(ARCH) $(INCLUDES)
# -diag-suppress 20280 to drop templated __global__ warnings in different TUs for NVCC
EXAMPLE_FULL_NVCCFLAGS := $(NVCCFLAGS) -diag-suppress 20280

all: $(EXAMPLE_BASIC_EXE) $(EXAMPLE_FULL_EXE)

$(EXAMPLE_FULL_EXE): $(EXAMPLE_FULL_SRC) $(ROOT)/monolithic/rsort.cuh
	$(NVCC) $(EXAMPLE_FULL_NVCCFLAGS) $(EXAMPLE_FULL_SRC) -o $(EXAMPLE_FULL_EXE)

$(EXAMPLE_BASIC_EXE): $(EXAMPLE_BASIC_SRC) $(ROOT)/monolithic/rsort.cuh
	$(NVCC) $(NVCCFLAGS) $(EXAMPLE_BASIC_SRC) -o $(EXAMPLE_BASIC_EXE)

clean:
	rm -f example_full example_full.exe example_basic example_basic.exe *.o *.obj

.PHONY: all clean
