NVCC      := nvcc
ARCH      := sm_86
STD       := c++17
OPT       := -O3

ROOT      := ..
INCLUDES  := -I$(ROOT)/include

ifeq ($(OS),Windows_NT)
    EXE_EXT := .exe
else
    EXE_EXT :=
endif

RSORT_EXE := rsort$(EXE_EXT)
CUB_EXE   := cub_radix$(EXE_EXT)
CUB_D_EXE := cub_radix_double$(EXE_EXT)

RSORT_SRC := radix_gpu.cu bench_parser.cpp
CUB_SRC   := cub_radix.cu bench_parser.cpp
CUB_D_SRC := cub_radix_double.cu bench_parser.cpp

NVCCFLAGS := $(OPT) -std=$(STD) -arch=$(ARCH) $(INCLUDES)

all: build cub cub_d

build: $(RSORT_EXE)

cub: $(CUB_EXE)

cub_d: $(CUB_D_EXE)

$(RSORT_EXE): $(RSORT_SRC)
	$(NVCC) $(NVCCFLAGS) $(RSORT_SRC) -o $(RSORT_EXE)

$(CUB_EXE): $(CUB_SRC)
	$(NVCC) $(NVCCFLAGS) $(CUB_SRC) -o $(CUB_EXE)

$(CUB_D_EXE): $(CUB_D_SRC)
	$(NVCC) $(NVCCFLAGS) $(CUB_D_SRC) -o $(CUB_D_EXE)

clean:
	rm -f rsort rsort.exe cub_radix cub_radix.exe cub_radix_double cub_radix_double.exe *.o *.obj

.PHONY: all build cub cub_d clean
