NVCC      := nvcc
ARCH      := sm_86
STD       := c++17
OPT       := -O3

ROOT      := ..
INCLUDES  := -I$(ROOT) -I$(ROOT)/benchmark

ifeq ($(OS),Windows_NT)
    EXE_EXT := .exe
else
    EXE_EXT :=
endif

VALIDATE_EXE := validate_mono$(EXE_EXT)

VALIDATE_SRC := validate_mono.cu $(ROOT)/benchmark/bench_parser.cpp

NVCCFLAGS := $(OPT) -std=$(STD) -arch=$(ARCH) $(INCLUDES)

all: $(VALIDATE_EXE)

$(VALIDATE_EXE): $(VALIDATE_SRC)
	$(NVCC) $(NVCCFLAGS) $(VALIDATE_SRC) -o $(VALIDATE_EXE)

clean:
	rm -f validate_mono validate_mono.exe *.o *.obj

.PHONY: all clean
