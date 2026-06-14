NVCC      := nvcc
TARGET    := rsort
ARCH      := sm_86
STD       := c++17
OPT       := -O3

SOURCES   := radix_gpu.cu bench_parser.cpp

NVCCFLAGS := $(OPT) -std=$(STD) -arch=$(ARCH)

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean