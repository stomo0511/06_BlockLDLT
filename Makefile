UNAME = $(shell uname)
ifeq ($(UNAME),Linux)
	CXX = g++-9
	LIB_DIR = /opt/intel/compilers_and_libraries/linux/lib/intel64
	LIBS = -pthread -lm -ldl
	MKL_ROOT = /opt/intel/compilers_and_libraries/linux/mkl
	MKL_LIB_DIR = $(MKL_ROOT)/lib/intel64
endif
ifeq ($(UNAME),Darwin)
	CXX = /usr/local/bin/g++-9
	LIB_DIR = /opt/intel/compilers_and_libraries/mac/lib
	MKL_ROOT = /opt/intel/compilers_and_libraries/mac/mkl
	MKL_LIB_DIR = $(MKL_ROOT)/lib
endif

CXXFLAGS = -m64 -fopenmp -O3

LIBS = -liomp5
MKL_INC_DIR = $(MKL_ROOT)/include
MKL_LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core 

OBJS =	BlockLDLT.o
#OBJS =	BlockLDLT.o trace.o

TARGET = BlockLDLT

all:	$(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -L$(LIB_DIR) -L$(MKL_LIB_DIR) $(MKL_LIBS) $(LIBS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -I$(MKL_INC_DIR) -o $@ $<

trace.o: trace.c
	$(CXX) -O3 -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
