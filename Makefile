UNAME = $(shell uname)
ifeq ($(UNAME),Linux)
	CXX = g++-7
	# CXX = icpc
	CXXFLAGS = -m64 -fopenmp -O3
	# CXXFLAGS = -m64 -openmp -O3
	LIB_DIR = /opt/intel/compilers_and_libraries/linux/lib/intel64
	MKL_ROOT = /opt/intel/compilers_and_libraries/linux/mkl
	MKL_LIB_DIR = $(MKL_ROOT)/lib/intel64
	MY_ROOT = /home/stomo/WorkSpace
	MY_UTIL_DIR = $(MY_ROOT)/00_Utils
endif
ifeq ($(UNAME),Darwin)
	CXX = /usr/local/bin/g++-9
	CXXFLAGS = -m64 -fopenmp -O3
	LIB_DIR = /opt/intel/compilers_and_libraries/mac/lib
	LIBS = -pthread -lm -ldl
	MKL_LIB_DIR = $(MKLROOT)/lib
	MY_ROOT = /Users/stomo/WorkSpace/C++
	MY_UTIL_DIR = $(MY_ROOT)/00_Utils
endif

# CXXFLAGS = -m64 -fopenmp -O3

MKL_INC_DIR = $(MKL_ROOT)/include
MKL_LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core 
LIBS = -liomp5

OBJS =	BlockLDLT.o $(MY_UTIL_DIR)/Utils.o

TARGET = BlockLDLT

all:	$(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -L$(LIB_DIR) $(LIBS) -L$(MKL_LIB_DIR) $(MKL_LIBS) 

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -I$(MKL_INC_DIR)  -I$(MY_UTIL_DIR) -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
