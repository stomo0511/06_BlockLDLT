UNAME = $(shell uname)
ifeq ($(UNAME),Linux)
	# CXX = g++-7
	CXX = icpc
	CXXFLAGS = -m64 -openmp -O3
	LIB_DIR = /opt/intel/compilers_and_libraries/linux/lib/intel64
	LIBS = -pthread -lm -ldl
	MKL_LIB_DIR = $(MKLROOT)/lib/intel64
endif
ifeq ($(UNAME),Darwin)
	CXX = /usr/local/bin/g++-9
	CXXFLAGS = -m64 -fopenmp -O3
	LIB_DIR = /opt/intel/compilers_and_libraries/mac/lib
	LIBS = -pthread -lm -ldl
	MKL_LIB_DIR = $(MKLROOT)/lib
endif

LIBS = -liomp5
MKL_INC_DIR = $(MKLROOT)/include
MKL_LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core 

OBJS =	BlockLDLT.o

TARGET = BlockLDLT

all:	$(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -L$(LIB_DIR) $(LIBS) -L$(MKL_LIB_DIR) $(MKL_LIBS) 

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -I$(MKL_INC_DIR) -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
