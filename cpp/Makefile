# Declaration of variables
#
CUDA_HOME="/usr/local/cuda-10.0"

#Use ‘nvcc –Xptxas –v a.cu’ to get register and shared memory usage.


CC = g++ -std=c++11
NVCC = $(CUDA_HOME)/bin/nvcc
CC_FLAGS += -O3 -fPIC -Wall -Wfatal-errors -D_FORCE_INLINES -rdynamic
NVCC_FLAGS := -std=c++11 -arch=sm_61 --ptxas-options=-v -c --compiler-options -fPIC --default-stream per-thread
 
TF_INCLUDE="include"
NUMPY_INCLUDE=`python3 -c "import numpy;print(numpy.get_include())"`
PYTHON_INCLUDE="/usr/include/python3.6"
PYTHON_LIB="/usr/lib/python3.6"
INC = -I$(TF_INCLUDE) -I$(NUMPY_INCLUDE) -I$(PYTHON_INCLUDE)
INC_CU = -I$(TF_INCLUDE) -I$(CUDA_HOME)/include -I./../externals/cub
CUDA_LIBS = -L$(CUDA_HOME)/targets/x86_64-linux/lib -lcudart -lcublas -lcurand


# File names
SRCDIR := src
BUILDDIR := build
#LIBTENSOR = libtensor
#LIBPARSER = libparser
LIBIO = libio
LIBBINOPS = libbinops

#LIBTENSOR_SO = $(BUILDDIR)/$(LIBTENSOR).so
#LIBPARSER_SO = $(BUILDDIR)/$(LIBPARSER).so

LIBIO_SO = $(BUILDDIR)/$(LIBIO).so
LIBBINOPS_SO = $(BUILDDIR)/$(LIBBINOPS).so

SRCEXT := cpp
SRCEXT_CU := cu

#LIBTENSOR
#LIBTENSOR_BUILD = $(shell grep "\.$(SRCEXT)$$" $(LIBTENSOR).build)
#LIBTENSOR_OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(LIBTENSOR_BUILD:.$(SRCEXT)=.o))
#LIBTENSOR_SOURCES_CU = $(shell grep "\.$(SRCEXT_CU)$$" $(LIBTENSOR).build)
#LIBTENSOR_OBJECTS_CU := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(LIBTENSOR_SOURCES_CU:.$(SRCEXT_CU)=.co))
#$(LIBTENSOR): $(LIBTENSOR_OBJECTS) $(LIBTENSOR_OBJECTS_CU)
#	$(CC) -shared -o $(LIBTENSOR_SO) $(CC_FLAGS) $(LIBTENSOR_OBJECTS) $(LIBTENSOR_OBJECTS_CU) $(CUDA_LIBS)


#LIBIO
LIBIO_BUILD = $(shell grep "\.$(SRCEXT)$$" $(LIBIO).build)
LIBIO_OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(LIBIO_BUILD:.$(SRCEXT)=.o))
LIBIO_SOURCES_CU = $(shell grep "\.$(SRCEXT_CU)$$" $(LIBIO).build)
LIBIO_OBJECTS_CU := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(LIBIO_SOURCES_CU:.$(SRCEXT_CU)=.co))
$(LIBIO): $(LIBIO_OBJECTS) $(LIBIO_OBJECTS_CU)
	$(CC) -shared -o $(LIBIO_SO) $(CC_FLAGS) $(LIBIO_OBJECTS) $(LIBIO_OBJECTS_CU) $(CUDA_LIBS)


#LIBBINOPS
LIBBINOPS_BUILD = $(shell grep "\.$(SRCEXT)$$" $(LIBBINOPS).build)
LIBBINOPS_OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(LIBBINOPS_BUILD:.$(SRCEXT)=.o))
LIBBINOPS_SOURCES_CU = $(shell grep "\.$(SRCEXT_CU)$$" $(LIBBINOPS).build)
LIBBINOPS_OBJECTS_CU := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(LIBBINOPS_SOURCES_CU:.$(SRCEXT_CU)=.co))
$(LIBBINOPS): $(LIBBINOPS_OBJECTS) $(LIBBINOPS_OBJECTS_CU)
	$(CC) -shared -o $(LIBBINOPS_SO) $(CC_FLAGS) $(LIBBINOPS_OBJECTS) $(LIBBINOPS_OBJECTS_CU) $(CUDA_LIBS)



##LIBPARSER
#LIBPARSER_BUILD = $(shell grep "\.$(SRCEXT)$$" $(LIBPARSER).build)
#LIBPARSER_OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(LIBPARSER_BUILD:.$(SRCEXT)=.o))
#$(LIBPARSER): $(LIBPARSER_OBJECTS)
#	$(CC) -shared -o $(LIBPARSER_SO) $(CC_FLAGS)  $(LIBPARSER_OBJECTS)



# BUILD CPP OBJECT FILES
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(@D)
	$(CC) $(INC) -c $(CC_FLAGS) $< -o $@ 

$(BUILDDIR)/%.co: $(SRCDIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(INC_CU) -c $(NVCC_FLAGS) $< -o $@ 
 
clean:
	#rm -f $(LIBTENSOR_SO) $(LIBTENSOR_OBJECTS) 
	#rm -f $(LIBPARSER_SO) $(LIBPARSER_OBJECTS) $(LIBTENSOR_OBJECTS_CU)
	rm -f $(LIBIO_SO) $(LIBIO_OBJECTS) $(LIBIO_OBJECTS_CU)
	rm -f $(LIBBINOPS_SO) $(LIBBINOPS_OBJECTS) $(LIBBINOPS_OBJECTS_CU)

