#https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable/tree/master
###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
#CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_INC_DIR =  
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = ./

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = ./

##########################################################

## Make variables ##

# Target executable name:
EXE = run

SRC = main.cpp
SRC_CUDA = MonteCarloGPU.cu
HEADER = basic.h board.h minimax.h ucb.h uct.h util.h
HEADER_CUDA = MonteCarloGPU.cuh
OBJ = $(OBJ_DIR)/main.o
OBJ_CUDA = $(OBJ_DIR)/MonteCarloGPU.o

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJ) $(OBJ_CUDA) $(HEADER) $(HEADER_CUDA)
	$(CC) $(CC_FLAGS) $(OBJ) $(OBJ_CUDA) -o $@  $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(CUDA_INC_DIR)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : SRC
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : SRC_CUDA
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)
