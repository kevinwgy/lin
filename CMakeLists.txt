cmake_minimum_required(VERSION 3.10)

# set the project name
project(M2CLinearSolver VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# use mpicc and mpicxx to compile
SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

# compiler flags (turn on all the warnings)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-reorder")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# folders for executable and libraries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

#--------------------------------------------------------
# find packages 
if(NOT DEFINED ENV{PETSC_DIR})
  message(FATAL_ERROR "ERROR: Could not find Petsc...")
endif()
#--------------------------------------------------------

# add include and link directories 
include_directories("${CMAKE_SOURCE_DIR}")
#include_directories(${CMAKE_SOURCE_DIR}/GeoTools)
#include_directories(${CMAKE_SOURCE_DIR}/MathTools)
link_directories(${CMAKE_SOURCE_DIR}/parser)

# compile the parser
add_custom_target(
   extern_lib
   COMMAND $(MAKE)
   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/parser
)

#-----------------------------------------
# Special tag(s). Don't turn on unless you know what they mean.
#
add_definitions(-DLINEAR_SOLVER_TEST=2)
#-----------------------------------------

# add the executable
add_executable(m2c_linsolve
Main.cpp
IoData.cpp
SpaceVariable.cpp
MeshGenerator.cpp
GlobalMeshInfo.cpp
GradientCalculatorCentral.cpp
GradientCalculatorFD3.cpp
Interpolator.cpp
SpaceOperatorLite.cpp
LinearOperator.cpp
LinearSystemSolver.cpp
Utils.cpp)

# link to libraries
target_link_libraries(m2c_linsolve petsc mpi parser)
add_dependencies(m2c_linsolve extern_lib)
