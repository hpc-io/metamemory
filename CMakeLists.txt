cmake_minimum_required(VERSION 3.21.2)

project(metamemory)
include_directories(./include)

################################################
# BUILD ctests
enable_testing()

################################################
file (GLOB METAMEM_API_POSIX_SRCS
  ./include/*.h
  ./src/*.c

  ./src/posix/*.c
)

add_library(posix_metamemapi SHARED ${METAMEM_API_POSIX_SRCS})
##################################################
# posix tests
add_executable(posix_00_vector_add.exe ./tests/posix/00_vector_add.c)
add_test( posix_00_vector_add posix_00_vector_add.exe 1048576 )

add_executable(posix_01_vector_add.exe ./tests/posix/01_vector_add.c)
target_link_libraries(posix_01_vector_add.exe PUBLIC posix_metamemapi)
add_test( posix_01_vector_add posix_01_vector_add.exe 1048576 )

add_executable(posix_02_vector_add.exe ./tests/posix/02_vector_add.c)
target_link_libraries(posix_02_vector_add.exe PUBLIC posix_metamemapi)
add_test( posix_02_vector_add posix_02_vector_add.exe 1048576 )

add_executable(posix_03_vector_add.exe ./tests/posix/03_vector_add.c)
target_link_libraries(posix_03_vector_add.exe PUBLIC posix_metamemapi)
add_test( posix_03_vector_add posix_03_vector_add.exe 1048576 )
##################################################

##################################################
set(METAMEM_CUDA_SUPPORT OFF)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  set(CMAKE_CUDA_STANDARD 11)
  enable_language(CUDA)
  set(METAMEM_CUDA_SUPPORT ON)
endif(CUDAToolkit_FOUND)

message(STATUS "METAMEM_CUDA_SUPPORT=" ${METAMEM_CUDA_SUPPORT})

if(METAMEM_CUDA_SUPPORT)
  file (GLOB METAMEM_API_CUDA_SRCS
    ./include/*.h
    ./src/*.c

    ./src/cuda/*.c
  )

  add_library(cuda_metamemapi SHARED ${METAMEM_API_CUDA_SRCS})
  set_target_properties(cuda_metamemapi PROPERTIES COMPILE_DEFINITIONS METAMEM_USE_CUDA)
  target_include_directories(cuda_metamemapi PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

  # cuda tests
  add_executable(cuda_00_vector_add.exe ./tests/cuda/00_vector_add.cu)
  target_include_directories(cuda_00_vector_add.exe PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(cuda_00_vector_add.exe PUBLIC CUDA::cudart)
  add_test( cuda_00_vector_add cuda_00_vector_add.exe 1048576 )

  add_executable(cuda_01_vector_add.exe ./tests/cuda/01_vector_add.cu)
  target_include_directories(cuda_01_vector_add.exe PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(cuda_01_vector_add.exe PUBLIC CUDA::cudart)
  target_link_libraries(cuda_01_vector_add.exe PUBLIC cuda_metamemapi)
  add_test( cuda_01_vector_add cuda_01_vector_add.exe 1048576 )

  add_executable(cuda_02_vector_add.exe ./tests/cuda/02_vector_add.cu)
  target_include_directories(cuda_02_vector_add.exe PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(cuda_02_vector_add.exe PUBLIC CUDA::cudart)
  target_link_libraries(cuda_02_vector_add.exe PUBLIC cuda_metamemapi)
  add_test( cuda_02_vector_add cuda_02_vector_add.exe 1048576 )

  add_executable(cuda_03_vector_add.exe ./tests/cuda/03_vector_add.cu)
  target_include_directories(cuda_03_vector_add.exe PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(cuda_03_vector_add.exe PUBLIC CUDA::cudart)
  target_link_libraries(cuda_03_vector_add.exe PUBLIC cuda_metamemapi)
  add_test( cuda_03_vector_add cuda_03_vector_add.exe 1048576 )
endif(METAMEM_CUDA_SUPPORT)
##################################################

##################################################
# TODO: hip support
set(METAMEM_HIP_SUPPORT OFF)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

# Find hip
find_package(HIP)
if(HIP_VERSION)
  enable_language(HIP)
  set(METAMEM_HIP_SUPPORT ON)
  set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_C_COMPILER amdclang) # TODO: hardcoded c compiler for now..
endif(HIP_VERSION)

message(STATUS "METAMEM_HIP_SUPPORT=" ${METAMEM_HIP_SUPPORT})

if(METAMEM_HIP_SUPPORT)
  file (GLOB METAMEM_API_HIP_SRCS
    ./include/*.h
    ./src/*.c

    ./src/hip/*.c
  )

  add_library(hip_metamemapi SHARED ${METAMEM_API_HIP_SRCS})
  set_target_properties(hip_metamemapi PROPERTIES COMPILE_DEFINITIONS METAMEM_USE_HIP)
  target_include_directories(hip_metamemapi PRIVATE ${HIP_ROOT_DIR}/include)
  target_link_libraries(hip_metamemapi hip::host)

  # hip tests
  add_executable(hip_00_vector_add.exe ./tests/hip/00_vector_add.cpp)
  target_include_directories(hip_00_vector_add.exe PRIVATE ${HIP_ROOT_DIR}/include)
  add_test( hip_00_vector_add hip_00_vector_add.exe 1048576 )
  target_link_libraries(hip_00_vector_add.exe hip::host)

  # hip tests
  add_executable(hip_01_vector_add.exe ./tests/hip/01_vector_add.cpp)
  target_include_directories(hip_01_vector_add.exe PRIVATE ${HIP_ROOT_DIR}/include)
  target_link_libraries(hip_01_vector_add.exe hip_metamemapi hip::host)
  add_test( hip_01_vector_add hip_01_vector_add.exe 1048576 )
endif(METAMEM_HIP_SUPPORT)
##################################################

set(CMAKE_CUDA_ARCHITECTURES 30 52)
# set(CMAKE_CUDA_ARCHITECTURES 52 70 72)


