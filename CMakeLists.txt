cmake_minimum_required(VERSION 3.20)

project(MPIProgress
  VERSION 0.3
  LANGUAGES C CXX
)

option(MPI_PROGRESS_BUILD_PASS            "Build LLVM pass plugin" ON)
option(MPI_PROGRESS_BUILD_RUNTIME         "Build helper runtime library" ON)
option(MPI_PROGRESS_BUILD_C_WRAPPER       "Generate C wrapper" ON)
option(MPI_PROGRESS_BUILD_CXX_WRAPPER     "Generate C++ wrapper" ON)
option(MPI_PROGRESS_BUILD_FORTRAN_WRAPPER "Generate Fortran wrapper" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Enable Fortran only if a Fortran wrapper is requested.
if(MPI_PROGRESS_BUILD_FORTRAN_WRAPPER)
  enable_language(Fortran)
endif()

if(MPI_PROGRESS_BUILD_PASS)
  find_package(LLVM REQUIRED CONFIG)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake from: ${LLVM_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
endif()

# MPI components needed by runtime / wrappers
set(MPI_REQUIRED_COMPONENTS)
if(MPI_PROGRESS_BUILD_RUNTIME OR MPI_PROGRESS_BUILD_C_WRAPPER)
  list(APPEND MPI_REQUIRED_COMPONENTS C)
endif()
if(MPI_PROGRESS_BUILD_CXX_WRAPPER)
  list(APPEND MPI_REQUIRED_COMPONENTS CXX)
endif()
if(MPI_PROGRESS_BUILD_FORTRAN_WRAPPER)
  list(APPEND MPI_REQUIRED_COMPONENTS Fortran)
endif()

if(MPI_REQUIRED_COMPONENTS)
  list(REMOVE_DUPLICATES MPI_REQUIRED_COMPONENTS)
  find_package(MPI REQUIRED COMPONENTS ${MPI_REQUIRED_COMPONENTS})
endif()

if(MPI_PROGRESS_BUILD_RUNTIME)
  find_package(Threads REQUIRED)
endif()

# Wrappers require both the pass and the runtime.
if(MPI_PROGRESS_BUILD_C_WRAPPER
   OR MPI_PROGRESS_BUILD_CXX_WRAPPER
   OR MPI_PROGRESS_BUILD_FORTRAN_WRAPPER)

  if(NOT MPI_PROGRESS_BUILD_PASS)
    message(FATAL_ERROR
      "Compiler wrappers require MPI_PROGRESS_BUILD_PASS=ON")
  endif()

  if(NOT MPI_PROGRESS_BUILD_RUNTIME)
    message(FATAL_ERROR
      "Compiler wrappers require MPI_PROGRESS_BUILD_RUNTIME=ON")
  endif()
endif()

# Frontend compiler selection for wrappers
if(MPI_PROGRESS_BUILD_C_WRAPPER)
  if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(MPI_PROGRESS_CLANG "${CMAKE_C_COMPILER}")
  else()
    find_program(MPI_PROGRESS_CLANG NAMES clang REQUIRED)
  endif()
  message(STATUS "C wrapper frontend: ${MPI_PROGRESS_CLANG}")
endif()

if(MPI_PROGRESS_BUILD_CXX_WRAPPER)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(MPI_PROGRESS_CLANGXX "${CMAKE_CXX_COMPILER}")
  else()
    find_program(MPI_PROGRESS_CLANGXX NAMES clang++ REQUIRED)
  endif()
  message(STATUS "C++ wrapper frontend: ${MPI_PROGRESS_CLANGXX}")
endif()

if(MPI_PROGRESS_BUILD_FORTRAN_WRAPPER)
  # Prefer the configured Fortran compiler, but require it to be flang-family.
  if(CMAKE_Fortran_COMPILER_ID MATCHES "LLVMFlang|Flang")
    set(MPI_PROGRESS_FLANG "${CMAKE_Fortran_COMPILER}")
  else()
    message(FATAL_ERROR
      "Fortran wrapper requires flang/flang-new as CMAKE_Fortran_COMPILER.\n"
      "Please reconfigure with e.g.:\n"
      "  -DCMAKE_Fortran_COMPILER=flang-new")
  endif()
  message(STATUS "Fortran wrapper frontend: ${MPI_PROGRESS_FLANG}")
endif()

add_subdirectory(src)

