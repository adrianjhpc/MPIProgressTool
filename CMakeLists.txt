cmake_minimum_required(VERSION 3.20)

project(MPIAsyncProgress
  VERSION 0.3
  LANGUAGES C CXX
)

option(MPI_PROGRESS_BUILD_PASS "Build LLVM pass plugin" ON)
option(MPI_PROGRESS_BUILD_RUNTIME "Build helper runtime library" ON)
option(MPI_PROGRESS_BUILD_WRAPPER "Generate compiler wrapper script" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(MPI_PROGRESS_BUILD_PASS)
  find_package(LLVM REQUIRED CONFIG)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake from: ${LLVM_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
endif()

if(MPI_PROGRESS_BUILD_RUNTIME OR MPI_PROGRESS_BUILD_WRAPPER)
  find_package(MPI REQUIRED COMPONENTS C)
  find_package(Threads REQUIRED)
endif()

if(MPI_PROGRESS_BUILD_WRAPPER)
  if(NOT MPI_PROGRESS_BUILD_PASS)
    message(FATAL_ERROR "MPI_PROGRESS_BUILD_WRAPPER requires MPI_PROGRESS_BUILD_PASS=ON")
  endif()

  if(NOT MPI_PROGRESS_BUILD_RUNTIME)
    message(FATAL_ERROR "MPI_PROGRESS_BUILD_WRAPPER requires MPI_PROGRESS_BUILD_RUNTIME=ON")
  endif()

  # We want the wrapper to drive Clang, since -fpass-plugin is a Clang feature.
  if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(MPI_PROGRESS_CLANG "${CMAKE_C_COMPILER}")
  else()
    find_program(MPI_PROGRESS_CLANG NAMES clang REQUIRED)
  endif()

  message(STATUS "Wrapper frontend compiler: ${MPI_PROGRESS_CLANG}")
endif()

add_subdirectory(src)

