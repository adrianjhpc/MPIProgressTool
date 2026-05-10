cmake_minimum_required(VERSION 3.20)

project(MPIProgress
  VERSION 0.1
  LANGUAGES C CXX
)

option(MPI_PROGRESS_BUILD_PASS "Build the LLVM pass plugin" ON)
option(MPI_PROGRESS_BUILD_RUNTIME "Build the MPI runtime helper" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Nice default output layout
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(MPI_PROGRESS_BUILD_PASS)
  find_package(LLVM REQUIRED CONFIG)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake from: ${LLVM_DIR}")

  # Make LLVM CMake helpers/macros visible, including llvm_map_components_to_libnames
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  # Apply LLVM compile definitions/includes globally for the pass target(s)
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
endif()

if(MPI_PROGRESS_BUILD_RUNTIME)
  find_package(MPI REQUIRED COMPONENTS C)
  find_package(Threads REQUIRED)
endif()

add_subdirectory(src)

