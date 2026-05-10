# ---------------------------------------------------------------------------
# LLVM pass plugin
# ---------------------------------------------------------------------------
if(MPI_PROGRESS_BUILD_PASS)
  set(MPI_PROGRESS_PASS_SOURCES
    mpi_progress_pass.cpp
  )

  add_library(MPIProgress MODULE
    ${MPI_PROGRESS_PASS_SOURCES}
  )

  target_compile_features(MPIProgress PRIVATE cxx_std_17)
  set_target_properties(MPIProgress PROPERTIES
    PREFIX ""
    OUTPUT_NAME "MPIProgress"
    POSITION_INDEPENDENT_CODE ON
  )

  # Match LLVM RTTI setting if needed
  if(DEFINED LLVM_ENABLE_RTTI AND NOT LLVM_ENABLE_RTTI)
    if(MSVC)
      target_compile_options(MPIProgress PRIVATE /GR-)
    else()
      target_compile_options(MPIProgress PRIVATE -fno-rtti)
    endif()
  endif()

  # Link the LLVM components needed by the pass
  llvm_map_components_to_libnames(MPI_PROGRESS_LLVM_LIBS
    Core
    Passes
    Analysis
    Support
    TransformUtils
    ScalarOpts
  )

  target_link_libraries(MPIProgress PRIVATE
    ${MPI_PROGRESS_LLVM_LIBS}
  )

  install(TARGETS MPIProgress
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
endif()

# ---------------------------------------------------------------------------
# Runtime helper library
# ---------------------------------------------------------------------------
if(MPI_PROGRESS_BUILD_RUNTIME)
  set(MPIPROG_MAX_ACTIVE 8192 CACHE STRING
      "Maximum number of tracked active MPI requests")
  set(MPIPROG_MAX_COMM_SNAPSHOT 16 CACHE STRING
      "Maximum number of communicators sampled per poll")

  add_library(mpiprogress SHARED
    mpi_progress_runtime.c
  )

  target_compile_features(mpiprogress PRIVATE c_std_11)
  target_link_libraries(mpiprogress PRIVATE
    MPI::MPI_C
    Threads::Threads
  )

  target_compile_definitions(mpiprogress PRIVATE
    MPIPROG_MAX_ACTIVE=${MPIPROG_MAX_ACTIVE}
    MPIPROG_MAX_COMM_SNAPSHOT=${MPIPROG_MAX_COMM_SNAPSHOT}
  )

  set_target_properties(mpiprogress PROPERTIES
    OUTPUT_NAME "mpiprogress"
    POSITION_INDEPENDENT_CODE ON
  )

  install(TARGETS mpiprogress
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
  )
endif()

