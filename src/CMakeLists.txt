# --------------------------------------------------------------------
# LLVM pass plugin
# --------------------------------------------------------------------
if(MPI_PROGRESS_BUILD_PASS)
  add_library(MPIProgress MODULE
    mpi_progress_pass.cpp
  )

  target_compile_features(MPIProgress PRIVATE cxx_std_17)

  set_target_properties(MPIProgress PROPERTIES
    PREFIX ""
    OUTPUT_NAME "MPIProgress"
    POSITION_INDEPENDENT_CODE ON
  )

  if(DEFINED LLVM_ENABLE_RTTI AND NOT LLVM_ENABLE_RTTI)
    if(MSVC)
      target_compile_options(MPIProgress PRIVATE /GR-)
    else()
      target_compile_options(MPIProgress PRIVATE -fno-rtti)
    endif()
  endif()

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

# --------------------------------------------------------------------
# Runtime helper
# --------------------------------------------------------------------
if(MPI_PROGRESS_BUILD_RUNTIME)
  add_library(mpiprogress SHARED
    mpi_progress_runtime.c
  )

  target_compile_features(mpiprogress PRIVATE c_std_11)

  target_link_libraries(mpiprogress PRIVATE
    MPI::MPI_C
    Threads::Threads
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

# --------------------------------------------------------------------
# Wrapper generation
# --------------------------------------------------------------------
if(MPI_PROGRESS_BUILD_WRAPPER)
  function(shell_quote_list OUTVAR)
    set(_result "")
    foreach(_arg IN LISTS ARGN)
      string(REPLACE "'" "'\"'\"'" _escaped "${_arg}")
      string(APPEND _result " '${_escaped}'")
    endforeach()
    string(STRIP "${_result}" _result)
    set(${OUTVAR} "${_result}" PARENT_SCOPE)
  endfunction()

  # MPI compile args for plain clang to compile MPI source.
  set(MPIP_C_COMPILE_ARGS)

  foreach(def IN LISTS MPI_C_COMPILE_DEFINITIONS)
    list(APPEND MPIP_C_COMPILE_ARGS "-D${def}")
  endforeach()

  list(APPEND MPIP_C_COMPILE_ARGS ${MPI_C_COMPILE_OPTIONS})

  foreach(inc IN LISTS MPI_C_INCLUDE_DIRS)
    list(APPEND MPIP_C_COMPILE_ARGS "-I${inc}")
  endforeach()

  # MPI link args for plain clang to link MPI binaries.
  set(MPIP_C_LINK_ARGS)

  if(MPI_C_LINK_FLAGS)
    separate_arguments(_MPI_C_LINK_FLAGS_LIST NATIVE_COMMAND "${MPI_C_LINK_FLAGS}")
    list(APPEND MPIP_C_LINK_ARGS ${_MPI_C_LINK_FLAGS_LIST})
  endif()

  list(APPEND MPIP_C_LINK_ARGS ${MPI_C_LIBRARIES})

  shell_quote_list(MPIP_C_COMPILE_ARGS_SH ${MPIP_C_COMPILE_ARGS})
  shell_quote_list(MPIP_C_LINK_ARGS_SH ${MPIP_C_LINK_ARGS})

  set(MPI_PROGRESS_PLUGIN_PATH
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/MPIProgress${CMAKE_SHARED_MODULE_SUFFIX}")
  set(MPI_PROGRESS_RUNTIME_DIR
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

  set(MPIPC_OUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpipc")

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/mpipc.in"
    "${MPIPC_OUT}"
    @ONLY
  )

  file(CHMOD "${MPIPC_OUT}"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )

  install(PROGRAMS "${MPIPC_OUT}" DESTINATION bin)
endif()

