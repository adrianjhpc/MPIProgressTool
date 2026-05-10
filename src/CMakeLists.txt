# --------------------------------------------------------------------
# Helper: shell-quote a CMake list for use inside a bash array
# --------------------------------------------------------------------
function(shell_quote_list OUTVAR)
  set(_result "")
  foreach(_arg IN LISTS ARGN)
    string(REPLACE "'" "'\"'\"'" _escaped "${_arg}")
    string(APPEND _result " '${_escaped}'")
  endforeach()
  string(STRIP "${_result}" _result)
  set(${OUTVAR} "${_result}" PARENT_SCOPE)
endfunction()

# Shared paths used by all wrappers
set(MPI_PROGRESS_PLUGIN_PATH
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/MPIProgress${CMAKE_SHARED_MODULE_SUFFIX}")
set(MPI_PROGRESS_RUNTIME_DIR
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

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
# Runtime helper library
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
# C wrapper: mpipc
# --------------------------------------------------------------------
if(MPI_PROGRESS_BUILD_C_WRAPPER)
  set(MPIP_C_COMPILE_ARGS)
  foreach(def IN LISTS MPI_C_COMPILE_DEFINITIONS)
    list(APPEND MPIP_C_COMPILE_ARGS "-D${def}")
  endforeach()
  list(APPEND MPIP_C_COMPILE_ARGS ${MPI_C_COMPILE_OPTIONS})
  foreach(inc IN LISTS MPI_C_INCLUDE_DIRS)
    list(APPEND MPIP_C_COMPILE_ARGS "-I${inc}")
  endforeach()

  set(MPIP_C_LINK_ARGS)
  if(MPI_C_LINK_FLAGS)
    separate_arguments(_MPI_C_LINK_FLAGS_LIST NATIVE_COMMAND "${MPI_C_LINK_FLAGS}")
    list(APPEND MPIP_C_LINK_ARGS ${_MPI_C_LINK_FLAGS_LIST})
  endif()
  list(APPEND MPIP_C_LINK_ARGS ${MPI_C_LIBRARIES})

  shell_quote_list(MPIP_C_COMPILE_ARGS_SH ${MPIP_C_COMPILE_ARGS})
  shell_quote_list(MPIP_C_LINK_ARGS_SH    ${MPIP_C_LINK_ARGS})

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

# --------------------------------------------------------------------
# C++ wrapper: mpipcxx
# --------------------------------------------------------------------
if(MPI_PROGRESS_BUILD_CXX_WRAPPER)
  set(MPIP_CXX_COMPILE_ARGS)
  foreach(def IN LISTS MPI_CXX_COMPILE_DEFINITIONS)
    list(APPEND MPIP_CXX_COMPILE_ARGS "-D${def}")
  endforeach()
  list(APPEND MPIP_CXX_COMPILE_ARGS ${MPI_CXX_COMPILE_OPTIONS})
  foreach(inc IN LISTS MPI_CXX_INCLUDE_DIRS)
    list(APPEND MPIP_CXX_COMPILE_ARGS "-I${inc}")
  endforeach()

  set(MPIP_CXX_LINK_ARGS)
  if(MPI_CXX_LINK_FLAGS)
    separate_arguments(_MPI_CXX_LINK_FLAGS_LIST NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
    list(APPEND MPIP_CXX_LINK_ARGS ${_MPI_CXX_LINK_FLAGS_LIST})
  endif()
  list(APPEND MPIP_CXX_LINK_ARGS ${MPI_CXX_LIBRARIES})

  shell_quote_list(MPIP_CXX_COMPILE_ARGS_SH ${MPIP_CXX_COMPILE_ARGS})
  shell_quote_list(MPIP_CXX_LINK_ARGS_SH    ${MPIP_CXX_LINK_ARGS})

  set(MPIPXX_OUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpipcxx")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/mpipcxx.in"
    "${MPIPXX_OUT}"
    @ONLY
  )
  file(CHMOD "${MPIPXX_OUT}"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )

  install(PROGRAMS "${MPIPXX_OUT}" DESTINATION bin)
endif()

# --------------------------------------------------------------------
# Fortran wrapper: mpipfortran
# --------------------------------------------------------------------
if(MPI_PROGRESS_BUILD_FORTRAN_WRAPPER)
  set(MPIP_FORTRAN_COMPILE_ARGS)
  foreach(def IN LISTS MPI_Fortran_COMPILE_DEFINITIONS)
    list(APPEND MPIP_FORTRAN_COMPILE_ARGS "-D${def}")
  endforeach()
  list(APPEND MPIP_FORTRAN_COMPILE_ARGS ${MPI_Fortran_COMPILE_OPTIONS})
  foreach(inc IN LISTS MPI_Fortran_INCLUDE_DIRS)
    list(APPEND MPIP_FORTRAN_COMPILE_ARGS "-I${inc}")
  endforeach()

  set(MPIP_FORTRAN_LINK_ARGS)
  if(MPI_Fortran_LINK_FLAGS)
    separate_arguments(_MPI_Fortran_LINK_FLAGS_LIST NATIVE_COMMAND "${MPI_Fortran_LINK_FLAGS}")
    list(APPEND MPIP_FORTRAN_LINK_ARGS ${_MPI_Fortran_LINK_FLAGS_LIST})
  endif()
  list(APPEND MPIP_FORTRAN_LINK_ARGS ${MPI_Fortran_LIBRARIES})

  shell_quote_list(MPIP_FORTRAN_COMPILE_ARGS_SH ${MPIP_FORTRAN_COMPILE_ARGS})
  shell_quote_list(MPIP_FORTRAN_LINK_ARGS_SH    ${MPIP_FORTRAN_LINK_ARGS})

  set(MPIPFORTRAN_OUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpipfortran")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/mpipfortran.in"
    "${MPIPFORTRAN_OUT}"
    @ONLY
  )
  file(CHMOD "${MPIPFORTRAN_OUT}"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )

  install(PROGRAMS "${MPIPFORTRAN_OUT}" DESTINATION bin)
endif()

