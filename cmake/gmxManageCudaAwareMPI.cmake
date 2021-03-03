#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2009,2011,2012,2014,2015 by the GROMACS development team.
# Copyright (c) 2016,2020,2021, by the GROMACS development team, led by
# Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
# and including many others, as listed in the AUTHORS file in the
# top-level source directory and at http://www.gromacs.org.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at http://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out http://www.gromacs.org.

# - Define macro to check if underlying MPI is CUDA-aware
#
#  GMX_TEST_CUDA_AWARE_MPI(VARIABLE)
#
#  VARIABLE will be set to 0 (Not CUDA-aware), 1(CUDA-aware), 2(Cannot detect)
#

include(CheckCSourceCompiles)
MACRO(GMX_TEST_CUDA_AWARE_MPI VARIABLE)
  if(NOT DEFINED CUDA_AWARE_MPI_COMPILE_OK)
    MESSAGE(STATUS "Checking for CUDA_AWARE_MPI")
    list(JOIN MPI_COMPILE_FLAGS " " CMAKE_REQUIRED_FLAGS)
    set(CMAKE_REQUIRED_INCLUDES ${MPI_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${MPI_LIBRARIES})
    find_file(HAVE_MPIEXT mpi-ext.h ${MPI_INCLUDE_PATH} NO_DEFAULT_PATH)
    MESSAGE(STATUS "Found mpi-ext.h: ${HAVE_MPIEXT}")
    if(NOT HAVE_MPIEXT)
      MESSAGE(STATUS "Checking for CUDA_AWARE_MPI - not known")
      set(CUDA_AWARE_MPI_COMPILE_OK "2" CACHE INTERNAL "Result of cuda_aware_mpi check")
    else()
      check_cxx_source_compiles(
       "#include <mpi.h>
        #include <mpi-ext.h>
        int main(void) 
        {
        #if defined(MPIX_CUDA_AWARE_SUPPORT) && (MPIX_CUDA_AWARE_SUPPORT==1)
          return 0;
        #else
        #error MPI implementation isn't CUDA-aware
        #endif
        }" CUDA_AWARE_MPI_COMPILE_OK)

      if(CUDA_AWARE_MPI_COMPILE_OK)
        MESSAGE(STATUS "Checking for CUDA_AWARE_MPI - yes")
        set(CUDA_AWARE_MPI_COMPILE_OK "1" CACHE INTERNAL "Result of cuda_aware_mpi check")
      else()
        MESSAGE(STATUS "Checking for CUDA_AWARE_MPI - no")
        set(CUDA_AWARE_MPI_COMPILE_OK "0" CACHE INTERNAL "Result of cuda_aware_mpi check")
      endif()
    endif()
    set(CMAKE_REQUIRED_FLAGS)
    set(CMAKE_REQUIRED_INCLUDES)
    set(CMAKE_REQUIRED_LIBRARIES)
  endif()
  if (CUDA_AWARE_MPI_COMPILE_OK)
    set(${VARIABLE} ${CUDA_AWARE_MPI_COMPILE_OK}
      "Result of test for CUDA_AWARE_MPI")
  endif()
ENDMACRO(GMX_TEST_CUDA_AWARE_MPI VARIABLE)

# Test if CUDA-aware MPI is supported
gmx_test_cuda_aware_mpi(CUDA_AWARE_MPI_SUPPORTED)
if(CUDA_AWARE_MPI_SUPPORTED EQUAL 1)
  set(HAVE_CUDA_AWARE_MPI 1)
  set(HAVE_MPIEXT_HEADER 1)
elseif(CUDA_AWARE_MPI_SUPPORTED EQUAL 2)
  set(HAVE_CUDA_AWARE_MPI 0)
  set(HAVE_MPIEXT_HEADER 1)
else()
  set(HAVE_CUDA_AWARE_MPI 0)
  set(HAVE_MPIEXT_HEADER 0)
endif()

# Test for and warn about unsuitable MPI versions
#
# Execute the ompi_info binary with the full path of the compiler wrapper
# found, otherwise we run the risk of false positives.
find_file(MPI_INFO_BIN ompi_info
          HINTS ${_mpi_c_compiler_path} ${_mpiexec_path}
                ${_cmake_c_compiler_path} ${_cmake_cxx_compiler_path}
          NO_DEFAULT_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)
if (MPI_INFO_BIN)
  exec_program(${MPI_INFO_BIN}
    ARGS -v ompi full
    OUTPUT_VARIABLE OPENMPI_TYPE
    RETURN_VALUE OPENMPI_EXEC_RETURN)
  if(OPENMPI_EXEC_RETURN EQUAL 0)
    string(REGEX REPLACE ".*Open MPI: \([0-9]+\\.[0-9]*\\.?[0-9]*\).*" "\\1" OPENMPI_VERSION ${OPENMPI_TYPE})
    if(OPENMPI_VERSION VERSION_LESS "1.7.0")
      MESSAGE(WARNING " This OpenMPI version is too old and not CUDA-aware, "
        "for better multi-GPU performance consider using a more recent CUDA-aware MPI.")
    endif()
    unset(OPENMPI_VERSION)
    unset(OPENMPI_TYPE)
    unset(OPENMPI_EXEC_RETURN)
  endif()
endif()
unset(MPI_INFO_BIN CACHE)

# Execute the mpiname binary with the full path of the compiler wrapper
# found, otherwise we run the risk of false positives.
find_file(MPINAME_BIN mpiname
          HINTS ${_mpi_c_compiler_path}
                ${_cmake_c_compiler_path} ${_cmake_cxx_compiler_path}
          NO_DEFAULT_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)
if (MPINAME_BIN)
  exec_program(${MPINAME_BIN}
    ARGS -n -v
    OUTPUT_VARIABLE MVAPICH2_TYPE
    RETURN_VALUE MVAPICH2_EXEC_RETURN)
  if(MVAPICH2_EXEC_RETURN EQUAL 0)
    string(REGEX MATCH "MVAPICH2" MVAPICH2_NAME ${MVAPICH2_TYPE})
    # Want to check for MVAPICH2 in case some other library supplies mpiname
    string(REGEX REPLACE "MVAPICH2 \([0-9]+\\.[0-9]*[a-z]?\\.?[0-9]*\)" "\\1" MVAPICH2_VERSION ${MVAPICH2_TYPE})
    if(${MVAPICH2_NAME} STREQUAL "MVAPICH2" AND MVAPICH2_VERSION VERSION_LESS "1.9")
      MESSAGE(WARNING " This MVAPICH2 version is too old and not CUDA-aware, "
        "for better multi-GPU performance consider using a more recent CUDA-aware MPI.")
    endif()
    unset(MVAPICH2_VERSION)
    unset(MVAPICH2_NAME)
    unset(MVAPICH2_TYPE)
    unset(MVAPICH2_EXEC_RETURN)
  endif()
endif()
unset(MPINAME_BIN CACHE)




