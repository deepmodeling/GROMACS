#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2020, by the GROMACS development team, led by
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

# Bare bones solution to detect an installation of SYCL (Intel DPC++)
# Based on FindHWLOC.cmake, but does not use much from it.

if (NOT SYCL_FOUND)
    set(SYCL_DIR "" CACHE PATH "Installation directory of SYCL library")
endif()

set(ENV_SYCL_DIR "$ENV{SYCL_DIR}")
set(ENV_SYCL_INCDIR "$ENV{SYCL_INCDIR}")
set(ENV_SYCL_LIBDIR "$ENV{SYCL_LIBDIR}")
set(SYCL_GIVEN_BY_USER "FALSE")
if ( SYCL_DIR OR ( SYCL_INCDIR AND SYCL_LIBDIR) OR ENV_SYCL_DIR OR (ENV_SYCL_INCDIR AND ENV_SYCL_LIBDIR) )
    set(SYCL_GIVEN_BY_USER "TRUE")
endif()

if (NOT SYCL_FIND_QUIETLY)
    message(STATUS "Looking for SYCL installation")
endif()

# Looking for include
# -------------------

# Add system include paths to search include
# ------------------------------------------
unset(_inc_env)
if(ENV_SYCL_INCDIR)
    list(APPEND _inc_env "${ENV_SYCL_INCDIR}")
elseif(ENV_SYCL_DIR)
    list(APPEND _inc_env "${ENV_SYCL_DIR}")
    list(APPEND _inc_env "${ENV_SYCL_DIR}/include")
    list(APPEND _inc_env "${ENV_SYCL_DIR}/include/sycl/CL")
else()
    if(WIN32)
        string(REPLACE ":" ";" _inc_env "$ENV{INCLUDE}")
    else()
        string(REPLACE ":" ";" _path_env "$ENV{INCLUDE}")
        list(APPEND _inc_env "${_path_env}")
        string(REPLACE ":" ";" _path_env "$ENV{C_INCLUDE_PATH}")
        list(APPEND _inc_env "${_path_env}")
        string(REPLACE ":" ";" _path_env "$ENV{CPATH}")
        list(APPEND _inc_env "${_path_env}")
        string(REPLACE ":" ";" _path_env "$ENV{INCLUDE_PATH}")
        list(APPEND _inc_env "${_path_env}")
    endif()
endif()
list(APPEND _inc_env "${CMAKE_PLATFORM_IMPLICIT_INCLUDE_DIRECTORIES}")
list(APPEND _inc_env "${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES}")
list(REMOVE_DUPLICATES _inc_env)

# set paths where to look for
set(PATH_TO_LOOK_FOR "${_inc_env}")

# Try to find the SYCL header in the given paths
# -------------------------------------------------
# call cmake macro to find the header path
if(SYCL_INCDIR)
    set(SYCL_sycl.hpp_DIRS "SYCL_sycl.hpp_DIRS-NOTFOUND")
    find_path(SYCL_sycl.hpp_DIRS
        NAMES sycl.hpp
        HINTS ${SYCL_INCDIR})
else()
    if(SYCL_DIR)
        set(SYCL_sycl.hpp_DIRS "SYCL_sycl.hpp_DIRS-NOTFOUND")
        find_path(SYCL_sycl.hpp_DIRS
	    NAMES sycl.hpp
            HINTS ${SYCL_DIR}
            PATH_SUFFIXES "include" "include/sycl/CL")
    else()
        set(SYCL_sycl.hpp_DIRS "SYCL_sycl.hpp_DIRS-NOTFOUND")
        find_path(SYCL_sycl.hpp_DIRS
	    NAMES sycl.hpp
	    HINTS ${PATH_TO_LOOK_FOR}
            PATH_SUFFIXES "sycl/CL")
    endif()
endif()
mark_as_advanced(SYCL_sycl.hpp_DIRS)

# Add path to cmake variable
# ------------------------------------
if (SYCL_sycl.hpp_DIRS)
    set(SYCL_INCLUDE_DIRS "${SYCL_sycl.hpp_DIRS}")
else ()
    set(SYCL_INCLUDE_DIRS "SYCL_INCLUDE_DIRS-NOTFOUND")
    if(NOT SYCL_FIND_QUIETLY)
        message(STATUS "Looking for sycl -- sycl.hpp not found")
    endif()
endif ()

if (SYCL_INCLUDE_DIRS)
    list(REMOVE_DUPLICATES SYCL_INCLUDE_DIRS)
endif ()


# Looking for lib
# ---------------

# Add system library paths to search lib
# --------------------------------------
unset(_lib_env)
if(ENV_SYCL_LIBDIR)
    list(APPEND _lib_env "${ENV_SYCL_LIBDIR}")
elseif(ENV_SYCL_DIR)
    list(APPEND _lib_env "${ENV_SYCL_DIR}")
    list(APPEND _lib_env "${ENV_SYCL_DIR}/lib")
else()
    if(WIN32)
        string(REPLACE ":" ";" _lib_env "$ENV{LIB}")
    else()
        if(APPLE)
	    string(REPLACE ":" ";" _lib_env "$ENV{DYLD_LIBRARY_PATH}")
        else()
	    string(REPLACE ":" ";" _lib_env "$ENV{LD_LIBRARY_PATH}")
        endif()
        list(APPEND _lib_env "${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}")
        list(APPEND _lib_env "${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}")
    endif()
endif()
list(REMOVE_DUPLICATES _lib_env)

# set paths where to look for
set(PATH_TO_LOOK_FOR "${_lib_env}")

# Try to find the sycl lib in the given paths
# ----------------------------------------------

# call cmake macro to find the lib path
if(SYCL_LIBDIR)
    set(SYCL_sycl_LIBRARY "SYCL_sycl_LIBRARY-NOTFOUND")
    find_library(SYCL_sycl_LIBRARY
        NAMES sycl
        HINTS ${SYCL_LIBDIR})
else()
    if(SYCL_DIR)
        set(SYCL_sycl_LIBRARY "SYCL_sycl_LIBRARY-NOTFOUND")
        find_library(SYCL_sycl_LIBRARY
	    NAMES sycl
	    HINTS ${SYCL_DIR}
	    PATH_SUFFIXES lib lib32 lib64)
    else()
        set(SYCL_sycl_LIBRARY "SYCL_sycl_LIBRARY-NOTFOUND")
        find_library(SYCL_sycl_LIBRARY
	    NAMES sycl
	    HINTS ${PATH_TO_LOOK_FOR})
    endif()
endif()
mark_as_advanced(SYCL_sycl_LIBRARY)

# If found, add path to cmake variable
# ------------------------------------
if (SYCL_sycl_LIBRARY)
    get_filename_component(sycl_lib_path ${SYCL_sycl_LIBRARY} PATH)
    # set cmake variables (respects naming convention)
    set(SYCL_LIBRARIES    "${SYCL_sycl_LIBRARY}")
    set(SYCL_LIBRARY_DIRS "${sycl_lib_path}")
else ()
    set(SYCL_LIBRARIES    "SYCL_LIBRARIES-NOTFOUND")
    set(SYCL_LIBRARY_DIRS "SYCL_LIBRARY_DIRS-NOTFOUND")
    if(NOT SYCL_FIND_QUIETLY)
        message(STATUS "Looking for sycl -- lib sycl not found")
    endif()
endif ()

if (SYCL_LIBRARY_DIRS)
    list(REMOVE_DUPLICATES SYCL_LIBRARY_DIRS)
endif ()

# check a function to validate the find
if(SYCL_LIBRARIES)

    set(REQUIRED_INCDIRS)
    set(REQUIRED_LIBDIRS)
    set(REQUIRED_LIBS)

    # SYCL
    if (SYCL_INCLUDE_DIRS)
        set(REQUIRED_INCDIRS "${SYCL_INCLUDE_DIRS}")
    endif()
    if (SYCL_LIBRARY_DIRS)
        set(REQUIRED_LIBDIRS "${SYCL_LIBRARY_DIRS}")
    endif()
    set(REQUIRED_LIBS "${SYCL_LIBRARIES}")

    # set required libraries for link
    set(CMAKE_REQUIRED_INCLUDES "${REQUIRED_INCDIRS}")
    set(CMAKE_REQUIRED_LIBRARIES)
    foreach(lib_dir ${REQUIRED_LIBDIRS})
        list(APPEND CMAKE_REQUIRED_LIBRARIES "-L${lib_dir}")
    endforeach()
    list(APPEND CMAKE_REQUIRED_LIBRARIES "${REQUIRED_LIBS}")
    string(REGEX REPLACE "^ -" "-" CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")

    set(CMAKE_REQUIRED_INCLUDES)
    set(CMAKE_REQUIRED_FLAGS)
    set(CMAKE_REQUIRED_LIBRARIES)
endif(SYCL_LIBRARIES)

if (SYCL_LIBRARIES)
  if (SYCL_LIBRARY_DIRS)
    list(GET SYCL_LIBRARY_DIRS 0 first_lib_path)
  else()
    list(GET SYCL_LIBRARIES 0 first_lib)
    get_filename_component(first_lib_path "${first_lib}" PATH)
  endif()
  if (${first_lib_path} MATCHES "/lib(32|64)?$")
    string(REGEX REPLACE "/lib(32|64)?$" "" not_cached_dir "${first_lib_path}")
    set(SYCL_DIR_FOUND "${not_cached_dir}" CACHE PATH "Installation directory of SYCL library" FORCE)
  else()
    set(SYCL_DIR_FOUND "${first_lib_path}" CACHE PATH "Installation directory of SYCL library" FORCE)
  endif()
endif()
mark_as_advanced(SYCL_DIR)
mark_as_advanced(SYCL_DIR_FOUND)

# Function for converting hex version numbers from SYCL_API_VERSION if necessary
function(HEX2DEC str res)
    string(LENGTH "${str}" len)
    if("${len}" EQUAL 1)
        if("${str}" MATCHES "[0-9]")
            set(${res} "${str}" PARENT_SCOPE)
        elseif( "${str}" MATCHES "[aA]")
            set(${res} 10 PARENT_SCOPE)
        elseif( "${str}" MATCHES "[bB]")
            set(${res} 11 PARENT_SCOPE)
        elseif( "${str}" MATCHES "[cC]")
            set(${res} 12 PARENT_SCOPE)
        elseif( "${str}" MATCHES "[dD]")
            set(${res} 13 PARENT_SCOPE)
        elseif( "${str}" MATCHES "[eE]")
            set(${res} 14 PARENT_SCOPE)
        elseif( "${str}" MATCHES "[fF]")
            set(${res} 15 PARENT_SCOPE)
        else()
            return()
        endif()
    else()
        string(SUBSTRING "${str}" 0 1 str1)
        string(SUBSTRING "${str}" 1 -1 str2)
        hex2dec(${str1} res1)
        hex2dec(${str2} res2)
        math(EXPR val "16 * ${res1} + ${res2}")
        set(${res} "${val}" PARENT_SCOPE)
    endif()
endfunction()

if(SYCL_INCLUDE_DIRS)
    # Parse header to get SYCL version
    # Also used to check that library and header versions match
    file(READ "${SYCL_INCLUDE_DIRS}/sycl/version.hpp"
         HEADER_CONTENTS LIMIT 16384)
    string(REGEX REPLACE ".*#define __SYCL_COMPILER_VERSION ([0-9]+).*" "\\1"
        SYCL_COMPILER_VERSION "${HEADER_CONTENTS}")
    set(SYCL_VERSION ${SYCL_COMPILER_VERSION} CACHE STRING "SYCL library version")
    set(GMX_SYCL_COMPILER_VERSION ${SYCL_COMPILER_VERSION} CACHE STRING "SYCL compiler version during configuration time")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SYCL
                                  REQUIRED_VARS SYCL_LIBRARIES SYCL_INCLUDE_DIRS
                                  VERSION_VAR SYCL_VERSION)

mark_as_advanced(SYCL_INCLUDE_DIRS SYCL_LIBRARIES SYCL_VERSION)
