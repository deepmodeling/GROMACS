#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2010,2011,2012,2013,2014 by the GROMACS development team.
# Copyright (c) 2015,2017,2018,2019,2020, by the GROMACS development team, led by
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

# TODO: Generate a header instead that can be included from baseversion.cpp.
# That probably simplifies things somewhat.
gmx_configure_version_file(
    baseversion-gen.cpp.cmakein baseversion-gen.cpp
    REMOTE_HASH
    EXTRA_VARS
        GMX_SOURCE_DOI
        GMX_RELEASE_HASH
        GMX_SOURCE_HASH
        )

# Set up the module library
file(GLOB TARGET_SOURCES *.cpp)
add_library(utility OBJECT
    # Normal source files
    ${TARGET_SOURCES}
    # Generated source files
    ${CMAKE_CURRENT_BINARY_DIR}/baseversion-gen.cpp
    )
set_target_properties(utility PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_source_files_properties(
    binaryinformation.cpp
    fatalerror.cpp
    PROPERTIES
    COMPILE_DEFINITIONS HAVE_CONFIG_H
    )

# Set up dependencies that are internal to this module
target_include_directories(utility PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
# Include directories for dependencies not yet made into an object
# library.
target_include_directories(utility SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/src/external/thread_mpi/include
    )

# Source files have the following private module dependencies.
target_link_libraries(utility PRIVATE
    common
    legacy_api
    ${FFT_LIBRARIES}
    )

# This module provides interfaces to other library modules in the build tree.
# The installed API it supports are exported through the legacy_api
# and libgromacs targets.
target_include_directories(utility PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

# MPI dependency needs to be satisifed
target_link_libraries(utility PUBLIC "${MPI_C_LIBRARIES}")

# TODO make this work better when improving GPU modules
if (GMX_GPU_CUDA)
    gmx_add_libgromacs_sources(cuda_version_information.cu)
endif()

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
