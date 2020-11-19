#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2013,2014,2015,2016,2019,2020, by the GROMACS development team, led by
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

# Set up the module library
add_library(fileio OBJECT)
target_sources(fileio PRIVATE
               checkpoint.cpp
               confio.cpp
               enxio.cpp
               espio.cpp
               filetypes.cpp
               g96io.cpp
               gmx_internal_xdr.cpp
               gmxfio.cpp
               gmxfio_xdr.cpp
               groio.cpp
               libxdrf.cpp
               matio.cpp
               md5.cpp
               mrcdensitymap.cpp
               mrcdensitymapheader.cpp
               mrcserializer.cpp
               mtxio.cpp
               oenv.cpp
               pdbio.cpp
               readinp.cpp
               timecontrol.cpp
               tngio.cpp
               tpxio.cpp
               trrio.cpp
               trxio.cpp
               warninp.cpp
               writeps.cpp
               xdrd.cpp
               xtcio.cpp
               xvgr.cpp)
set_target_properties(fileio PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(fileio PRIVATE HAVE_CONFIG_H)

if(GMX_USE_PLUGINS)
    target_sources(fileio vmdio.cpp)
    add_library(vmddlopen OBJECT ${CMAKE_SOURCE_DIR}/src/external/vmd_molfile/vmddlopen.cpp)
    gmx_target_compile_options(vmddlopen)
    if (WIN32)
        gmx_target_warning_suppression(vmddlopen /wd4996 HAS_NO_MSVC_UNSAFE_FUNCTION)
    endif()
    list(APPEND libgromacs_object_library_dependencies vmddlopen)
    set(libgromacs_object_library_dependencies ${libgromacs_object_library_dependencies} PARENT_SCOPE)
endif()

# Set up dependencies that are internal to this module
target_include_directories(fileio PRIVATE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
# Include directories for dependencies not yet made into an object
# library. Ref #3812
target_include_directories(fileio SYSTEM PRIVATE
                           ${PROJECT_SOURCE_DIR}/src/external/thread_mpi/include
                           )

# Source files have the following dependencies on library infrastructure.
target_link_libraries(fileio PRIVATE
                      common
                      legacy_modules
                      )

# Source files have the following private module dependencies.
target_link_libraries(fileio PRIVATE
    gmxlib
    math
    mdtypes
    tng_io
    )

# Public interface for modules, including dependencies and interfaces
target_include_directories(fileio PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(fileio PUBLIC
                      legacy_api
                      )

target_link_libraries(fileio PRIVATE common)

# Module dependencies
# fileio interfaces convey transitive dependence on these modules.
target_link_libraries(fileio PUBLIC
                      utility
                      )
# Source files have the following private module dependencies.
target_link_libraries(fileio PRIVATE tng_io)
# TODO: Explicitly link specific modules.
target_link_libraries(fileio PRIVATE legacy_modules)

if (BUILD_TESTING)
     add_subdirectory(tests)
endif()
