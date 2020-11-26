#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2010,2013,2014,2015,2018 by the GROMACS development team.
# Copyright (c) 2019,2020, by the GROMACS development team, led by
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
add_library(trajectoryanalysis OBJECT)
target_sources(trajectoryanalysis PRIVATE
               analysismodule.cpp
               analysissettings.cpp
               cmdlinerunner.cpp
               modules/angle.cpp
               modules/convert_trj.cpp
               modules.cpp
               modules/distance.cpp
               modules/extract_cluster.cpp
               modules/freevolume.cpp
               modules/pairdist.cpp
               modules/rdf.cpp
               modules/sasa.cpp
               modules/select.cpp
               modules/surfacearea.cpp
               modules/trajectory.cpp
               runnercommon.cpp
               topologyinformation.cpp
               )
set_target_properties(trajectoryanalysis PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set up dependencies that are internal to this module
target_include_directories(trajectoryanalysis PRIVATE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

# Source files have the following dependencies on library infrastructure.
target_link_libraries(trajectoryanalysis PRIVATE
                      analysisdata
                      commandline
                      common
                      coordinateio
                      fileio
                      legacy_modules
                      topology
)

# Public interface for modules, including dependencies and interfaces
target_include_directories(trajectoryanalysis PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(trajectoryanalysis PUBLIC
                      legacy_api
                      )

target_link_libraries(trajectoryanalysis PRIVATE common)

# Module dependencies
# trajectoryanalysis interfaces convey transitive dependence on these modules.
target_link_libraries(trajectoryanalysis PUBLIC
                      gmxana
                      gmxpreprocess
                      options
                      utility
                      )
# Source files have the following private module dependencies.
#target_link_libraries(trajectoryanalysis PRIVATE NOTHING)
# TODO: Explicitly link specific modules.
target_link_libraries(trajectoryanalysis PRIVATE legacy_modules)

if(GMX_INSTALL_LEGACY_API)
    install(FILES
	    analysismodule.h
	    analysissettings.h
	    cmdlinerunner.h
	    topologyinformation.h
            DESTINATION include/gromacs/trajectoryanalysis)
endif()

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
