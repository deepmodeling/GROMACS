/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2016 by the GROMACS development team.
 * Copyright (c) 2017,2018,2019,2020, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#ifndef GMX_HARDWARE_PRINTHARDWARE_H
#define GMX_HARDWARE_PRINTHARDWARE_H

#include <cstdio>

#include "gromacs/utility/gmxmpi.h"

struct gmx_hw_info_t;

namespace gmx
{
class MDLogger;
}

/*! \brief Report information about the detected hardware
 *
 * \param[in] fplog         Log file to write to, when valid
 * \param[in] warnToStdErr  Whether to issue warnings on stderr
 * \param[in] mdlog         Log file to write to, when valid
 * \param[in] hwinfo        The hardware detection to report
 * \param[in] simulationCommunicator  The communicator for the simulation of this rank
 *
 * Note that passing both of fplog and mdlog is intended, so that the
 * SIMD module stays free of logger dependencies.
 *
 * \todo Return strings from simdCheck (so that the SIMD module
 * remains low on dependencies) and write them to mdlog, so that fplog
 * is not needed here. */
void gmx_print_detected_hardware(FILE*                fplog,
                                 bool                 warnToStdErr,
                                 const gmx::MDLogger& mdlog,
                                 const gmx_hw_info_t& hwinfo,
                                 MPI_Comm             simulationCommunicator);

#endif
