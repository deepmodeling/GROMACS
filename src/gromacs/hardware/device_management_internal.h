/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2016, by the GROMACS development team.
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
/*! \libinternal \file
 *  \brief Declares functions to manage GPU resources.
 *
 *  This has several implementations: one for each supported GPU platform,
 *  and a stub implementation if the build does not support GPUs.
 *
 *  \author Anca Hamuraru <anca@streamcomputing.eu>
 *  \author Dimitrios Karkoulis <dimitris.karkoulis@gmail.com>
 *  \author Teemu Virolainen <teemu@streamcomputing.eu>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \author Szilárd Páll <pall.szilard@gmail.com>
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \inlibraryapi
 * \ingroup module_hardware
 */
#ifndef GMX_HARDWARE_DEVICE_MANAGEMENT_INTERNAL_H
#define GMX_HARDWARE_DEVICE_MANAGEMENT_INTERNAL_H

#include <memory>
#include <string>
#include <vector>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/iserializer.h"

struct DeviceInformation;

/*! \brief Return whether GPUs can be detected.
 *
 * Returns true when this is a build of GROMACS configured to support
 * GPU usage, GPU detection is not disabled by \c GMX_DISABLE_GPU_DETECTION
 * environment variable and a valid device driver, ICD, and/or runtime was
 * detected. Does not throw.
 *
 * \param[out] errorMessage  When returning false on a build configured with
 *                           GPU support and non-nullptr was passed,
 *                           the string contains a descriptive message about
 *                           why GPUs cannot be detected.
 */
bool canPerformDeviceDetection(std::string* errorMessage);

/*! \brief Return whether GPU detection is functioning correctly
 *
 * Returns true when this is a build of GROMACS configured to support
 * GPU usage, and a valid device driver, ICD, and/or runtime was detected.
 *
 * This function is not intended to be called from build
 * configurations that do not support GPUs, and there will be no
 * descriptive message in that case.
 *
 * \param[out] errorMessage  When returning false on a build configured with
 *                           GPU support and non-nullptr was passed,
 *                           the string contains a descriptive message about
 *                           why GPUs cannot be detected.
 *
 * Does not throw.
 */
bool isDeviceDetectionFunctional(std::string* errorMessage);

/*! \brief Find all GPUs in the system.
 *
 *  Will detect every GPU supported by the device driver in use.
 *  Must only be called if \c canPerformDeviceDetection() has returned true.
 *  This routine also checks for the compatibility of each device and fill the
 *  deviceInfo array with the required information on each device: ID, device
 *  properties, status.
 *
 *  Note that this function leaves the GPU runtime API error state clean;
 *  this is implemented ATM in the CUDA flavor.
 *
 *  \todo:  Check if errors do propagate in OpenCL as they do in CUDA and
 *          whether there is a mechanism to "clear" them.
 *
 * \return  Standard vector with the list of devices found
 *
 *  \throws InternalError if a GPU API returns an unexpected failure (because
 *          the call to canDetectGpus() should always prevent this occuring)
 */
std::vector<std::unique_ptr<DeviceInformation>> findDevices();

#endif // GMX_HARDWARE_DEVICE_MANAGEMENT_INTERNAL_H
