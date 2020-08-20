/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2017 The GROMACS development team.
 * Copyright (c) 2018,2019,2020, by the GROMACS development team, led by
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
/*! \internal \file
 *  \brief Defines the OpenCL implementations of the DevicesManager class.
 *
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_hardware
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/hardware/device_information.h"
#include "gromacs/hardware/device_management.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

bool isDeviceDetectionFunctional(std::string* /* errorMessage */)
{
    // SYCL-TODO:
    // Check here if devices can be detected
    // Fill in the error message
    return false;
}

std::vector<std::unique_ptr<DeviceInformation>> findDevices()
{
    // SYCL-TODO:
    // Do the device detection and return the standard vector of SYCL-specific device informations.
    std::vector<std::unique_ptr<DeviceInformation>> deviceInfos(0);
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    deviceInfos.reserve(devices.size());
    for (int i = 0; i < static_cast<int>(devices.size()); i++)
    {
        deviceInfos[i] = std::make_unique<DeviceInformation>();
        deviceInfos[i]->id = i;
        deviceInfos[i]->syclDevice = syclDevice;
    }
    return deviceInfos;
}

void setDevice(const DeviceInformation& /* deviceInfo */)
{
    // SYCL-TODO:
    // Activate the selected device
}

void freeDevice(DeviceInformation* /* deviceInfo */)
{
    // SYCL-TODO:
    // Release the device
}

std::string getDeviceInformationString(const DeviceInformation& /* deviceInfo */)
{
    // SYCL-TODO:
    // Retrun a string with human-friendly information on the device.
}
