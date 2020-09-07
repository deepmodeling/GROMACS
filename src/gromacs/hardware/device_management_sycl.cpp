/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020, by the GROMACS development team, led by
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
    std::vector<cl::sycl::device>                   devices = cl::sycl::device::get_devices();
    deviceInfos.reserve(devices.size());
    for (auto syclDevice : devices)
    {
        deviceInfos.emplace_back(std::make_unique<DeviceInformation>());

        int i = deviceInfos.size() - 1;

        deviceInfos[i]->id         = i;
        deviceInfos[i]->syclDevice = syclDevice;

        // DELETE-ME: Code for testing purposes

        std::cout << "    Device: " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "    Type:   ";
        if (device.is_gpu())
        {
            std::cout << "GPU" << std::endl;
        }
        else if (device.is_cpu())
        {
            std::cout << "CPU" << std::endl;
        }
        else
        {
            std::cout << "Unknown" << std::endl;
        }
        std::cout << "    Is host: " << (device.is_host() ? "Yes" : "No") << std::endl;
        std::cout << "    Is accelerator: " << (device.is_accelerator() ? "Yes" : "No") << std::endl;
        std::cout << "    Max work group size: "
                  << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
        std::cout << "    Local mem size: " << device.get_info<sycl::info::device::local_mem_size>()
                  << std::endl;

        // DELETE-ME: End of testing code
    }
    return deviceInfos;
}

void setActiveDevice(const DeviceInformation& /* deviceInfo */)
{
    // SYCL-TODO:
    // Activate the selected device
}

void releaseDevice(DeviceInformation* /* deviceInfo */)
{
    // SYCL-TODO:
    // Release the device
}

std::string getDeviceInformationString(const DeviceInformation& /* deviceInfo */)
{
    // SYCL-TODO:
    // Retrun a string with human-friendly information on the device.
}
