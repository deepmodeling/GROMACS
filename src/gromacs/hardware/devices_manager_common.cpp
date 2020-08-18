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
/*! \libinternal \file
 *  \brief Defines the implementations of the DevicesManager class that are common for CPU, CUDA and OpenCL.
 *
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \inlibraryapi
 * \ingroup module_hardware
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/hardware/device_information.h"
#include "gromacs/hardware/devices_manager.h"
#include "gromacs/utility/fatalerror.h"

bool DevicesManager::canPerformGpuDetection(std::string* errorMessage)
{
    if (GMX_GPU && getenv("GMX_DISABLE_GPU_DETECTION") == nullptr)
    {
        return DevicesManager::isGpuDetectionFunctional(errorMessage);
    }
    else
    {
        return false;
    }
}

std::vector<int> DevicesManager::getCompatibleGpus() const
{
    // Possible minor over-allocation here, but not important for anything
    std::vector<int> compatibleGpus;
    compatibleGpus.reserve(numDevices_);
    for (int i = 0; i < numDevices_; i++)
    {
        if (deviceInfos_[i].status == DeviceStatus::Compatible)
        {
            compatibleGpus.push_back(i);
        }
    }
    return compatibleGpus;
}

std::vector<int> DevicesManager::getCompatibleGpus(const std::vector<std::unique_ptr<DeviceInformation>>& devicesInformation)
{
    // Possible minor over-allocation here, but not important for anything
    std::vector<int> compatibleGpus;
    compatibleGpus.reserve(devicesInformation.size());
    for (int i = 0; i < static_cast<int>(devicesInformation.size()); i++)
    {
        if (devicesInformation[i]->status == DeviceStatus::Compatible)
        {
            compatibleGpus.push_back(i);
        }
    }
    return compatibleGpus;
}

bool DevicesManager::isGpuCompatible(const DeviceInformation& deviceInformation)
{
    return (deviceInformation.status == DeviceStatus::Compatible);
}

int DevicesManager::numCompatibleDevices(const std::vector<std::unique_ptr<DeviceInformation>>& deviceInfos)
{
    int numCompatibleDevices = 0;
    for (const auto& deviceInfo : deviceInfos)
    {
        if (DevicesManager::isGpuCompatible(*deviceInfo))
        {
            numCompatibleDevices++;
        }
    }
    return numCompatibleDevices;
}

DeviceInformation* DevicesManager::getDeviceInformation(int deviceId) const
{
    if (deviceId < 0 || deviceId >= numDevices_)
    {
        gmx_incons("Invalid GPU deviceId requested");
    }
    return &deviceInfos_[deviceId];
}

std::string DevicesManager::getGpuCompatibilityDescription(
        const std::vector<std::unique_ptr<DeviceInformation>>& deviceInfos,
        int                                                    deviceId)
{
    return (deviceId >= static_cast<int>(deviceInfos.size())
                    ? c_deviceStateString[DeviceStatus::Nonexistent]
                    : c_deviceStateString[deviceInfos[deviceId]->status]);
}

bool DevicesManager::canComputeOnGpu()
{
    bool           canComputeOnGpu = false;
    DevicesManager devicesManager{};
    if (DevicesManager::canPerformGpuDetection(nullptr))
    {
        devicesManager.findGpus();
        canComputeOnGpu = !devicesManager.getCompatibleGpus().empty();
    }
    return canComputeOnGpu;
}

void DevicesManager::serializeDeviceInformations(const std::vector<std::unique_ptr<DeviceInformation>>& deviceInfos,
                                                 gmx::ISerializer* serializer)
{
    int numDevices = deviceInfos.size();
    serializer->doInt(&numDevices);
    for (auto& deviceInfo : deviceInfos)
    {
        serializer->doOpaque(reinterpret_cast<char*>(deviceInfo.get()), sizeof(DeviceInformation));
    }
}

std::vector<std::unique_ptr<DeviceInformation>> DevicesManager::deserializeDeviceInformations(gmx::ISerializer* serializer)
{
    int numDevices = 0;
    serializer->doInt(&numDevices);
    std::vector<std::unique_ptr<DeviceInformation>> deviceInfos(numDevices);
    for (int i = 0; i < numDevices; i++)
    {
        deviceInfos[i] = std::make_unique<DeviceInformation>();
        serializer->doOpaque(reinterpret_cast<char*>(deviceInfos[i].get()), sizeof(DeviceInformation));
    }
    return deviceInfos;
}
