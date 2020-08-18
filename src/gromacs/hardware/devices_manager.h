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
 *  \brief Declares the class to manage GPU resources.
 *
 *  This class has several implementations: one for each supported GPU platform,
 *  and a stub implementation if the build does not support GPUs.
 *
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \inlibraryapi
 * \ingroup module_hardware
 */
#ifndef GMX_HARDWARE_DEVICES_MANAGER_H
#define GMX_HARDWARE_DEVICES_MANAGER_H

#include <memory>
#include <string>
#include <vector>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/iserializer.h"

struct DeviceInformation;

/*! \brief The GPU devices management class
 *
 * The class includes the information on either CUDA or OpenCL devices as well as all the
 * logic needed to initialize them, identify compatible devices and set the default device
 * for current rank.
 *
 * \todo Use a std::vector (requires PImpling of including the device information header).
 */
class DevicesManager
{
public:
    //! Constructor.
    DevicesManager() = default;
    //! Destructor.
    ~DevicesManager();

    /*! \brief Return whether GPUs can be detected.
     *
     * Returns true when this is a build of \Gromacs configured to support
     * GPU usage, GPU detection is not disabled by an environment variable
     * and a valid device driver, ICD, and/or runtime was detected.
     * Does not throw.
     * \param[out] errorMessage  When returning false on a build configured with
     *                           GPU support and non-nullptr was passed,
     *                           the string contains a descriptive message about
     *                           why GPUs cannot be detected.
     */
    static bool canPerformGpuDetection(std::string* errorMessage);

    /*! \brief Return whether GPU detection is functioning correctly
     *
     * Returns true when this is a build of \Gromacs configured to support
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
    static bool isGpuDetectionFunctional(std::string* errorMessage);

    /*! \brief Find all GPUs in the system.
     *
     *  Will detect every GPU supported by the device driver in use.
     *  Must only be called if canPerformGpuDetection() has returned true.
     *  This routine also checks for the compatibility of each and fill the
     *  deviceInfo array with the required information on each the
     *  device: ID, device properties, status.
     *
     *  Note that this function leaves the GPU runtime API error state clean;
     *  this is implemented ATM in the CUDA flavor.
     *
     *  \todo:  Check if errors do propagate in OpenCL as they do in CUDA and
     *          whether there is a mechanism to "clear" them.
     *
     *  \throws InternalError if a GPU API returns an unexpected failure (because
     *          the call to canDetectGpus() should always prevent this occuring)
     */
    void findGpus();

    static std::vector<std::unique_ptr<DeviceInformation>> findDevices();

    /*! \brief Return a container of the detected GPUs that are compatible.
     *
     * This function filters the result of the detection for compatible
     * GPUs, based on the previously run compatibility tests.
     *
     * \return  Vector of IDs of GPUs already recorded as compatible
     */
    std::vector<int> getCompatibleGpus() const;

    /*! \brief Return a container of the detected GPUs that are compatible.
     *
     * This function filters the result of the detection for compatible
     * GPUs, based on the previously run compatibility tests.
     *
     * \return  Vector of IDs of GPUs already recorded as compatible
     */
    static std::vector<int> getCompatibleGpus(const std::vector<std::unique_ptr<DeviceInformation>>& devicesInformation);

    static bool isGpuCompatible(const DeviceInformation& deviceInformation);


    /*! \brief Set the active GPU
     *
     * \param[in] deviceId  Index of selected device.
     *
     * Issues a fatal error for any critical errors that occur during
     * initialization.
     */
    void        setDevice(int deviceId) const;
    static void setDevice(const DeviceInformation& deviceInfo);

    /*! \brief Return a pointer to the device information for \c deviceId
     *
     * \param[in] deviceId  ID for the GPU device requested.
     *
     * \returns  Pointer to the device info for \c deviceId.
     */
    DeviceInformation* getDeviceInformation(int deviceId) const;

    /*! \brief Formats and returns a device information string for a given GPU.
     *
     * Given an index *directly* into the array of available GPUs, returns
     * a formatted info string for the respective GPU which includes ID, name,
     * compute capability, and detection status.
     *
     * \param[in] deviceId  An index *directly* into the array of available GPUs
     *
     * \returns A string describing the device.
     */
    std::string        getDeviceInformationString(int deviceId) const;
    static std::string getDeviceInformationString(const DeviceInformation& deviceInfo);

    /*! \brief Return a string describing how compatible the GPU with given \c deviceId is.
     *
     * \param[in] deviceInfos An information on available devices.
     * \param[in] devdeviceId An index of the device to check
     * \returns               A string describing the compatibility status, useful for error messages.
     */
    static std::string getGpuCompatibilityDescription(const std::vector<std::unique_ptr<DeviceInformation>>& deviceInfos,
                                                      int deviceId);

    /*! \brief Returns the size of the DeviceInformation struct.
     *
     * The size of DeviceInformation depends on the the implementation used.
     * Needed for communication.
     *
     * \returns  Size in bytes of DeviceInformation
     */
    static size_t getDeviceInformationSize();

    /*! \brief Checks if one can compute on the GPU
     *
     * \returns  True if the build supports GPUs and there are at least one available.
     */
    static bool canComputeOnGpu();

    static void serializeDeviceInformations(const std::vector<std::unique_ptr<DeviceInformation>>& deviceInfos,
                                            gmx::ISerializer* serializer);

    static std::vector<std::unique_ptr<DeviceInformation>> deserializeDeviceInformations(gmx::ISerializer* serializer);

    //! Total number of GPU devices detected on this physical node
    int numDevices() const { return numDevices_; }
    //! Number of GPU devices detected on this physical node that are compatible.
    int numCompatibleDevices() const { return numCompatibleDevices_; }


    //! Total number of GPU devices detected on this physical node
    int numDevices_ = 0;
    //! Number of GPU devices detected on this physical node that are compatible.
    int numCompatibleDevices_ = 0;
    //! Information about each GPU device detected on this physical node
    DeviceInformation* deviceInfos_ = nullptr;
};

#endif // GMX_HARDWARE_DEVICES_MANAGER_H
