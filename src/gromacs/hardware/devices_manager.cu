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
#include <cuda_runtime.h>

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"

#include "devices_manager.h"

/*! \brief Frees up the CUDA GPU used by the active context at the time of calling.
 *
 * If \c deviceInfo is nullptr, then it is understood that no device
 * was selected so no context is active to be freed. Otherwise, the
 * context is explicitly destroyed and therefore all data uploaded to
 * the GPU is lost. This must only be called when none of this data is
 * required anymore, because subsequent attempts to free memory
 * associated with the context will otherwise fail.
 *
 * Calls gmx_warning upon errors.
 *
 * TODO: This should go through all the devices, not only the one currently active.
 *       Reseting only one device will not work, e.g. in CUDA.
 */
DevicesManager::~DevicesManager()
{
    // One should only attempt to clear the device context when
    // it has been used, but currently the only way to know that a GPU
    // device was used is that deviceInfo will be non-null.
    if (deviceInfo_ != nullptr)
    {
        cudaError_t stat;

        int gpuid;
        stat = cudaGetDevice(&gpuid);
        if (stat == cudaSuccess)
        {
            if (debug)
            {
                fprintf(stderr, "Cleaning up context on GPU ID #%d\n", gpuid);
            }

            stat = cudaDeviceReset();
            if (stat != cudaSuccess)
            {
                gmx_warning("Failed to free GPU #%d: %s", gpuid, cudaGetErrorString(stat));
            }
        }
    }
}

/*! \internal \brief
 * Max number of devices supported by CUDA (for consistency checking).
 *
 * In reality it is 16 with CUDA <=v5.0, but let's stay on the safe side.
 */
static int c_cudaMaxDeviceCount = 32;

/** Dummy kernel used for sanity checking. */
static __global__ void dummy_kernel(void) {}

static cudaError_t checkCompiledTargetCompatibility(int deviceId, const cudaDeviceProp& deviceProp)
{
    cudaFuncAttributes attributes;
    cudaError_t        stat = cudaFuncGetAttributes(&attributes, dummy_kernel);

    if (cudaErrorInvalidDeviceFunction == stat)
    {
        fprintf(stderr,
                "\nWARNING: The %s binary does not include support for the CUDA architecture of "
                "the GPU ID #%d (compute capability %d.%d) detected during detection. "
                "By default, GROMACS supports all architectures of compute "
                "capability >= 3.0, so your GPU "
                "might be rare, or some architectures were disabled in the build. \n"
                "Consult the install guide for how to use the GMX_CUDA_TARGET_SM and "
                "GMX_CUDA_TARGET_COMPUTE CMake variables to add this architecture. \n",
                gmx::getProgramContext().displayName(), deviceId, deviceProp.major, deviceProp.minor);
    }

    return stat;
}

/*!
 * \brief Runs GPU sanity checks.
 *
 * Runs a series of checks to determine that the given GPU and underlying CUDA
 * driver/runtime functions properly.
 *
 * \param[in]  dev_id      the device ID of the GPU or -1 if the device has already been initialized
 * \param[in]  dev_prop    The device properties structure
 * \returns                0 if the device looks OK, -1 if it sanity checks failed, and -2 if the device is busy
 *
 * TODO: introduce errors codes and handle errors more smoothly.
 */
static int doSanityChecks(int dev_id, const cudaDeviceProp& dev_prop)
{
    cudaError_t cu_err;
    int         dev_count, id;

    cu_err = cudaGetDeviceCount(&dev_count);
    if (cu_err != cudaSuccess)
    {
        fprintf(stderr, "Error %d while querying device count: %s\n", cu_err, cudaGetErrorString(cu_err));
        return -1;
    }

    /* no CUDA compatible device at all */
    if (dev_count == 0)
    {
        return -1;
    }

    /* things might go horribly wrong if cudart is not compatible with the driver */
    if (dev_count < 0 || dev_count > c_cudaMaxDeviceCount)
    {
        return -1;
    }

    if (dev_id == -1) /* device already selected let's not destroy the context */
    {
        cu_err = cudaGetDevice(&id);
        if (cu_err != cudaSuccess)
        {
            fprintf(stderr, "Error %d while querying device id: %s\n", cu_err, cudaGetErrorString(cu_err));
            return -1;
        }
    }
    else
    {
        id = dev_id;
        if (id > dev_count - 1) /* pfff there's no such device */
        {
            fprintf(stderr,
                    "The requested device with id %d does not seem to exist (device count=%d)\n",
                    dev_id, dev_count);
            return -1;
        }
    }

    /* both major & minor is 9999 if no CUDA capable devices are present */
    if (dev_prop.major == 9999 && dev_prop.minor == 9999)
    {
        return -1;
    }
    /* we don't care about emulation mode */
    if (dev_prop.major == 0)
    {
        return -1;
    }

    if (id != -1)
    {
        cu_err = cudaSetDevice(id);
        if (cu_err != cudaSuccess)
        {
            fprintf(stderr, "Error %d while switching to device #%d: %s\n", cu_err, id,
                    cudaGetErrorString(cu_err));
            return -1;
        }
    }

    cu_err = checkCompiledTargetCompatibility(dev_id, dev_prop);
    // Avoid triggering an error if GPU devices are in exclusive or prohibited mode;
    // it is enough to check for cudaErrorDevicesUnavailable only here because
    // if we encounter it that will happen in cudaFuncGetAttributes in the above function.
    if (cu_err == cudaErrorDevicesUnavailable)
    {
        return -2;
    }
    else if (cu_err != cudaSuccess)
    {
        return -1;
    }

    /* try to execute a dummy kernel */
    try
    {
        KernelLaunchConfig config;
        config.blockSize[0]                = 512;
        const auto          dummyArguments = prepareGpuKernelArguments(dummy_kernel, config);
        DeviceInformation   deviceInfo;
        const DeviceContext deviceContext(deviceInfo);
        const DeviceStream  deviceStream(deviceContext, DeviceStreamPriority::Normal, false);
        launchGpuKernel(dummy_kernel, config, deviceStream, nullptr, "Dummy kernel", dummyArguments);
    }
    catch (gmx::GromacsException& ex)
    {
        // launchGpuKernel error is not fatal and should continue with marking the device bad
        fprintf(stderr,
                "Error occurred while running dummy kernel sanity check on device #%d:\n %s\n", id,
                formatExceptionMessageToString(ex).c_str());
        return -1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        return -1;
    }

    /* destroy context if we created one */
    if (id != -1)
    {
        cu_err = cudaDeviceReset();
        CU_RET_ERR(cu_err, "cudaDeviceReset failed");
    }

    return 0;
}

/*! \brief Returns true if the gpu characterized by the device properties is
 *  supported by the native gpu acceleration.
 *
 * \param[in] dev_prop  the CUDA device properties of the gpus to test.
 * \returns             true if the GPU properties passed indicate a compatible
 *                      GPU, otherwise false.
 */
static bool isDeviceGenerationSupported(const cudaDeviceProp& dev_prop)
{
    return (dev_prop.major >= 3);
}

/*! \brief Checks if a GPU with a given ID is supported by the native GROMACS acceleration.
 *
 *  Returns a status value which indicates compatibility or one of the following
 *  errors: incompatibility or insanity (=unexpected behavior).
 *
 *  As the error handling only permits returning the state of the GPU, this function
 *  does not clear the CUDA runtime API status allowing the caller to inspect the error
 *  upon return. Note that this also means it is the caller's responsibility to
 *  reset the CUDA runtime state.
 *
 *  \param[in]  deviceId   the ID of the GPU to check.
 *  \param[in]  deviceProp the CUDA device properties of the device checked.
 *  \returns               the status of the requested device
 */
static DeviceStatus isDeviceSupported(int deviceId, const cudaDeviceProp& deviceProp)
{
    if (!isDeviceGenerationSupported(deviceProp))
    {
        return DeviceStatus::Incompatible;
    }

    /* TODO: currently we do not make a distinction between the type of errors
     * that can appear during sanity checks. This needs to be improved, e.g if
     * the dummy test kernel fails to execute with a "device busy message" we
     * should appropriately report that the device is busy instead of insane.
     */
    const int checkResult = doSanityChecks(deviceId, deviceProp);
    switch (checkResult)
    {
        case 0: return DeviceStatus::Compatible;
        case -1: return DeviceStatus::Insane;
        case -2: return DeviceStatus::Unavailable;
        default:
            GMX_RELEASE_ASSERT(false, "Invalid sanity checks return value");
            return DeviceStatus::Compatible;
    }
}

bool DevicesManager::isGpuDetectionFunctional(std::string* errorMessage)
{
    cudaError_t stat;
    int         driverVersion = -1;
    stat                      = cudaDriverGetVersion(&driverVersion);
    GMX_ASSERT(stat != cudaErrorInvalidValue,
               "An impossible null pointer was passed to cudaDriverGetVersion");
    GMX_RELEASE_ASSERT(
            stat == cudaSuccess,
            gmx::formatString("An unexpected value was returned from cudaDriverGetVersion %s: %s",
                              cudaGetErrorName(stat), cudaGetErrorString(stat))
                    .c_str());
    bool foundDriver = (driverVersion > 0);
    if (!foundDriver)
    {
        // Can't detect GPUs if there is no driver
        if (errorMessage != nullptr)
        {
            errorMessage->assign("No valid CUDA driver found");
        }
        return false;
    }

    int numDevices;
    stat = cudaGetDeviceCount(&numDevices);
    if (stat != cudaSuccess)
    {
        if (errorMessage != nullptr)
        {
            /* cudaGetDeviceCount failed which means that there is
             * something wrong with the machine: driver-runtime
             * mismatch, all GPUs being busy in exclusive mode,
             * invalid CUDA_VISIBLE_DEVICES, or some other condition
             * which should result in GROMACS issuing at least a
             * warning. */
            errorMessage->assign(cudaGetErrorString(stat));
        }

        // Consume the error now that we have prepared to handle
        // it. This stops it reappearing next time we check for
        // errors. Note that if CUDA_VISIBLE_DEVICES does not contain
        // valid devices, then cudaGetLastError returns the
        // (undocumented) cudaErrorNoDevice, but this should not be a
        // problem as there should be no future CUDA API calls.
        // NVIDIA bug report #2038718 has been filed.
        cudaGetLastError();
        // Can't detect GPUs
        return false;
    }

    // We don't actually use numDevices here, that's not the job of
    // this function.
    return true;
}

void DevicesManager::findGpus()
{
    numCompatibleDevices_ = 0;

    cudaError_t stat = cudaGetDeviceCount(&numDevices_);
    if (stat != cudaSuccess)
    {
        GMX_THROW(gmx::InternalError(
                "Invalid call of findGpus() when CUDA API returned an error, perhaps "
                "canDetectGpus() was not called appropriately beforehand."));
    }

    // We expect to start device support/sanity checks with a clean runtime error state
    gmx::ensureNoPendingCudaError("");

    DeviceInformation* devs;
    snew(devs, numDevices_);
    for (int i = 0; i < numDevices_; i++)
    {
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(cudaDeviceProp));
        stat = cudaGetDeviceProperties(&prop, i);
        DeviceStatus checkResult;
        if (stat != cudaSuccess)
        {
            // Will handle the error reporting below
            checkResult = DeviceStatus::Insane;
        }
        else
        {
            checkResult = isDeviceSupported(i, prop);
        }

        devs[i].id   = i;
        devs[i].prop = prop;
        devs[i].stat = checkResult;

        if (checkResult == DeviceStatus::Compatible)
        {
            numCompatibleDevices_++;
        }
        else
        {
            // TODO:
            //  - we inspect the CUDA API state to retrieve and record any
            //    errors that occurred during isDeviceSupported() here,
            //    but this would be more elegant done within isDeviceSupported()
            //    and only return a string with the error if one was encountered.
            //  - we'll be reporting without rank information which is not ideal.
            //  - we'll end up warning also in cases where users would already
            //    get an error before mdrun aborts.
            //
            // Here we also clear the CUDA API error state so potential
            // errors during sanity checks don't propagate.
            if ((stat = cudaGetLastError()) != cudaSuccess)
            {
                gmx_warning("An error occurred while sanity checking device #%d; %s: %s",
                            devs[i].id, cudaGetErrorName(stat), cudaGetErrorString(stat));
            }
        }
    }

    stat = cudaPeekAtLastError();
    GMX_RELEASE_ASSERT(stat == cudaSuccess,
                       gmx::formatString("We promise to return with clean CUDA state, but "
                                         "non-success state encountered: %s: %s",
                                         cudaGetErrorName(stat), cudaGetErrorString(stat))
                               .c_str());

    deviceInfo_ = devs;
}

void DevicesManager::setDevice(int deviceId) const
{
    GMX_ASSERT(deviceId >= 0 && deviceId < numDevices_ && deviceInfo_ != nullptr,
               "Trying to set invalid device");

    cudaError_t stat;

    stat = cudaSetDevice(deviceId);
    if (stat != cudaSuccess)
    {
        auto message = gmx::formatString("Failed to initialize GPU #%d", deviceId);
        CU_RET_ERR(stat, message.c_str());
    }

    if (debug)
    {
        fprintf(stderr, "Initialized GPU ID #%d: %s\n", deviceId, deviceInfo_[deviceId].prop.name);
    }
}

std::string DevicesManager::getDeviceInformationString(int deviceId) const
{
    if (deviceId < 0 && deviceId >= numDevices_)
    {
        return "";
    }

    const DeviceInformation& deviceInfo = deviceInfo_[deviceId];

    bool gpuExists =
            (deviceInfo.stat != DeviceStatus::Nonexistent && deviceInfo.stat != DeviceStatus::Insane);

    if (!gpuExists)
    {
        return gmx::formatString("#%d: %s, stat: %s", deviceInfo.id, "N/A",
                                 c_deviceStateString[deviceInfo.stat]);
    }
    else
    {
        return gmx::formatString("#%d: NVIDIA %s, compute cap.: %d.%d, ECC: %3s, stat: %s",
                                 deviceInfo.id, deviceInfo.prop.name, deviceInfo.prop.major,
                                 deviceInfo.prop.minor, deviceInfo.prop.ECCEnabled ? "yes" : " no",
                                 c_deviceStateString[deviceInfo.stat]);
    }
}

size_t DevicesManager::getDeviceInformationSize()
{
    return sizeof(DeviceInformation);
}
