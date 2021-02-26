/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021, by the GROMACS development team, led by
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
 *  \brief Defines DeviceEvent class for CUDA.
 *
 *  \author Andrey Alekseenko <al42and@gmail.com>
 *  \inlibraryapi
 */

#include "gmxpre.h"

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/gputraits.cuh"
#include "gromacs/utility/gmxassert.h"

#include "device_event.h"

// In CUDA, the DeviceEvent can only get invalidated by a move ctor.
// No need to check for this value anywhere except in the dtor.
static constexpr DeviceEvent::NativeType sc_nullEvent = nullptr;

DeviceEvent::DeviceEvent()
{
    cudaError_t stat = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    GMX_RELEASE_ASSERT(stat == cudaSuccess,
                       ("cudaEventCreate failed. " + gmx::getDeviceErrorString(stat)).c_str());
}

DeviceEvent::DeviceEvent(DeviceEvent::NativeType event) : event_(event) {}
DeviceEvent::DeviceEvent(DeviceEvent&& other) noexcept :
    event_(std::exchange(other.event_, sc_nullEvent))
{
}

DeviceEvent::~DeviceEvent()
{
    if (event_ != sc_nullEvent)
    {
        cudaError_t gmx_used_in_debug stat = cudaEventDestroy(event_);
        GMX_RELEASE_ASSERT(stat == cudaSuccess,
                           ("cudaEventDestroy failed. " + gmx::getDeviceErrorString(stat)).c_str());
    }
}

bool DeviceEvent::isValid() const
{
    return true;
}

bool DeviceEvent::isReady() const
{
    cudaError_t stat = cudaEventQuery(event_);
    GMX_ASSERT((stat == cudaSuccess) || (stat == cudaErrorNotReady),
               ("cudaEventQuery failed. " + gmx::getDeviceErrorString(stat)).c_str());
    return (stat == cudaSuccess);
}

void DeviceEvent::wait()
{
    cudaError_t gmx_used_in_debug stat = cudaEventSynchronize(event_);
    GMX_ASSERT(stat == cudaSuccess,
               ("cudaEventSynchronize failed. " + gmx::getDeviceErrorString(stat)).c_str());
}
bool DeviceEvent::timingSupported() const
{
    return false;
}

uint64_t DeviceEvent::getExecutionTime()
{
    GMX_RELEASE_ASSERT(false, "Timing not supported in CUDA");
    return 0;
}

const DeviceEvent::NativeType& DeviceEvent::getNative() const
{
    return event_;
}
