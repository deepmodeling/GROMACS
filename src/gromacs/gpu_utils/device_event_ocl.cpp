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
 *  \brief Defines DeviceEvent class for OpenCL.
 *
 *  \author Andrey Alekseenko <al42and@gmail.com>
 *  \inlibraryapi
 */

#include "gmxpre.h"

#include "gromacs/gpu_utils/gputraits_ocl.h"
#include "gromacs/gpu_utils/oclutils.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"

#include "device_event.h"

//! Magic value to indicate uninitialized state.
static constexpr DeviceEvent::NativeType sc_nullEvent = nullptr;

DeviceEvent::DeviceEvent() : event_(sc_nullEvent) {}
DeviceEvent::DeviceEvent(DeviceEvent::NativeType event) : event_(event) {}
DeviceEvent::DeviceEvent(DeviceEvent&& other) noexcept :
    event_(std::exchange(other.event_, sc_nullEvent))
{
}

DeviceEvent::~DeviceEvent()
{
    if (event_ != sc_nullEvent)
    {
        clReleaseEvent(event_);
    }
}

bool DeviceEvent::isValid() const
{
    return event_ != sc_nullEvent;
}

template<typename T>
static T getEventInfo(cl_event event, cl_event_info param_name)
{
    T      result;
    cl_int clError = clGetEventInfo(event, param_name, sizeof(T), &result, nullptr);
    if (clError != CL_SUCCESS)
    {
        GMX_THROW(gmx::InternalError("Failed to retrieve event info: " + ocl_get_error_string(clError)));
    }
    return result;
}

template<typename T>
static T getEventProfilingInfo(cl_event event, cl_profiling_info param_name)
{
    T      result;
    cl_int clError = clGetEventProfilingInfo(event, param_name, sizeof(T), &result, nullptr);
    if (clError != CL_SUCCESS)
    {
        GMX_THROW(gmx::InternalError("Failed to retrieve event info: " + ocl_get_error_string(clError)));
    }
    return result;
}

bool DeviceEvent::isReady() const
{
    GMX_ASSERT(isValid(), "Event must be valid in order to call .isReady()");
    auto result = getEventInfo<cl_int>(event_, CL_EVENT_COMMAND_EXECUTION_STATUS);
    return (result == CL_COMPLETE);
}

void DeviceEvent::wait()
{
    GMX_ASSERT(isValid(), "Event must be valid in order to call .wait()");
    cl_int clError = clWaitForEvents(1, &event_);
    if (CL_SUCCESS != clError)
    {
        GMX_THROW(gmx::InternalError("Failed to synchronize on the GPU event: "
                                     + ocl_get_error_string(clError)));
    }
    GMX_ASSERT(isReady(), "Event somehow not ready after clWaitForEvents");
}
bool DeviceEvent::timingSupported() const
{
    GMX_ASSERT(isValid(), "Event must be valid in order to call .timingSupported()");
    cl_int clError = clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_QUEUED, 0, nullptr, nullptr);
    if (clError != CL_SUCCESS && clError != CL_PROFILING_INFO_NOT_AVAILABLE)
    {
        GMX_THROW(gmx::InternalError("Failed to synchronize on the GPU event: "
                                     + ocl_get_error_string(clError)));
    }
    return (clError != CL_PROFILING_INFO_NOT_AVAILABLE);
}

uint64_t DeviceEvent::getExecutionTime()
{
    GMX_ASSERT(isValid(), "Event must be valid in order to call .getExecutionTime()");
    auto timeStartNanoseconds = getEventProfilingInfo<cl_ulong>(event_, CL_PROFILING_COMMAND_START);
    auto timeEndNanoseconds   = getEventProfilingInfo<cl_ulong>(event_, CL_PROFILING_COMMAND_END);
    return timeEndNanoseconds - timeStartNanoseconds;
}

const DeviceEvent::NativeType& DeviceEvent::event() const
{
    GMX_ASSERT(isValid(), "Event must be valid in order to call .event()");
    return event_;
}

void DeviceEvent::setEvent(DeviceEvent::NativeType v)
{
    if (event_ != sc_nullEvent)
    {
        clReleaseEvent(event_);
    }
    event_ = v;
}

void DeviceEvent::resetEvent()
{
    setEvent(sc_nullEvent);
}

/* static */ cl_event* DeviceEvent::getEventPtrForApiCall(DeviceEvent* deviceEvent)
{
    if (deviceEvent == nullptr)
    {
        return nullptr;
    }
    else
    {
        deviceEvent->resetEvent();
        return &deviceEvent->event_;
    }
}
