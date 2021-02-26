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
 *  \brief Defines DeviceEvent class for SYCL.
 *
 *  \author Andrey Alekseenko <al42and@gmail.com>
 *  \inlibraryapi
 */

#include "gmxpre.h"

#include "device_event.h"

DeviceEvent::DeviceEvent() : event_(std::nullopt) {}
DeviceEvent::DeviceEvent(DeviceEvent::NativeType event) : event_(event) {}
DeviceEvent::DeviceEvent(DeviceEvent&& other) noexcept :
    event_(std::exchange(other.event_, std::nullopt))
{
}

DeviceEvent::~DeviceEvent() = default;

bool DeviceEvent::isValid() const
{
    return event_.has_value();
}

bool DeviceEvent::isReady() const
{
    using namespace cl::sycl::info;
    GMX_ASSERT(isValid(), "Event must be valid in order to call .isReady()");
    return event_->get_info<event::command_execution_status>() == event_command_status::complete;
}

void DeviceEvent::wait()
{
    GMX_ASSERT(isValid(), "Event must be valid in order to call .wait()");
    event_->wait_and_throw();
    GMX_ASSERT(isReady(), "Event somehow not ready after sycl::event::wait_and_throw");
}
bool DeviceEvent::timingSupported() const
{
    GMX_ASSERT(isValid(), "Event must be valid in order to call .timingSupported()");
    using namespace cl::sycl::info;
    try
    {
        event_->get_profiling_info<event_profiling::command_submit>();
    }
    catch (cl::sycl::exception&)
    {
        return false;
    }
    return true;
}

uint64_t DeviceEvent::getExecutionTime()
{
    GMX_ASSERT(isValid(), "Event must be valid in order to call .getExecutionTime()");
    using namespace cl::sycl::info;
    uint64_t timeStartNanoseconds = event_->get_profiling_info<event_profiling::command_start>();
    uint64_t timeEndNanoseconds   = event_->get_profiling_info<event_profiling::command_end>();
    return timeEndNanoseconds - timeStartNanoseconds;
}

const DeviceEvent::NativeType& DeviceEvent::getNative() const
{
    GMX_ASSERT(isValid(), "Event must be valid in order to call .getNative()");
    return event_.value();
}

void DeviceEvent::setNative(DeviceEvent::NativeType v)
{
    event_ = v;
}

void DeviceEvent::resetNative()
{
    event_.reset();
}
