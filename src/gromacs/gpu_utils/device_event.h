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
#ifndef GMX_GPU_UTILS_DEVICE_EVENT_H
#define GMX_GPU_UTILS_DEVICE_EVENT_H

/*! \libinternal \file
 *  \brief Declares DeviceEvent class.
 *
 *  \author Andrey Alekseenko <al42and@gmail.com>
 *  \inlibraryapi
 */

#include "config.h"

#include <optional>
#include <iostream>

#include "gromacs/utility/gmxassert.h"

#if GMX_GPU_OPENCL
#    include "gromacs/gpu_utils/gmxopencl.h"
#endif
#if GMX_GPU_SYCL
#    include "gromacs/gpu_utils/gmxsycl.h"
#endif


/*! \libinternal \brief
 * Class encapsulating backend-specific GPU Event objects for synchronization with consumption counter.
 *
 * Please note, that different approaches to device handling in backends lead to the differences
 * in how this object behaves on different platforms.
 *
 * The class serves as a wrapper around native event objects (\c cudaEvent_t in CUDA, \c cl_event
 * in OpenCL, and \c sycl::event in SYCL).
 * It has to balance different approaches to device events:
 * - In CUDA, the event is explicitly created, and then can be enqueued into the device stream,
 * where it serves as a marker, not associated with any specific operation. The only available
 * timing info is the time the event was triggered.
 * - In OpenCL each enqueued operation can return an event associated with it, which can be used
 * to check the status of the operation and its duration.
 * - SYCL is similar to OpenCL, except an event is always returned.
 *
 * Due to the way CUDA works, we opt to create a few persistent DeviceEvents, rather than a new
 * event for each operation.
 * Thus, in CUDA the underlying \c cudaEvent_t object is reused when we want to enqueue a new marker.
 * In OpenCL and SYCL, the underlying object is replaced each time we want to update it.
 */
class DeviceEvent
{
public:
#if GMX_GPU_CUDA
    using NativeType = cudaEvent_t;
#elif GMX_GPU_OPENCL
    using NativeType = cl_event;
#elif GMX_GPU_SYCL
    using NativeType = cl::sycl::event;
#endif

    //! Construct uninitialized event in OpenCL/SYCL, initialized but unsubmitted in CUDA.
    DeviceEvent();
    //! Construct from native event. Take ownership of it.
    DeviceEvent(DeviceEvent::NativeType event);
    DeviceEvent(const DeviceEvent&) = delete;
    DeviceEvent& operator=(const DeviceEvent&) = delete;
    DeviceEvent(DeviceEvent&& other) noexcept;
    DeviceEvent& operator=(DeviceEvent&& other) noexcept
    {
        std::swap(event_, other.event_);
        return *this;
    }
    ~DeviceEvent();

    // Check status of the object
    /*! Return true if the event is fully initialized.
     *
     * Always \c true in CUDA.
     *
     * In CUDA, it makes more sense to create an event once and reuse it.
     * In OpenCL/SYCL, each separate function call returns a new event object.
     * This makes it problematic to ensure the validity of the object at construction.
     */
    [[nodiscard]] bool isValid() const;
    /*! Return true if the enqueued event is ready/complete.
     *
     * In CUDA, that means that the marked event has been reached.
     * In OpenCL and SYCL, that means that the associated operation has completed.
     * Undefined behavior if the event is not valid.
     */
    [[nodiscard]] bool isReady() const;

    /*! Wait (block the current thread) until the even is ready.
     *
     * Undefined behavior if the event is not valid.
     */
    void wait();

    /*! Return true if the event supports measuring the execution time of associated operation.
     *
     * CUDA: Always \c false.
     *
     * OpenCL and SYCL: \c true if the queue was constructed with profiling enabled, \c false
     * otherwise. This function does some API calls internally and does not cache their result,
     * so it is more expensive than just checking a boolean.
     *
     * Undefined behavior if the event is not valid.
     */
    [[nodiscard]] bool timingSupported() const;
    /*! Return the execution time of associated event.
     *
     * Wait for the completion of the event,
     * Undefined behavior if the event is not valid or the event does not support timing (see
     * \ref timingSupported).
     *
     * \return Execution time of the associated event in nanoseconds.
     */
    uint64_t getExecutionTime();

    /*! Return the backend-specific native object (as a const reference).
     *
     * Undefined behavior if the event is not valid.
     */
    [[nodiscard]] const NativeType& getNative() const;
#if GMX_GPU_OPENCL || GMX_GPU_SYCL || defined(DOXYGEN)
    //! Set the internal native object to \p v. Release previously stored object, if any.
    void setNative(NativeType v);
    //! Release the backend-specific native object, if any. Set \ref isValid to false.
    void resetNative();
#endif

#if GMX_GPU_OPENCL || defined(DOXYGEN)
    /*! Convert \c DeviceEvent* to \c cl_event* that can be to OpenCL API calls returning an event.
     *
     * Many OpenCL API calls take a pointer to \c cl_event to return a handle of an event associated
     * with this API call. This pointer can be \c NULL, in which case no event is returned.
     *
     * This function is intended to convert \ref DeviceEvent class for exactly such uses.
     *
     * If \p deviceEvent is \c nullptr, it returns \c nullptr.
     * Otherwise, it releases the native event stored in \p deviceEvent (if any), and returns
     * a pointer to the private native object (now not containing anything). When API calls
     * returns an event, it will be placed in the \p deviceEvent, which will now own it.
     *
     * A pointer returned must be used exactly once. This is not enforced programmatically.
     * Please, use this function carefully and preferably in a narrow scope.
     */
    static cl_event* getNativePtrForApiCall(DeviceEvent* deviceEvent);
#endif

private:
#if GMX_GPU_CUDA
    cudaEvent_t event_;
#elif GMX_GPU_OPENCL
    cl_event event_;
#elif GMX_GPU_SYCL
    std::optional<cl::sycl::event> event_;
#endif
};

#endif // GMX_GPU_UTILS_DEVICE_EVENT_H
