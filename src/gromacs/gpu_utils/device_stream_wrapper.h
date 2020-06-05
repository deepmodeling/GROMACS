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
#ifndef GMX_GPU_UTILS_DEVICE_STREAM_WRAPPER_H
#define GMX_GPU_UTILS_DEVICE_STREAM_WRAPPER_H

/*! \libinternal \file
 *
 * \brief Declarations for DeviceStream class.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_gpu_utils
 * \inlibraryapi
 */


#include "gromacs/utility/classhelpers.h"

struct DeviceInformation;
class DeviceContext;
class DeviceStream;

//! Enumeration describing the priority with which a stream operates.
enum class DeviceStreamPriority : int
{
    //! High-priority stream
    High,
    //! Normal-priority stream
    Normal,
    //! Conventional termination of the enumeration
    Count
};

/*! \libinternal \brief Declaration of platform-agnostic device stream/queue.
 *
 * The command stream (or command queue) is a sequence of operations that are executed
 * in they order they were issued. Several streams may co-exist to represent concurency.
 * This class declares the interfaces, that are exposed to platform-agnostic code and
 * it should be implemented for each compute architecture (e.g. CUDA and OpenCL).
 *
 * Destruction of the \p DeviceStream calls the destructor of the underlying low-level
 * stream/queue, hence should only be called when the stream is no longer needed. To
 * prevent accidental stream destruction, while copying or moving a \p DeviceStream
 * object, copy and move constructors and copy and move assignments are not allowed
 * and the \p DeviceStream object should be passed as a pointer or constant reference.
 *
 */
class DeviceStreamWrapper
{
public:
    //! Default constructor
    DeviceStreamWrapper();
    //! Destructor
    ~DeviceStreamWrapper();

    /*! \brief Initialize
     *
     * \param[in] deviceContext  Device context (not used in CUDA).
     * \param[in] priority       Stream priority: high or normal.
     * \param[in] useTiming      If the timing should be enabled (not used in CUDA).
     */
    void init(const DeviceContext& deviceContext, DeviceStreamPriority priority, bool useTiming);

    /*! \brief Construct and init.
     *
     * \param[in] deviceContext  Device context (only used in OpenCL).
     * \param[in] priority       Stream priority: high or normal (only used in CUDA).
     * \param[in] useTiming      If the timing should be enabled (only used in OpenCL).
     */
    DeviceStreamWrapper(const DeviceContext& deviceContext, DeviceStreamPriority priority, bool useTiming);

    /*! \brief Get the device stream container.
     *
     * \returns Contained device stream.
     */
    const DeviceStream& deviceStream() const;

private:
    //! Implementation type.
    class Impl;
    //! Implementation object.
    gmx::PrivateImplPointer<Impl> impl_;
};

#endif // GMX_GPU_UTILS_DEVICE_STREAM_WRAPPER_H
