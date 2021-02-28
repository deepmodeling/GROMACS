/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020,2021, by the GROMACS development team, led by
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
 *  \brief Declare utility routines for SYCL
 *
 *  \author Andrey Alekseenko <al42and@gmail.com>
 *  \inlibraryapi
 */
#ifndef GMX_GPU_UTILS_SYCLUTILS_H
#define GMX_GPU_UTILS_SYCLUTILS_H

#include <string>

#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

class DeviceEvent;
class DeviceStream;
enum class GpuApiCallBehavior;

/*! \internal
 * \brief SYCL GPU runtime data
 *
 * The device runtime data is meant to hold objects associated with a GROMACS rank's
 * (thread or process) use of a single device (multiple devices per rank is not
 * implemented). These objects should be constructed at the point where a device
 * gets assigned to a rank and released at when this assignment is no longer valid
 * (i.e. at cleanup in the current implementation).
 */
struct gmx_device_runtime_data_t
{
};

#ifndef DOXYGEN

/*! \brief Allocate host memory in malloc style */
void pmalloc(void** h_ptr, size_t nbytes);

/*! \brief Free host memory in malloc style */
void pfree(void* h_ptr);

/* To properly mark function as [[noreturn]], we must do it everywhere it is declared, which
 * will pollute common headers.*/
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-noreturn"

/*! \brief
 * A wrapper function for setting up all the SYCL kernel arguments.
 * Calls the recursive functions above.
 *
 * \tparam    Args            Types of all the kernel arguments
 * \param[in] kernel          Kernel function handle
 * \param[in] config          Kernel configuration for launching
 * \param[in] argsPtrs        Pointers to all the kernel arguments
 * \returns A handle for the prepared parameter pack to be used with launchGpuKernel() as the last argument.
 */
template<typename... Args>
void* prepareGpuKernelArguments(void* /*kernel*/, const KernelLaunchConfig& /*config*/, const Args*... /*argsPtrs*/)
{
    GMX_THROW(gmx::NotImplementedError("Not implemented on SYCL yet"));
}

/*! \brief Launches the SYCL kernel and handles the errors.
 *
 * \param[in] kernel          Kernel function handle
 * \param[in] config          Kernel configuration for launching
 * \param[in] deviceStream    GPU stream to launch kernel in
 * \param[in] timingEvent     Timing event, fetched from GpuRegionTimer
 * \param[in] kernelName      Human readable kernel description, for error handling only
 * \throws gmx::InternalError on kernel launch failure
 */
inline void launchGpuKernel(void* /*kernel*/,
                            const KernelLaunchConfig& /*config*/,
                            const DeviceStream& /*deviceStream*/,
                            DeviceEvent* /*timingEvent*/,
                            const char* /*kernelName*/,
                            const void* /*kernelArgs*/)
{
    GMX_THROW(gmx::NotImplementedError("Not implemented on SYCL yet"));
}

/*! \brief Pretend to check a SYCL stream for unfinished work (dummy implementation).
 *
 *  \returns  Not implemented in SYCL.
 */
static inline bool haveStreamTasksCompleted(const DeviceStream& /* deviceStream */)
{
    GMX_THROW(gmx::NotImplementedError("Not implemented on SYCL yet"));
}

#    pragma clang diagnostic pop

#endif // !DOXYGEN

#endif
