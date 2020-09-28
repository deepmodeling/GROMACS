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
 *  \brief
 *
 *  SYCL non-bonded prune-only kernel. Not really a kernel yet.
 *
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "nbnxm_sycl_kernel_pruneonly.h"

#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/math/utilities.h"
#include "gromacs/nbnxm/gpu_types_common.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"

//! \internal KernelParams, using cl::sycl:accessor's
struct KernelParamsOnDevice
{
    // SYCL-TODO
    int numParts;
    int part;
};


// We need to use functor to store any kernel arguments
template<bool haveFreshList>
class NbnxmSyclKernelPruneonlyFunctor
{
public:
    NbnxmSyclKernelPruneonlyFunctor(const KernelParamsOnDevice& params_) : params(params_) {}

    //! Main kernel function
    void operator()(cl::sycl::nd_item<3> itemIdx) const;

private:
    KernelParamsOnDevice params;
};

template<bool haveFreshList>
void NbnxmSyclKernelPruneonlyFunctor<haveFreshList>::operator()(cl::sycl::nd_item<3> itemIdx) const
{
    // SYCL-TODO
}

// Specs are not very clear, but it seems that invoking kernel functors must be done in the
// same compilation unit as the definition of the kernel.
template<bool haveFreshList>
cl::sycl::event
NbnxmSyclKernelPruneonlyLauncher<haveFreshList>::launch(const struct KernelLaunchConfig& config,
                                                        const DeviceStream& deviceStream,
                                                        CommandEvent gmx_unused* timingEvent,
                                                        const struct NbnxmSyclKernelPruneonlyParams& args)
{
    const cl::sycl::range<3> globalSize{ config.gridSize[0], config.gridSize[1], config.gridSize[2] };
    const cl::sycl::range<3> localSize{ config.blockSize[0], config.blockSize[1], config.blockSize[2] };
    const cl::sycl::nd_range<3> executionRange(globalSize, localSize);

    cl::sycl::queue q = deviceStream.stream();

    cl::sycl::event e = q.submit([&](cl::sycl::handler& cgh) {
        struct KernelParamsOnDevice d_args;
        d_args.numParts = args.numParts;
        d_args.part     = args.part;
        // SYCL-TODO: Set-up necessary accessors
        auto kernel = NbnxmSyclKernelPruneonlyFunctor<haveFreshList>{ d_args };
        cgh.parallel_for(executionRange, kernel);
    });

    return e;
}

INbnxmSyclKernelPruneonlyLauncher* getNbnxmSyclKernelPruneonlyLauncher(bool haveFreshList)
{
    if (haveFreshList)
    {
        return new NbnxmSyclKernelPruneonlyLauncher<true>();
    }
    else
    {
        return new NbnxmSyclKernelPruneonlyLauncher<false>();
    }
}
