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
 *  NBNXM SYCL kernels
 *
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "nbnxm_sycl_kernel_pruneonly.h"

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/utility/template_mp.h"

#include "nbnxm_sycl_types.h"

/*! \brief Macro defining default for the prune kernel's j4 processing concurrency.
 *
 *  The GMX_NBNXN_PRUNE_KERNEL_J4_CONCURRENCY macro allows compile-time override.
 */
#ifndef GMX_NBNXN_PRUNE_KERNEL_J4_CONCURRENCY
#    define GMX_NBNXN_PRUNE_KERNEL_J4_CONCURRENCY 4
#endif
static constexpr int c_syclPruneKernelJ4Concurrency = GMX_NBNXN_PRUNE_KERNEL_J4_CONCURRENCY;

/*! \brief cluster size = number of atoms per cluster. */
static constexpr int c_clSize = c_nbnxnGpuClusterSize;

namespace Nbnxm
{

using cl::sycl::access::mode;

/*! \brief Prune-only kernel for NBNXM.
 *
 */
template<bool haveFreshList>
auto nbnxmKernelPruneOnly(cl::sycl::handler&                            cgh,
                          DeviceAccessor<float4, mode::read>            a_xq,
                          DeviceAccessor<float3, mode::read>            a_shiftVec,
                          DeviceAccessor<nbnxn_cj4_t, mode::read_write> a_plistCJ4,
                          DeviceAccessor<nbnxn_sci_t, mode::read>       a_plistSci,
                          const float gmx_unused rlistOuterSq,
                          const float gmx_unused rlistInnerSq,
                          const int gmx_unused numParts,
                          const int gmx_unused part)
{
    cgh.require(a_xq);
    cgh.require(a_shiftVec);
    cgh.require(a_plistCJ4);
    cgh.require(a_plistSci);

    return [=](cl::sycl::nd_item<3> gmx_unused itemIdx) {

    };
}

// SYCL 1.2.1 requires providing a unique type for a kernel. Should not be needed for SYCL2020.
template<bool haveFreshList>
class NbnxmKernelPruneOnlyName;

template<bool haveFreshList, class... Args>
cl::sycl::event launchNbnxmKernelPruneOnly(const DeviceStream& deviceStream,
                                           const int           numSciInPart,
                                           Args&&... args)
{
    // Should not be needed for SYCL2020.
    using kernelNameType = NbnxmKernelPruneOnlyName<haveFreshList>;

    /* Kernel launch config:
     * - The thread block dimensions match the size of i-clusters, j-clusters,
     *   and j-cluster concurrency, in x, y, and z, respectively.
     * - The 1D block-grid contains as many blocks as super-clusters.
     */
    const unsigned long         numBlocks = numSciInPart;
    const cl::sycl::range<3>    blockSize{ c_clSize, c_clSize, c_syclPruneKernelJ4Concurrency };
    const cl::sycl::range<3>    globalSize{ numBlocks * blockSize[0], blockSize[1], blockSize[2] };
    const cl::sycl::nd_range<3> range{ globalSize, blockSize };

    cl::sycl::queue q = deviceStream.stream();

    cl::sycl::event e = q.submit([&](cl::sycl::handler& cgh) {
        auto kernel = nbnxmKernelPruneOnly<haveFreshList>(cgh, std::forward<Args>(args)...);
        cgh.parallel_for<kernelNameType>(range, kernel);
    });

    GMX_THROW(gmx::NotImplementedError("Not yet implemented for SYCL"));
}

template<class... Args>
cl::sycl::event chooseAndLaunchNbnxmKernelPruneOnly(bool haveFreshList, Args&&... args)
{
    return gmx::dispatchTemplatedFunction(
            [&](auto haveFreshList_) {
                return launchNbnxmKernelPruneOnly<haveFreshList_>(std::forward<Args>(args)...);
            },
            haveFreshList);
}

void launchNbnxmKernelPruneOnly(NbnxmGpu*                 nb,
                                const InteractionLocality iloc,
                                const int                 numParts,
                                const int                 part,
                                const int                 numSciInPart)
{
    sycl_atomdata_t*    adat          = nb->atdat;
    NBParamGpu*         nbp           = nb->nbparam;
    gpu_plist*          plist         = nb->plist[iloc];
    const bool          haveFreshList = plist->haveFreshList;
    const DeviceStream& deviceStream  = *nb->deviceStreams[iloc];

    cl::sycl::event e = chooseAndLaunchNbnxmKernelPruneOnly(
            haveFreshList, deviceStream, numSciInPart, adat->xq, adat->shiftVec, plist->cj4,
            plist->sci, nbp->rlistOuter_sq, nbp->rlistInner_sq, numParts, part);
}

} // namespace Nbnxm
