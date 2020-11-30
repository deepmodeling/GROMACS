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
#include "gromacs/nbnxm/sycl/nbnxm_sycl_kernel_utils.h"
#include "gromacs/utility/template_mp.h"

#include "nbnxm_sycl_types.h"

using cl::sycl::access::fence_space;
using cl::sycl::access::mode;
using cl::sycl::access::target;

namespace Nbnxm
{

/*! \brief Prune-only kernel for NBNXM.
 *
 */
template<bool haveFreshList>
auto nbnxmKernelPruneOnly(cl::sycl::handler&                            cgh,
                          DeviceAccessor<float4, mode::read>            a_xq,
                          DeviceAccessor<float3, mode::read>            a_shiftVec,
                          DeviceAccessor<nbnxn_cj4_t, mode::read_write> a_plistCJ4,
                          DeviceAccessor<nbnxn_sci_t, mode::read>       a_plistSci,
                          DeviceAccessor<unsigned int, haveFreshList ? mode::write : mode::read> a_plistIMask,
                          const float rlistOuterSq,
                          const float rlistInnerSq,
                          const int   numParts,
                          const int   part)
{
    cgh.require(a_xq);
    cgh.require(a_shiftVec);
    cgh.require(a_plistCJ4);
    cgh.require(a_plistSci);
    cgh.require(a_plistIMask);

    /* shmem buffer for i x+q pre-loading */
    cl::sycl::accessor<float4, 2, mode::read_write, target::local> xib(
            cl::sycl::range<2>(c_nbnxnGpuNumClusterPerSupercluster, c_clSize), cgh);

    /* the cjs buffer's use expects a base pointer offset for pairs of warps in the j-concurrent execution */
    cl::sycl::accessor<int, 2, mode::read_write, target::local> cjs(
            cl::sycl::range<2>(c_syclPruneKernelJ4Concurrency,
                               c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize),
            cgh);

    cl::sycl::stream debug(10240, 128, cgh);

    /* Requirements:
     * Work group (block) must have range (c_clSize, c_clSize, ...) (for localId calculation, easy
     * to change) Sub group (warp) must have length 8 (more complicated to change) */
    return [=](cl::sycl::nd_item<1> itemIdx) [[intel::reqd_sub_group_size(8)]]
    {
        const cl::sycl::id<3> localId = unflattenId<c_clSize, c_clSize>(itemIdx.get_local_id());
        // thread/block/warp id-s
        const unsigned        tidxi = localId[0];
        const unsigned        tidxj = localId[1];
        const cl::sycl::id<2> tidxji(localId[0], localId[1]);
        const int             tidx  = tidxj * itemIdx.get_group_range(0) + tidxi;
        const unsigned        tidxz = localId[2];
        const unsigned        bidx  = itemIdx.get_group(0);

        const sycl_pf::sub_group sg   = itemIdx.get_sub_group();
        const unsigned           widx = tidx / sg.get_local_range()[0]; /* warp index */

        // my i super-cluster's index = sciOffset + current bidx * numParts + part
        const nbnxn_sci_t nb_sci     = a_plistSci[bidx * numParts + part];
        const int         sci        = nb_sci.sci;           /* super-cluster */
        const int         cij4_start = nb_sci.cj4_ind_start; /* first ...*/
        const int         cij4_end   = nb_sci.cj4_ind_end;   /* and last index of j clusters */

        if (tidxz == 0)
        {
            for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i += c_clSize)
            {
                /* Pre-load i-atom x and q into shared memory */
                const int ci = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxj + i;
                const int ai = ci * c_clSize + tidxi;

                /*
                debug << "tidxi=" << tidxi << " tidxj=" << tidxj << " ci=" << ci <<
                        " ai=" << ai << cl::sycl::endl;
                        */

                /* We don't need q, but using float4 in shmem avoids bank conflicts.
                   (but it also wastes L2 bandwidth). */
                const float4 xq    = a_xq[ai];
                const float3 shift = a_shiftVec[nb_sci.shift];
                const float4 xi(xq[0] + shift[0], xq[1] + shift[1], xq[2] + shift[2], xq[3]);
                xib[tidxj + i][tidxi] = xi;
            }
        }
        itemIdx.barrier(fence_space::local_space);

        /* loop over the j clusters = seen by any of the atoms in the current super-cluster.
         * The loop stride c_syclPruneKernelJ4Concurrency ensures that consecutive warps-pairs are
         * assigned consecutive j4's entries. */
        for (int j4 = cij4_start + tidxz; j4 < cij4_end; j4 += c_syclPruneKernelJ4Concurrency)
        {
            unsigned imaskFull, imaskCheck, imaskNew;

            if constexpr (haveFreshList)
            {
                /* Read the mask from the list transferred from the CPU */
                imaskFull = a_plistCJ4[j4].imei[widx].imask;
                /* We attempt to prune all pairs present in the original list */
                imaskCheck = imaskFull;
                imaskNew   = 0;
            }
            else
            {
                /* Read the mask from the "warp-pruned" by rlistOuter mask array */
                imaskFull = a_plistIMask[j4 * c_nbnxnGpuClusterpairSplit + widx];
                /* Read the old rolling pruned mask, use as a base for new */
                imaskNew = a_plistCJ4[j4].imei[widx].imask;
                /* We only need to check pairs with different mask */
                imaskCheck = (imaskNew ^ imaskFull);
            }

            if ((tidxj == 0 || tidxj == c_splitClSize) && tidxi < c_nbnxnGpuJgroupSize)
            {
                cjs[tidxz][tidxi + tidxj * c_nbnxnGpuJgroupSize / c_splitClSize] =
                        a_plistCJ4[j4].cj[tidxi];
            }
            itemIdx.barrier(fence_space::local_space);

            if (imaskCheck)
            {
                for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
                {
                    if (imaskCheck & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster)))
                    {
                        unsigned mask_ji = (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster));

                        const int loadOffset =
                                (tidxj & c_splitClSize) * c_nbnxnGpuJgroupSize / c_splitClSize;
                        const int cj = cjs[tidxz][jm + loadOffset];
                        const int aj = cj * c_clSize + tidxj;

                        /* load j atom data */
                        const float4 tmp = a_xq[aj];
                        const float3 xj(tmp[0], tmp[1], tmp[2]);

                        for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
                        {
                            if (imaskCheck & mask_ji)
                            {
                                // load i-cluster coordinates from shmem
                                const float4 xi = xib[i][tidxi];
                                // distance between i and j atoms
                                float3 rv(xi[0], xi[1], xi[2]);
                                rv -= xj;
                                const float r2 = norm2(rv);
                                /* If _none_ of the atoms pairs are in rlistOuter
                                 * range, the bit corresponding to the current
                                 * cluster-pair in imask gets set to 0. */
                                if (haveFreshList && !(sycl_pf::group_any_of(sg, r2 < rlistOuterSq)))
                                {
                                    imaskFull &= ~mask_ji;
                                }
                                /* If any atom pair is within range, set the bit
                                 * corresponding to the current cluster-pair. */
                                if (sycl_pf::group_any_of(sg, r2 < rlistInnerSq))
                                {
                                    imaskNew |= mask_ji;
                                }
                            } // (imaskCheck & mask_ji)

                            /* shift the mask bit by 1 */
                            mask_ji += mask_ji;
                        } // (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
                    } // (imaskCheck & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster)))
                } // for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
                if constexpr (haveFreshList)
                {
                    /* copy the list pruned to rlistOuter to a separate buffer */
                    a_plistIMask[j4 * c_nbnxnGpuClusterpairSplit + widx] = imaskFull;
                }
                /* update the imask with only the pairs up to rlistInner */
                a_plistCJ4[j4].imei[widx].imask = imaskNew;
            } // (imaskCheck)
            itemIdx.barrier(fence_space::local_space);
        } // for (int j4 = cij4_start + tidxz; j4 < cij4_end; j4 += c_syclPruneKernelJ4Concurrency)
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
        cgh.parallel_for<kernelNameType>(flattenNDRange(range), kernel);
    });

    return e;
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
            plist->sci, plist->imask, nbp->rlistOuter_sq, nbp->rlistInner_sq, numParts, part);

    e.wait_and_throw(); // SYCL-TODO: remove
}

} // namespace Nbnxm
