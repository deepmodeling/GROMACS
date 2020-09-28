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
 *  Stubs of functions that must be defined by nbnxm sycl implementation.
 *
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "gromacs/nbnxm/gpu_common.h"
#include "gromacs/utility/exceptions.h"

#include "nbnxm_sycl_kernel.h"
#include "nbnxm_sycl_kernel_pruneonly.h"
#include "nbnxm_sycl_types.h"

namespace Nbnxm
{

/*! \brief Convenience constants */
//@{
static constexpr int c_blockSize = c_nbnxnGpuClusterSize;
//@}

/*! \brief
 * Launch asynchronously the download of nonbonded forces from the GPU
 * (and energies/shift forces if required).
 */
void gpu_launch_cpyback(NbnxmGpu*                nb,
                        struct nbnxn_atomdata_t* nbatom,
                        const gmx::StepWorkload& stepWork,
                        const AtomLocality       atomLocality)
{
    GMX_ASSERT(nb, "Need a valid nbnxn_gpu object");
    GMX_ASSERT(!stepWork.useGpuFBufferOps, "GpuBufferOps not supported on SYCL");

    int adat_begin, adat_len; /* local/nonlocal offset and length used for xq and f */

    /* determine interaction locality from atom locality */
    const InteractionLocality iloc = gpuAtomToInteractionLocality(atomLocality);

    /* extract the data */
    sycl_atomdata_t*    adat         = nb->atdat;
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    /* don't launch non-local copy-back if there was no non-local work to do */
    if ((iloc == InteractionLocality::NonLocal) && !haveGpuShortRangeWork(*nb, iloc))
    {
        return;
    }

    getGpuAtomRange(adat, atomLocality, &adat_begin, &adat_len);

    /* With DD the local D2H transfer can only start after the non-local
       kernel has finished. */
    if (iloc == InteractionLocality::Local && nb->bUseTwoStreams)
    {
        // SYCL-TODO: Remove in favor of data-dependency-based scheduling
        nb->nonlocal_done.waitForEvent();
    }

    /* DtoH f */
    GMX_ASSERT(adat->f.elementSize() == sizeof(float3),
               "The size of the force buffer element should be equal to the size of float3.");
    copyFromDeviceBuffer(reinterpret_cast<float3*>(nbatom->out[0].f.data()) + adat_begin, &adat->f,
                         adat_begin, adat_len, deviceStream, GpuApiCallBehavior::Async, nullptr);

    /* After the non-local D2H is launched the nonlocal_done event can be
       recorded which signals that the local D2H can proceed. This event is not
       placed after the non-local kernel because we want the non-local data
       back first. */
    if (iloc == InteractionLocality::NonLocal)
    {
        nb->nonlocal_done.enqueueWaitEvent(deviceStream);
    }

    /* only transfer energies in the local stream */
    if (iloc == InteractionLocality::Local)
    {
        /* DtoH fshift when virial is needed */
        if (stepWork.computeVirial)
        {
            GMX_ASSERT(sizeof(nb->nbst.fshift[0]) == adat->fshift.elementSize(),
                       "Sizes of host- and device-side shift vectors should be the same.");
            copyFromDeviceBuffer(reinterpret_cast<float*>(nb->nbst.fshift), &adat->fshift, 0,
                                 SHIFTS, deviceStream, GpuApiCallBehavior::Async, nullptr);
        }

        /* DtoH energies */
        if (stepWork.computeEnergy)
        {
            GMX_ASSERT(sizeof(nb->nbst.e_lj[0]) == adat->e_lj.elementSize(),
                       "Sizes of host- and device-side LJ energy terms should be the same.");
            copyFromDeviceBuffer(nb->nbst.e_lj, &adat->e_lj, 0, 1, deviceStream,
                                 GpuApiCallBehavior::Async, nullptr);
            GMX_ASSERT(sizeof(nb->nbst.e_el[0]) == adat->e_el.elementSize(),
                       "Sizes of host- and device-side electrostatic energy terms should be the "
                       "same.");
            copyFromDeviceBuffer(nb->nbst.e_el, &adat->e_el, 0, 1, deviceStream,
                                 GpuApiCallBehavior::Async, nullptr);
        }
    }
}

/*! \brief Launch asynchronously the xq buffer host to device copy. */
void gpu_copy_xq_to_gpu(NbnxmGpu* nb, const nbnxn_atomdata_t* nbatom, const AtomLocality atomLocality)
{
    GMX_ASSERT(nb, "Need a valid nbnxn_gpu object");
    GMX_ASSERT(atomLocality == AtomLocality::Local || atomLocality == AtomLocality::NonLocal,
               "Only local and non-local xq transfers are supported");

    const InteractionLocality iloc = gpuAtomToInteractionLocality(atomLocality);

    int adat_begin, adat_len; /* local/nonlocal offset and length used for xq and f */

    sycl_atomdata*      adat         = nb->atdat;
    gpu_plist*          plist        = nb->plist[iloc];
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    /* Don't launch the non-local H2D copy if there is no dependent
       work to do: neither non-local nor other (e.g. bonded) work
       to do that has as input the nbnxn coordaintes.
       Doing the same for the local kernel is more complicated, since the
       local part of the force array also depends on the non-local kernel.
       So to avoid complicating the code and to reduce the risk of bugs,
       we always call the local local x+q copy (and the rest of the local
       work in nbnxn_gpu_launch_kernel().
     */
    if ((iloc == InteractionLocality::NonLocal) && !haveGpuShortRangeWork(*nb, iloc))
    {
        plist->haveFreshList = false;
        return;
    }

    /* calculate the atom data index range based on locality */
    if (atomLocality == AtomLocality::Local)
    {
        adat_begin = 0;
        adat_len   = adat->natoms_local;
    }
    else
    {
        adat_begin = adat->natoms_local;
        adat_len   = adat->natoms - adat->natoms_local;
    }

    /* HtoD x, q */
    GMX_ASSERT(adat->xq.elementSize() == sizeof(float4),
               "The size of the xyzq buffer element should be equal to the size of float4.");
    copyToDeviceBuffer(&adat->xq, reinterpret_cast<const float4*>(nbatom->x().data()) + adat_begin,
                       adat_begin, adat_len, deviceStream, GpuApiCallBehavior::Async, nullptr);
}

void gpu_launch_kernel_pruneonly(NbnxmGpu* nb, const InteractionLocality iloc, const int numParts)
{
    sycl_atomdata*      adat         = nb->atdat;
    NBParamGpu*         nbp          = nb->nbparam;
    gpu_plist*          plist        = nb->plist[iloc];
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    if (plist->haveFreshList)
    {
        GMX_ASSERT(numParts == 1, "With first pruning we expect 1 part");

        /* Set rollingPruningNumParts to signal that it is not set */
        plist->rollingPruningNumParts = 0;
        plist->rollingPruningPart     = 0;
    }
    else
    {
        if (plist->rollingPruningNumParts == 0)
        {
            plist->rollingPruningNumParts = numParts;
        }
        else
        {
            GMX_ASSERT(numParts == plist->rollingPruningNumParts,
                       "It is not allowed to change numParts in between list generation steps");
        }
    }

    /* Use a local variable for part and update in plist, so we can return here
     * without duplicating the part increment code.
     */
    const int part = plist->rollingPruningPart;

    plist->rollingPruningPart++;
    if (plist->rollingPruningPart >= plist->rollingPruningNumParts)
    {
        plist->rollingPruningPart = 0;
    }

    /* Compute the number of list entries to prune in this pass */
    const int numSciInPart = (plist->nsci - part) / numParts;

    /* Don't launch the kernel if there is no work to do (not allowed with CUDA) */
    if (numSciInPart <= 0)
    {
        plist->haveFreshList = false;
        return;
    }

    /* Kernel launch config:
     * - The thread block dimensions match the size of i-clusters, j-clusters,
     *   and j-cluster concurrency, in x, y, and z, respectively.
     * - The 1D block-grid contains as many blocks as super-clusters.
     */
    // SYCL-TODO: Set properly
    KernelLaunchConfig config;
    config.blockSize[0]     = c_blockSize;
    config.blockSize[1]     = c_blockSize;
    config.gridSize[0]      = plist->nsci;
    config.sharedMemorySize = 0;

    if (debug)
    {
        fprintf(debug,
                "Pruning GPU kernel launch configuration:\n\tThread block: %zux%zux%zu\n\t"
                "\tGrid: %zux%zu\n\t#Super-clusters/clusters: %d/%d (%d)\n"
                "\tShMem: %zu\n",
                config.blockSize[0], config.blockSize[1], config.blockSize[2], config.gridSize[0],
                config.gridSize[1], numSciInPart * c_nbnxnGpuNumClusterPerSupercluster,
                c_nbnxnGpuNumClusterPerSupercluster, plist->na_c, config.sharedMemorySize);
    }

    NbnxmSyclKernelPruneonlyParams kernelArgs(adat, nbp, plist, numParts, part);
    auto* kernelLauncher = getNbnxmSyclKernelPruneonlyLauncher(plist->haveFreshList);

    kernelLauncher->launch(config, deviceStream, nullptr, kernelArgs);

    /* TODO: consider a more elegant way to track which kernel has been called
       (combined or separate 1st pass prune, rolling prune). */
    if (plist->haveFreshList)
    {
        plist->haveFreshList = false;
    }
}

/*! As we execute nonbonded workload in separate streams, before launching
   the kernel we need to make sure that he following operations have completed:
   - atomdata allocation and related H2D transfers (every nstlist step);
   - pair list H2D transfer (every nstlist step);
   - shift vector H2D transfer (every nstlist step);
   - force (+shift force and energy) output clearing (every step).

   SYCL-TODO: Update this description

   These operations are issued in the local stream at the beginning of the step
   and therefore always complete before the local kernel launch. The non-local
   kernel is launched after the local on the same device/context hence it is
   inherently scheduled after the operations in the local stream (including the
   above "misc_ops") on pre-GK110 devices with single hardware queue, but on later
   devices with multiple hardware queues the dependency needs to be enforced.
   We use the misc_ops_and_local_H2D_done event to record the point where
   the local x+q H2D (and all preceding) tasks are complete and synchronize
   with this event in the non-local stream before launching the non-bonded kernel.
 */
void gpu_launch_kernel(NbnxmGpu* nb, const gmx::StepWorkload& stepWork, const Nbnxm::InteractionLocality iloc)
{
    sycl_atomdata_t*    adat         = nb->atdat;
    NBParamGpu*         nbp          = nb->nbparam;
    gpu_plist*          plist        = nb->plist[iloc];
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    /* Don't launch the non-local kernel if there is no work to do.
       Doing the same for the local kernel is more complicated, since the
       local part of the force array also depends on the non-local kernel.
       So to avoid complicating the code and to reduce the risk of bugs,
       we always call the local kernel, and later (not in
       this function) the stream wait, local f copyback and the f buffer
       clearing. All these operations, except for the local interaction kernel,
       are needed for the non-local interactions. The skip of the local kernel
       call is taken care of later in this function. */
    if (canSkipNonbondedWork(*nb, iloc))
    {
        plist->haveFreshList = false;
        return;
    }

    if (nbp->useDynamicPruning && plist->haveFreshList)
    {
        /* Prunes for rlistOuter and rlistInner, sets plist->haveFreshList=false
           (TODO: ATM that's the way the timing accounting can distinguish between
           separate prune kernel and combined force+prune, maybe we need a better way?).
         */
        gpu_launch_kernel_pruneonly(nb, iloc, 1);
    }

    if (plist->nsci == 0)
    {
        /* Don't launch an empty local kernel */
        return;
    }

    /* Kernel launch config:
     * - The thread block dimensions match the size of i-clusters, j-clusters,
     *   and j-cluster concurrency, in x, y, and z, respectively.
     * - The 1D block-grid contains as many blocks as super-clusters.
     */
    // SYCL-TODO: Set up properly
    KernelLaunchConfig config;
    config.blockSize[0]     = c_blockSize;
    config.blockSize[1]     = c_blockSize;
    config.gridSize[0]      = plist->nsci;
    config.sharedMemorySize = 0;

    const bool doPrune = (plist->haveFreshList && !nb->timers->interaction[iloc].didPrune);

    NbnxmSyclKernelParams args(adat, nbp, plist, stepWork.computeVirial);

    auto* kernelLauncher = getNbnxmSyclKernelLauncher(doPrune, stepWork.computeEnergy,
                                                      static_cast<eelType>(nbp->eeltype),
                                                      static_cast<evdwType>(nbp->vdwtype));

    kernelLauncher->launch(config, deviceStream, nullptr, args);
}

} // namespace Nbnxm
