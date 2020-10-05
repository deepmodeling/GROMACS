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
 *  SYCL non-bonded kernel. Not really a kernel yet.
 *
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "nbnxm_sycl_kernel.h"

#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/math/utilities.h"
#include "gromacs/nbnxm/gpu_types_common.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"

//! \internal Twin of NbnxmSyclKernelParams, but using cl::sycl:accessor's
struct KernelParamsOnDevice
{
};

// We need to use functor to store any kernel arguments
template<bool doPruneNBL, bool doCalcEnergies, enum eelType flavorEL, enum evdwType flavorLJ>
class NbnxmSyclKernelFunctor
{
public:
    NbnxmSyclKernelFunctor(cl::sycl::handler& cgh, const NbnxmSyclKernelParams& params);

    //! Main kernel function
    void operator()(cl::sycl::nd_item<3> itemIdx) const;

private:
    int                                                               natoms_;
    int                                                               natoms_local_;
    cl::sycl::accessor<float4, 1, cl::sycl::access::mode::read>       xq_;
    cl::sycl::accessor<float3, 1, cl::sycl::access::mode::read_write> f_;
    cl::sycl::accessor<float3, 1, cl::sycl::access::mode::read_write> f_shift_;
    cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write>  e_el_;
    cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write>  e_lj_;
    // SYCL-TODO: other types
    bool bCalcFshift_;
};

template<bool doPruneNBL, bool doCalcEnergies, enum eelType flavorEL, enum evdwType flavorLJ>
NbnxmSyclKernelFunctor<doPruneNBL, doCalcEnergies, flavorEL, flavorLJ>::NbnxmSyclKernelFunctor(
        cl::sycl::handler&           cgh,
        const NbnxmSyclKernelParams& params) :
    natoms_(params.atomdata->natoms),
    natoms_local_(params.atomdata->natoms_local),
    xq_(params.atomdata->xq.buffer_->get_access<cl::sycl::access::mode::read>(cgh)),
    f_(params.atomdata->f.buffer_->get_access<cl::sycl::access::mode::read_write>(cgh)),
    f_shift_(params.atomdata->fshift.buffer_->get_access<cl::sycl::access::mode::read_write>(cgh)),
    e_el_(params.atomdata->e_el.buffer_->get_access<cl::sycl::access::mode::read_write>(cgh)),
    e_lj_(params.atomdata->e_lj.buffer_->get_access<cl::sycl::access::mode::read_write>(cgh)),
    // SYCL-TODO: initialize the rest of the fields
    bCalcFshift_(params.bCalcFshift)
{
}

template<bool doPruneNBL, bool doCalcEnergies, enum eelType flavorEL, enum evdwType flavorLJ>
void NbnxmSyclKernelFunctor<doPruneNBL, doCalcEnergies, flavorEL, flavorLJ>::
     operator()(cl::sycl::nd_item<3> itemIdx) const
{
    // Replacements of macros
    constexpr bool flavorELCutoff = (flavorEL == eelTypeCUT);
    constexpr bool flavorELRF     = (flavorEL == eelTypeRF);

    constexpr bool flavorELEwaldAna = (flavorEL == eelTypeEWALD_ANA || flavorEL == eelTypeEWALD_ANA_TWIN);
    constexpr bool flavorELEwaldTab = (flavorEL == eelTypeEWALD_TAB || flavorEL == eelTypeEWALD_TAB_TWIN);
    constexpr bool flavorELEwaldAny = (flavorELEwaldAna || flavorELEwaldTab);

    constexpr bool flavorLJCombGeom      = (flavorLJ == evdwTypeCUTCOMBGEOM);
    constexpr bool flavorLJCombLB        = (flavorLJ == evdwTypeCUTCOMBLB);
    constexpr bool flavorLJEwaldCombGeom = (flavorLJ == evdwTypeEWALDGEOM);
    constexpr bool flavorLJEwaldCombLB   = (flavorLJ == evdwTypeEWALDLB);
    constexpr bool flavorLJForceSwitch   = (flavorLJ == evdwTypeFSWITCH);
    constexpr bool flavorLJPotSwitch     = (flavorLJ == evdwTypePSWITCH);

    constexpr bool flavorLJEwald = (flavorLJEwaldCombGeom || flavorLJEwaldCombLB);
    constexpr bool flavorLJComb  = (flavorLJCombGeom || flavorLJCombLB);

    constexpr bool doExclusionForces =
            (flavorELEwaldAny || flavorELRF || flavorLJEwald || (flavorELCutoff && doCalcEnergies));

    // SYCL-TODO: The code below just sets arbitrary values
    const int numThreads     = itemIdx.get_global_range().size();
    const int atomsPerThread = natoms_ / numThreads + 1;
    const int threadId       = itemIdx.get_global_linear_id();
    const int atStart        = threadId * atomsPerThread;
    const int atEnd          = cl::sycl::min((threadId + 1) * atomsPerThread, natoms_);
    for (int at = atStart; at < atEnd; at++)
    {
        f_[at] = { 0, xq_[at].y(), at };
    }
}

// Specs are not very clear, but it seems that invoking kernel functors must be done in the
// same compilation unit as the definition of the kernel.
template<bool doPruneNBL, bool doCalcEnergies, enum eelType flavorEL, enum evdwType flavorLJ>
cl::sycl::event NbnxmSyclKernelLauncher<doPruneNBL, doCalcEnergies, flavorEL, flavorLJ>::launch(
        const struct KernelLaunchConfig& config,
        const DeviceStream&              deviceStream,
        CommandEvent gmx_unused*            timingEvent,
        const struct NbnxmSyclKernelParams& args)
{
    const cl::sycl::range<3> globalSize{ config.gridSize[0], config.gridSize[1], config.gridSize[2] };
    const cl::sycl::range<3> localSize{ config.blockSize[0], config.blockSize[1], config.blockSize[2] };
    const cl::sycl::nd_range<3> executionRange(globalSize, localSize);

    cl::sycl::queue q = deviceStream.stream();

    cl::sycl::event e = q.submit([&](cl::sycl::handler& cgh) {
        // SYCL-TODO: Set-up remaining accessors
        auto kernel = NbnxmSyclKernelFunctor<doPruneNBL, doCalcEnergies, flavorEL, flavorLJ>{ cgh, args };
        cgh.parallel_for(executionRange, kernel);
    });

    return e;
}

template<bool doPruneNBL, bool doCalcEnergies, enum eelType flavorEL, enum evdwType flavorLJ>
static inline INbnxmSyclKernelLauncher* getLauncher()
{
    return new NbnxmSyclKernelLauncher<doPruneNBL, doCalcEnergies, flavorEL, flavorLJ>();
}

template<bool doPruneNBL, bool doCalcEnergies, enum eelType flavorEL>
static inline INbnxmSyclKernelLauncher* getLauncher(enum evdwType flavorLJ)
{
    switch (flavorLJ)
    {
        case evdwTypeCUT: return getLauncher<doPruneNBL, doCalcEnergies, flavorEL, evdwTypeCUT>();
        case evdwTypeCUTCOMBGEOM:
            return getLauncher<doPruneNBL, doCalcEnergies, flavorEL, evdwTypeCUTCOMBGEOM>();
        case evdwTypeCUTCOMBLB:
            return getLauncher<doPruneNBL, doCalcEnergies, flavorEL, evdwTypeCUTCOMBLB>();
        case evdwTypeFSWITCH:
            return getLauncher<doPruneNBL, doCalcEnergies, flavorEL, evdwTypeFSWITCH>();
        case evdwTypePSWITCH:
            return getLauncher<doPruneNBL, doCalcEnergies, flavorEL, evdwTypePSWITCH>();
        case evdwTypeEWALDGEOM:
            return getLauncher<doPruneNBL, doCalcEnergies, flavorEL, evdwTypeEWALDGEOM>();
        case evdwTypeEWALDLB:
            return getLauncher<doPruneNBL, doCalcEnergies, flavorEL, evdwTypeEWALDLB>();
        case evdwTypeNR: return getLauncher<doPruneNBL, doCalcEnergies, flavorEL, evdwTypeNR>();
        default: GMX_THROW(gmx::InternalError("Invalid LJ flavor"));
    }
}


template<bool doPruneNBL, bool doCalcEnergies>
static inline INbnxmSyclKernelLauncher* getLauncher(enum eelType flavorEL, enum evdwType flavorLJ)
{
    switch (flavorEL)
    {
        case eelTypeCUT: return getLauncher<doPruneNBL, doCalcEnergies, eelTypeCUT>(flavorLJ);
        case eelTypeRF: return getLauncher<doPruneNBL, doCalcEnergies, eelTypeRF>(flavorLJ);
        case eelTypeEWALD_TAB:
            return getLauncher<doPruneNBL, doCalcEnergies, eelTypeEWALD_TAB>(flavorLJ);
        case eelTypeEWALD_TAB_TWIN:
            return getLauncher<doPruneNBL, doCalcEnergies, eelTypeEWALD_TAB_TWIN>(flavorLJ);
        case eelTypeEWALD_ANA:
            return getLauncher<doPruneNBL, doCalcEnergies, eelTypeEWALD_ANA>(flavorLJ);
        case eelTypeEWALD_ANA_TWIN:
            return getLauncher<doPruneNBL, doCalcEnergies, eelTypeEWALD_ANA_TWIN>(flavorLJ);
        case eelTypeNR: return getLauncher<doPruneNBL, doCalcEnergies, eelTypeNR>(flavorLJ);
        default: GMX_THROW(gmx::InternalError("Invalid EL flavor"));
    }
}

INbnxmSyclKernelLauncher* getNbnxmSyclKernelLauncher(bool          doPruneNBL,
                                                     bool          doCalcEnergies,
                                                     enum eelType  flavorEL,
                                                     enum evdwType flavorLJ)
{
    if (doPruneNBL)
    {
        if (doCalcEnergies)
        {
            return getLauncher<true, true>(flavorEL, flavorLJ);
        }
        else
        {
            return getLauncher<true, false>(flavorEL, flavorLJ);
        }
    }
    else
    {
        if (doCalcEnergies)
        {
            return getLauncher<false, true>(flavorEL, flavorLJ);
        }
        else
        {
            return getLauncher<false, false>(flavorEL, flavorLJ);
        }
    }
}
