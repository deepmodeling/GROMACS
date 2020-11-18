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

#include "nbnxm_sycl_kernel.h"

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/utility/template_mp.h"

#include "nbnxm_sycl_types.h"

/*! \brief cluster size = number of atoms per cluster. */
static constexpr int c_clSize = c_nbnxnGpuClusterSize;

namespace Nbnxm
{

template<enum ElecType elecType, enum VdwType vdwType>
struct EnergyFunctionProperties {
    static constexpr bool elecCutoff = (elecType == ElecType::Cut);
    static constexpr bool elecRF     = (elecType == ElecType::RF);
    static constexpr bool elecEwaldAna =
            (elecType == ElecType::EwaldAna || elecType == ElecType::EwaldAnaTwin);
    static constexpr bool elecEwaldTab =
            (elecType == ElecType::EwaldTab || elecType == ElecType::EwaldTabTwin);
    static constexpr bool elecEwaldTwin =
            (elecType == ElecType::EwaldAnaTwin || elecType == ElecType::EwaldTabTwin);
    static constexpr bool elecEwald = (elecEwaldAna || elecEwaldTab);
    static constexpr bool vdwComb = (vdwType == VdwType::CutCombLB || vdwType == VdwType::CutCombGeom);
    static constexpr bool vdwEwald = (vdwType == VdwType::EwaldLB || vdwType == VdwType::EwaldGeom);
};

template<enum VdwType vdwType>
constexpr bool ljComb = EnergyFunctionProperties<ElecType::Count, vdwType>().vdwComb;

template<enum ElecType elecType> // Yes, ElecType
constexpr bool vdwCutoffCheck = EnergyFunctionProperties<elecType, VdwType::Count>().elecEwaldTwin;

template<enum ElecType elecType>
constexpr bool elecRF = EnergyFunctionProperties<elecType, VdwType::Count>().elecRF;

template<enum ElecType elecType>
constexpr bool elecEwald = EnergyFunctionProperties<elecType, VdwType::Count>().elecEwald;

using cl::sycl::access::mode;

/*! \brief Main kernel for NBNXM.
 *
 */
template<bool doPruneNBL, bool doCalcEnergies, enum ElecType elecType, enum VdwType vdwType>
auto nbnxmKernel(cl::sycl::handler&                                        cgh,
                 DeviceAccessor<float4, mode::read>                        a_xq,
                 DeviceAccessor<float3, mode::read_write>                  a_f,
                 DeviceAccessor<float3, mode::read>                        a_shiftVec,
                 DeviceAccessor<float3, mode::read_write>                  a_fShift,
                 OptionalAccessor<float, mode::read_write, doCalcEnergies> a_elecEnergy,
                 OptionalAccessor<float, mode::read_write, doCalcEnergies> a_vdwEnergy,
                 DeviceAccessor<nbnxn_cj4_t, doPruneNBL ? mode::read_write : mode::read> a_plistCJ4,
                 DeviceAccessor<nbnxn_sci_t, mode::read>                                 a_plistSci,
                 DeviceAccessor<nbnxn_excl_t, mode::read>              a_plistExcl,
                 OptionalAccessor<int, mode::read, !ljComb<vdwType>>   a_atomTypes,
                 OptionalAccessor<float2, mode::read, ljComb<vdwType>> a_ljComb,
                 const float gmx_unused rCoulombSq,
                 const float gmx_unused rVdwSq,
                 const float gmx_unused twoKRf,
                 const float gmx_unused ewaldBeta,
                 const float gmx_unused rlistOuterSq,
                 const float gmx_unused ewaldShift)
{
    static constexpr EnergyFunctionProperties<elecType, vdwType> props;

    cgh.require(a_xq);
    cgh.require(a_f);
    cgh.require(a_shiftVec);
    cgh.require(a_fShift);
    cgh.require(a_f);
    if constexpr (doCalcEnergies)
    {
        cgh.require(a_elecEnergy);
        cgh.require(a_vdwEnergy);
    }
    cgh.require(a_plistSci);
    cgh.require(a_plistCJ4);
    cgh.require(a_plistExcl);
    if constexpr (props.vdwComb)
    {
        cgh.require(a_ljComb);
    }
    else
    {
        cgh.require(a_atomTypes);
    }

    /* Macro to control the calculation of exclusion forces in the kernel
     * We do that with Ewald (elec/vdw) and RF. Cut-off only has exclusion
     * energy terms.
     */
    constexpr bool gmx_unused doExclusionForces =
            (props.elecEwald || props.elecRF || props.vdwEwald || (props.elecCutoff && doCalcEnergies));

    return [=](cl::sycl::nd_item<3> gmx_unused itemIdx) {

    };
}

// SYCL 1.2.1 requires providing a unique type for a kernel. Should not be needed for SYCL2020.
template<bool doPruneNBL, bool doCalcEnergies, enum ElecType elecType, enum VdwType vdwType>
class NbnxmKernelName;

template<bool doPruneNBL, bool doCalcEnergies, enum ElecType elecType, enum VdwType vdwType, class... Args>
cl::sycl::event launchNbnxmKernel(const DeviceStream& deviceStream, const int numSci, Args&&... args)
{
    // Should not be needed for SYCL2020.
    using kernelNameType = NbnxmKernelName<doPruneNBL, doCalcEnergies, elecType, vdwType>;

    /* Kernel launch config:
     * - The thread block dimensions match the size of i-clusters, j-clusters,
     *   and j-cluster concurrency, in x, y, and z, respectively.
     * - The 1D block-grid contains as many blocks as super-clusters.
     */
    const int                   numBlocks = numSci;
    const cl::sycl::range<3>    blockSize{ c_clSize, c_clSize, 1 };
    const cl::sycl::range<3>    globalSize{ numBlocks * blockSize[0], blockSize[1], blockSize[2] };
    const cl::sycl::nd_range<3> range{ globalSize, blockSize };

    cl::sycl::queue q = deviceStream.stream();

    cl::sycl::event e = q.submit([&](cl::sycl::handler& cgh) {
        auto kernel = nbnxmKernel<doPruneNBL, doCalcEnergies, elecType, vdwType>(
                cgh, std::forward<Args>(args)...);
        cgh.parallel_for<kernelNameType>(range, kernel);
    });

    GMX_THROW(gmx::NotImplementedError("Not yet implemented for SYCL"));
}

template<class... Args>
cl::sycl::event chooseAndLaunchNbnxmKernel(bool          doPruneNBL,
                                           bool          doCalcEnergies,
                                           enum ElecType elecType,
                                           enum VdwType  vdwType,
                                           Args&&... args)
{
    return gmx::dispatchTemplatedFunction(
            [&](auto doPruneNBL_, auto doCalcEnergies_, auto elecType_, auto vdwType_) {
                return launchNbnxmKernel<doPruneNBL_, doCalcEnergies_, elecType_, vdwType_>(
                        std::forward<Args>(args)...);
            },
            doPruneNBL, doCalcEnergies, elecType, vdwType);
}

void launchNbnxmKernel(NbnxmGpu* nb, const gmx::StepWorkload& stepWork, const InteractionLocality iloc)
{
    sycl_atomdata_t*    adat         = nb->atdat;
    NBParamGpu*         nbp          = nb->nbparam;
    gpu_plist*          plist        = nb->plist[iloc];
    const bool          doPruneNBL   = (plist->haveFreshList && !nb->didPrune[iloc]);
    const DeviceStream& deviceStream = *nb->deviceStreams[iloc];

    cl::sycl::event e = chooseAndLaunchNbnxmKernel(
            doPruneNBL, stepWork.computeEnergy, nbp->elecType, nbp->vdwType, deviceStream,
            plist->nsci, adat->xq, adat->f, adat->shiftVec, adat->fShift, adat->eElec, adat->eLJ,
            plist->cj4, plist->sci, plist->excl, adat->atomTypes, adat->ljComb, nbp->rcoulomb_sq,
            nbp->rvdw_sq, nbp->two_k_rf, nbp->ewald_beta, nbp->rlistOuter_sq, nbp->sh_ewald);
}

} // namespace Nbnxm
