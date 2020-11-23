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
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/template_mp.h"

#include "nbnxm_sycl_types.h"

/*! \brief cluster size = number of atoms per cluster. */
static constexpr int c_clSize = c_nbnxnGpuClusterSize;

/*! \brief j-cluster size after split (4 in the current implementation). */
static constexpr int c_splitClSize = c_clSize / c_nbnxnGpuClusterpairSplit;

/*! \brief Stride in the force accumulation buffer */
static constexpr int c_fbufStride = c_clSize * c_clSize;

/*! \brief 1/sqrt(pi), same value as \c M_FLOAT_1_SQRTPI in other NB kernels */
static constexpr float c_OneOverSqrtPi = 0.564189583547756F;

// TODO: tune
#define NTHREAD_Z 1

namespace Nbnxm
{

//! \bried Set of boolean constants mimicking preprocessor macros
template<enum ElecType elecType, enum VdwType vdwType>
struct EnergyFunctionProperties {
    static constexpr bool elecCutoff = (elecType == ElecType::Cut); ///< EL_CUTOFF
    static constexpr bool elecRF     = (elecType == ElecType::RF);  ///< EL_RF
    static constexpr bool elecEwaldAna =
            (elecType == ElecType::EwaldAna || elecType == ElecType::EwaldAnaTwin);
    static constexpr bool elecEwaldTab =
            (elecType == ElecType::EwaldTab || elecType == ElecType::EwaldTabTwin);
    static constexpr bool elecEwaldTwin =
            (elecType == ElecType::EwaldAnaTwin || elecType == ElecType::EwaldTabTwin);
    static constexpr bool elecEwald   = (elecEwaldAna || elecEwaldTab); ///< EL_EWALD_ANY
    static constexpr bool vdwCombLB   = (vdwType == VdwType::CutCombLB);
    static constexpr bool vdwCombGeom = (vdwType == VdwType::CutCombGeom); ///< LJ_COMB_GEOM
    static constexpr bool vdwComb     = (vdwCombLB || vdwCombGeom);        ///< LJ_COMB
    static constexpr bool vdwEwald = (vdwType == VdwType::EwaldLB || vdwType == VdwType::EwaldGeom);
    static constexpr bool vdwFSwitch = (vdwType == VdwType::FSwitch); ///< LJ_FORCE_SWITCH
    static constexpr bool vdwPSwitch = (vdwType == VdwType::PSwitch); ///< LJ_POT_SWITCH
};

template<enum VdwType vdwType>
constexpr bool ljComb = EnergyFunctionProperties<ElecType::Count, vdwType>().vdwComb;

template<enum ElecType elecType> // Yes, ElecType
constexpr bool vdwCutoffCheck = EnergyFunctionProperties<elecType, VdwType::Count>().elecEwaldTwin;

template<enum ElecType elecType>
constexpr bool elecRF = EnergyFunctionProperties<elecType, VdwType::Count>().elecRF;

template<enum ElecType elecType>
constexpr bool elecEwald = EnergyFunctionProperties<elecType, VdwType::Count>().elecEwald;

// Same values that are used in CUDA kernels.
static constexpr float c_oneSixth   = 0.16666667f;
static constexpr float c_oneTwelfth = 0.08333333f;

using cl::sycl::access::fence_space;
using cl::sycl::access::mode;
using cl::sycl::access::target;

static inline void convert_sigma_epsilon_to_c6_c12(const float sigma, const float epsilon, float* c6, float* c12)
{
    const float sigma2 = sigma * sigma;
    const float sigma6 = sigma2 * sigma2 * sigma2;
    *c6                = epsilon * sigma6;
    *c12               = *c6 * sigma6;
}

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
                 OptionalAccessor<float, mode::read, !ljComb<vdwType>> a_nbfp,
                 OptionalAccessor<float, mode::read, doCalcEnergies>   a_nbfpComb,
                 const float gmx_unused rCoulombSq,
                 const float gmx_unused rVdwSq,
                 const float gmx_unused twoKRf,
                 const float gmx_unused ewaldBeta,
                 const float gmx_unused rlistOuterSq,
                 const float gmx_unused ewaldShift,
                 const float            epsFac,
                 const float            ewaldCoeffLJ,
                 const int              numTypes,
                 const float            c_rf,
                 const float            repulsionShiftPot)
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

    // shmem buffer for i x+q pre-loading
    cl::sycl::accessor<float4, 2, mode::read_write, target::local> xqib(
            cl::sycl::range<2>(c_nbnxnGpuNumClusterPerSupercluster, c_clSize), cgh);
    // shmem buffer for cj, for each warp separately
    // the cjs buffer's use expects a base pointer offset for pairs of warps in the j-concurrent execution
    cl::sycl::accessor<int, 1, mode::read_write, target::local> cjs(
            cl::sycl::range<1>(NTHREAD_Z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize), cgh);

    auto atib = [&cgh]() {
        if constexpr (!props.vdwComb)
        {
            return cl::sycl::accessor<int, 2, mode::read_write, target::local>(
                    cl::sycl::range<2>(c_nbnxnGpuNumClusterPerSupercluster, c_clSize), cgh);
        }
        else
        {
            return nullptr;
        }
    }();

    auto ljcpib = [&cgh]() {
        if constexpr (props.vdwComb)
        {
            return cl::sycl::accessor<float2, 2, mode::read_write, target::local>(
                    cl::sycl::range<2>(c_nbnxnGpuNumClusterPerSupercluster, c_clSize), cgh);
        }
        else
        {
            return nullptr;
        }
    }();

    /* Macro to control the calculation of exclusion forces in the kernel
     * We do that with Ewald (elec/vdw) and RF. Cut-off only has exclusion
     * energy terms.
     */
    constexpr bool doExclusionForces =
            (props.elecEwald || props.elecRF || props.vdwEwald || (props.elecCutoff && doCalcEnergies));

    // i-cluster interaction mask for a super-cluster with all c_nbnxnGpuNumClusterPerSupercluster=8 bits set
    constexpr unsigned superClInteractionMask = ((1U << c_nbnxnGpuNumClusterPerSupercluster) - 1U);

    return [=](cl::sycl::nd_item<3> itemIdx) {
        /* thread/block/warp id-s */
        const unsigned        tidxi = itemIdx.get_local_id(0);
        const unsigned        tidxj = itemIdx.get_local_id(1);
        const cl::sycl::id<2> tidxji(itemIdx.get_local_id(1), itemIdx.get_local_id(0));
#if NTHREAD_Z == 1
        const unsigned tidxz = 0;
        const unsigned tidx  = itemIdx.get_local_linear_id();
#else
        const unsigned tidxz = itemIdx.get_local_id(2);
        const unsigned tidx  = tidxj * itemIdx.get_local_range(0) + tidxi;
#endif
        const unsigned bidx = itemIdx.get_group(0);
        // Relies on sub_group from SYCL2020 provisional spec / Intel implementation
        const sycl::ONEAPI::sub_group sg   = itemIdx.get_sub_group();
        const unsigned                widx = sg.get_local_id(); // index in sub-group (warp)
        float3 fci_buf[c_nbnxnGpuNumClusterPerSupercluster] = { { 0.0F, 0.0F, 0.0F } }; // i force buffer

        const nbnxn_sci_t nb_sci     = a_plistSci[bidx];
        const int         sci        = nb_sci.sci;
        const int         cij4_start = nb_sci.cj4_ind_start;
        const int         cij4_end   = nb_sci.cj4_ind_end;

        float4 xqbuf;

        if (tidxz == 0)
        {
            /* Pre-load i-atom x and q into shared memory */
            const int ci = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxj;
            const int ai = ci * c_clSize + tidxi;

            float3 shift = a_shiftVec[nb_sci.shift];
            xqbuf        = a_xq[ai];
            xqbuf += float4(shift[0], shift[1], shift[2], 0.0F);
            xqbuf[3] *= epsFac;
            xqib[tidxji] = xqbuf;

            if constexpr (!props.vdwComb)
            {
                // Pre-load the i-atom types into shared memory
                atib[tidxji] = a_atomTypes[ai];
            }
            else
            {
                // Pre-load the LJ combination parameters into shared memory
                ljcpib[tidxji] = a_ljComb[ai];
            }
        }
        itemIdx.barrier(fence_space::local_space);

        float lje_coeff2, lje_coeff6_6; // Only needed if (props.vdwEwald)
        if constexpr (props.vdwEwald)
        {
            lje_coeff2   = ewaldCoeffLJ * ewaldCoeffLJ;
            lje_coeff6_6 = lje_coeff2 * lje_coeff2 * lje_coeff2 * c_oneSixth;
        }

        float E_lj, E_el; // Only needed if (doCalcEnergies)
        if constexpr (doCalcEnergies)
        {
            E_lj = E_el = 0.0F;
            if constexpr (doExclusionForces)
            {
                if (nb_sci.shift == CENTRAL
                    && a_plistCJ4[cij4_start].cj[0] == sci * c_nbnxnGpuNumClusterPerSupercluster)
                {
                    // we have the diagonal: add the charge and LJ self interaction energy term
                    for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
                    {
                        // TODO: Are there other options?
                        if constexpr (props.elecEwald || props.elecRF || props.elecCutoff)
                        {
                            const float qi = xqib[cl::sycl::id<2>(i, tidxi)].w();
                            E_el += qi * qi;
                        }
                        if constexpr (props.vdwEwald)
                        {
                            E_lj += a_nbfp[a_atomTypes[(sci * c_nbnxnGpuNumClusterPerSupercluster + i) * c_clSize + tidxi]
                                           * (numTypes + 1) * 2];
                        }
                    }
                    /* divide the self term(s) equally over the j-threads, then multiply with the coefficients. */
                    if constexpr (props.vdwEwald)
                    {
                        E_lj /= c_clSize * NTHREAD_Z;
                        E_lj *= 0.5f * c_oneSixth * lje_coeff6_6;
                    }
                    if constexpr (props.elecEwald || props.elecRF || props.elecCutoff)
                    {
                        // Correct for epsfac^2 due to adding qi^2 */
                        E_el /= epsFac * c_clSize * NTHREAD_Z;
                        if constexpr (props.elecRF || props.elecCutoff)
                        {
                            E_el *= -0.5f * c_rf;
                        }
                        else
                        {
                            E_el *= -ewaldBeta * c_OneOverSqrtPi; /* last factor 1/sqrt(pi) */
                        }
                    }
                }
            }
        }

        const bool nonSelfInteraction =
                !(nb_sci.shift == CENTRAL & tidxj <= tidxi); // Only needed if (doExclusionForces)

        /* loop over the j clusters = seen by any of the atoms in the current super-cluster;
         * The loop stride NTHREAD_Z ensures that consecutive warps-pairs are assigned
         * consecutive j4's entries. */
        for (int j4 = cij4_start + tidxz; j4 < cij4_end; j4 += NTHREAD_Z)
        {
            const int wexcl_idx = a_plistCJ4[j4].imei[widx].excl_ind;
            int       imask     = a_plistCJ4[j4].imei[widx].imask;
            const int wexcl     = a_plistExcl[wexcl_idx].pair[sg.get_local_linear_id()];
            if (doPruneNBL || imask)
            {
                /* Pre-load cj into shared memory on both warps separately */
                if ((tidxj == 0 || tidxj == 4) && (tidxi < c_nbnxnGpuJgroupSize))
                {
                    cjs[tidxi + tidxj * c_nbnxnGpuJgroupSize / c_splitClSize] = a_plistCJ4[j4].cj[tidxi];
                }
                sg.barrier();

                for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
                {
                    if (imask & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster)))
                    {
                        int mask_ji = (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster));
                        int cj      = cjs[jm + (tidxj & 4) * c_nbnxnGpuJgroupSize / c_splitClSize];
                        int aj      = cj * c_clSize + tidxj;

                        /* load j atom data */
                        xqbuf = a_xq[aj];
                        const float3 xj(xqbuf[0], xqbuf[1], xqbuf[2]);
                        const float  qj_f = xqbuf[3];
                        int          typej;  // Only needed if (!props.vdwComb)
                        float2       ljcp_j; // Only needed if (props.vdwComb)
                        if constexpr (!props.vdwComb)
                        {
                            typej = a_atomTypes[aj];
                        }
                        else
                        {
                            ljcp_j = a_ljComb[aj];
                        }

                        float3 fcj_buf(0.0F, 0.0F, 0.0F);

                        for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
                        {
                            if (imask & mask_ji)
                            {
                                int ci = sci * c_nbnxnGpuNumClusterPerSupercluster + i; /* i cluster index */

                                /* all threads load an atom from i cluster ci into shmem! */
                                xqbuf = xqib[cl::sycl::id<2>(i, tidxi)];
                                float3 xi(xqbuf[0], xqbuf[1], xqbuf[2]);

                                /* distance between i and j atoms */
                                float3 rv = xi - xj;
                                float  r2 = norm2(rv);
                                if constexpr (doPruneNBL)
                                {
                                    /* If _none_ of the atoms pairs are in cutoff range,
                                     * the bit corresponding to the current
                                     * cluster-pair in imask gets set to 0. */
                                    // sycl::group_any_of in SYCL2020 provisional
                                    if (!sycl::ONEAPI::any_of(sg, r2 < rlistOuterSq))
                                    {
                                        imask &= ~mask_ji;
                                    }
                                }
                                const float int_bit = (wexcl & mask_ji) ? 1.0f : 0.0f;

                                // cutoff & exclusion check

                                const bool notExcluded = doExclusionForces
                                                                 ? (nonSelfInteraction | (ci != cj))
                                                                 : (wexcl & mask_ji);
                                if ((r2 < rCoulombSq) && notExcluded)
                                {
                                    float qi = xqbuf[3];
                                    int   typei; // Only needed if (!props.vdwComb)
                                    float c6, c12, sigma, epsilon;
                                    if constexpr (!props.vdwComb)
                                    {
                                        /* LJ 6*C6 and 12*C12 */
                                        typei         = atib[cl::sycl::id<2>(i, tidxi)];
                                        const int idx = (numTypes * typei + typej) * 2;
                                        c6  = a_nbfp[idx]; // TODO: Make a_nbfm into float2
                                        c12 = a_nbfp[idx + 1];
                                    }
                                    else
                                    {
                                        float2 ljcp_i = ljcpib[cl::sycl::id<2>(i, tidxi)];
                                        if constexpr (props.vdwCombGeom)
                                        {
                                            c6  = ljcp_i[0] * ljcp_j[0];
                                            c12 = ljcp_i[1] * ljcp_j[1];
                                        }
                                        else
                                        {
                                            // LJ 2^(1/6)*sigma and 12*epsilon
                                            sigma   = ljcp_i[0] + ljcp_j[0];
                                            epsilon = ljcp_i[1] * ljcp_j[1];
                                            if constexpr (doCalcEnergies || props.vdwFSwitch || props.vdwPSwitch)
                                            {
                                                convert_sigma_epsilon_to_c6_c12(sigma, epsilon, &c6, &c12);
                                            }
                                        } // props.vdwCombGeom
                                    }     // !props.vdwComb
                                    // Ensure distance do not become so small that r^-12 overflows
                                    r2                 = std::max(r2, c_nbnxnMinDistanceSquared);
                                    const float inv_r  = 1.0F / std::sqrt(r2);
                                    const float inv_r2 = inv_r * inv_r;
                                    float       inv_r6, F_invr, E_lj_p;
                                    if constexpr (!props.vdwCombLB || doCalcEnergies)
                                    {
                                        inv_r6 = inv_r2 * inv_r2 * inv_r2;
                                        if constexpr (doExclusionForces)
                                        {
                                            // SYCL-TODO: Check if true for SYCL
                                            /* We could mask inv_r2, but with Ewald masking both
                                             * inv_r6 and F_invr is faster */
                                            inv_r6 *= int_bit;
                                        }
                                        F_invr = inv_r6 * (c12 * inv_r6 - c6) * inv_r2;
                                        if constexpr (doCalcEnergies || props.vdwPSwitch)
                                        {
                                            E_lj_p = int_bit
                                                     * (c12 * (inv_r6 * inv_r6 + repulsionShiftPot) * c_oneTwelfth
                                                        - c6 * (inv_r6 + repulsionShiftPot) * c_oneSixth);
                                        }
                                    }
                                    else
                                    {
                                        float sig_r  = sigma * inv_r;
                                        float sig_r2 = sig_r * sig_r;
                                        float sig_r6 = sig_r2 * sig_r2 * sig_r2;
                                        if constexpr (doExclusionForces)
                                        {
                                            sig_r6 *= int_bit;
                                        }
                                        F_invr = epsilon * sig_r6 * (sig_r6 - 1.0f) * inv_r2;
                                    } // (!props.vdwCombLB || doCalcEnergies)

                                    // TODO: Continue from here
                                } // (r2 < rCoulombSq) && notExcluded
                            }     // (imask & mask_ji)
                        }         // for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
                    } // (imask & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster)))
                } // for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
            }     // (doPruneNBL || imask)
        }         // for (int j4 = cij4_start + tidxz; j4 < cij4_end; j4 += NTHREAD_Z)
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
    const cl::sycl::range<3>    blockSize{ c_clSize, c_clSize, NTHREAD_Z };
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
            plist->cj4, plist->sci, plist->excl, adat->atomTypes, adat->ljComb, nbp->nbfp,
            nbp->nbfp_comb, nbp->rcoulomb_sq, nbp->rvdw_sq, nbp->two_k_rf, nbp->ewald_beta,
            nbp->rlistOuter_sq, nbp->sh_ewald, nbp->epsfac, nbp->ewaldcoeff_lj, adat->numTypes,
            nbp->c_rf, nbp->repulsion_shift.cpot);
}

} // namespace Nbnxm
