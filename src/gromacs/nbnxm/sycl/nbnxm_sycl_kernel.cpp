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
#include "gromacs/nbnxm/sycl/nbnxm_sycl_kernel_utils.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/template_mp.h"

#include "nbnxm_sycl_types.h"

// TODO: tune
#define NTHREAD_Z 1

namespace Nbnxm
{

//! \brief Set of boolean constants mimicking preprocessor macros
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
    static constexpr bool elecEwald        = (elecEwaldAna || elecEwaldTab); ///< EL_EWALD_ANY
    static constexpr bool vdwCombLB        = (vdwType == VdwType::CutCombLB);
    static constexpr bool vdwCombGeom      = (vdwType == VdwType::CutCombGeom); ///< LJ_COMB_GEOM
    static constexpr bool vdwComb          = (vdwCombLB || vdwCombGeom);        ///< LJ_COMB
    static constexpr bool vdwEwaldCombGeom = (vdwType == VdwType::EwaldGeom); ///< LJ_EWALD_COMB_GEOM
    static constexpr bool vdwEwaldCombLB   = (vdwType == VdwType::EwaldLB);   ///< LJ_EWALD_COMB_LB
    static constexpr bool vdwEwald         = (vdwEwaldCombGeom || vdwEwaldCombLB); ///< LJ_EWALD
    static constexpr bool vdwFSwitch       = (vdwType == VdwType::FSwitch); ///< LJ_FORCE_SWITCH
    static constexpr bool vdwPSwitch       = (vdwType == VdwType::PSwitch); ///< LJ_POT_SWITCH
};

template<enum VdwType vdwType>
constexpr bool ljComb = EnergyFunctionProperties<ElecType::Count, vdwType>().vdwComb;

template<enum ElecType elecType> // Yes, ElecType
constexpr bool vdwCutoffCheck = EnergyFunctionProperties<elecType, VdwType::Count>().elecEwaldTwin;

template<enum ElecType elecType>
constexpr bool elecRF = EnergyFunctionProperties<elecType, VdwType::Count>().elecRF;

template<enum ElecType elecType>
constexpr bool elecEwald = EnergyFunctionProperties<elecType, VdwType::Count>().elecEwald;

template<enum ElecType elecType>
constexpr bool elecEwaldTab = EnergyFunctionProperties<elecType, VdwType::Count>().elecEwaldTab;

template<enum VdwType vdwType>
constexpr bool ljEwald = EnergyFunctionProperties<ElecType::Count, vdwType>().vdwEwald;

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

// SYCL-TODO: Merge with calculate_force_switch_F_E by the means of template
static inline void calculate_force_switch_F(const shift_consts_t dispersionShift,
                                            const shift_consts_t repulsionShift,
                                            const float          rVdwSwitch,
                                            const float          c6,
                                            const float          c12,
                                            const float          inv_r,
                                            const float          r2,
                                            float*               F_invr)
{
    /* force switch constants */
    const float dispShiftV2 = dispersionShift.c2;
    const float dispShiftV3 = dispersionShift.c3;
    const float repuShiftV2 = repulsionShift.c2;
    const float repuShiftV3 = repulsionShift.c3;

    const float r       = r2 * inv_r;
    float       rSwitch = r - rVdwSwitch;
    rSwitch             = rSwitch >= 0.0F ? rSwitch : 0.0F;

    *F_invr += -c6 * (dispShiftV2 + dispShiftV3 * rSwitch) * rSwitch * rSwitch * inv_r
               + c12 * (repuShiftV2 + repuShiftV3 * rSwitch) * rSwitch * rSwitch * inv_r;
}

static inline void calculate_force_switch_F_E(const shift_consts_t dispersionShift,
                                              const shift_consts_t repulsionShift,
                                              const float          rVdwSwitch,
                                              const float          c6,
                                              const float          c12,
                                              const float          inv_r,
                                              const float          r2,
                                              float*               F_invr,
                                              float*               E_lj)
{
    /* force switch constants */
    const float dispShiftV2 = dispersionShift.c2;
    const float dispShiftV3 = dispersionShift.c3;
    const float repuShiftV2 = repulsionShift.c2;
    const float repuShiftV3 = repulsionShift.c3;

    const float dispShiftF2 = dispShiftV2 / 3;
    const float dispShiftF3 = dispShiftV3 / 4;
    const float repuShiftF2 = repuShiftV2 / 3;
    const float repuShiftF3 = repuShiftV3 / 4;

    const float r       = r2 * inv_r;
    float       rSwitch = r - rVdwSwitch;
    rSwitch             = rSwitch >= 0.0F ? rSwitch : 0.0F;

    *F_invr += -c6 * (dispShiftV2 + dispShiftV3 * rSwitch) * rSwitch * rSwitch * inv_r
               + c12 * (repuShiftV2 + repuShiftV3 * rSwitch) * rSwitch * rSwitch * inv_r;
    *E_lj += c6 * (dispShiftF2 + dispShiftF3 * rSwitch) * rSwitch * rSwitch * rSwitch
             - c12 * (repuShiftF2 + repuShiftF3 * rSwitch) * rSwitch * rSwitch * rSwitch;
}

/*! \brief Fetch C6 grid contribution coefficients and return the product of these.
 */
static inline float calculate_lj_ewald_c6grid(const DeviceAccessor<float, mode::read> a_nbfpComb,
                                              const int                               typei,
                                              const int                               typej)
{
    // SYCL-TODO: Pass by const reference?
    return a_nbfpComb[2 * typei] * a_nbfpComb[2 * typej];
}


/*! Calculate LJ-PME grid force contribution with
 *  geometric combination rule.
 */
static inline void calculate_lj_ewald_comb_geom_F(const DeviceAccessor<float, mode::read> a_nbfpComb,
                                                  const int                               typei,
                                                  const int                               typej,
                                                  const float                             r2,
                                                  const float                             inv_r2,
                                                  const float lje_coeff2,
                                                  const float lje_coeff6_6,
                                                  float*      F_invr)
{
    // SYCL-TODO: Merge with calculate_lj_ewald_comb_geom_F_E by templating on doCalcEnergies
    const float c6grid = calculate_lj_ewald_c6grid(std::move(a_nbfpComb), typei, typej);

    /* Recalculate inv_r6 without exclusion mask */
    const float inv_r6_nm = inv_r2 * inv_r2 * inv_r2;
    const float cr2       = lje_coeff2 * r2;
    const float expmcr2   = expf(-cr2);
    const float poly      = 1.0F + cr2 + 0.5F * cr2 * cr2;

    /* Subtract the grid force from the total LJ force */
    *F_invr += c6grid * (inv_r6_nm - expmcr2 * (inv_r6_nm * poly + lje_coeff6_6)) * inv_r2;
}


/*! Calculate LJ-PME grid force + energy contribution with
 *  geometric combination rule.
 */
static inline void calculate_lj_ewald_comb_geom_F_E(const DeviceAccessor<float, mode::read> a_nbfpComb,
                                                    const float sh_lj_ewald,
                                                    const int   typei,
                                                    const int   typej,
                                                    const float r2,
                                                    const float inv_r2,
                                                    const float lje_coeff2,
                                                    const float lje_coeff6_6,
                                                    const float int_bit,
                                                    float*      F_invr,
                                                    float*      E_lj)
{
    const float c6grid = calculate_lj_ewald_c6grid(std::move(a_nbfpComb), typei, typej);

    /* Recalculate inv_r6 without exclusion mask */
    const float inv_r6_nm = inv_r2 * inv_r2 * inv_r2;
    const float cr2       = lje_coeff2 * r2;
    const float expmcr2   = expf(-cr2);
    const float poly      = 1.0F + cr2 + 0.5F * cr2 * cr2;

    /* Subtract the grid force from the total LJ force */
    *F_invr += c6grid * (inv_r6_nm - expmcr2 * (inv_r6_nm * poly + lje_coeff6_6)) * inv_r2;

    /* Shift should be applied only to real LJ pairs */
    const float sh_mask = sh_lj_ewald * int_bit;
    *E_lj += c_oneSixth * c6grid * (inv_r6_nm * (1.0F - expmcr2 * poly) + sh_mask);
}

/*! Calculate LJ-PME grid force + energy contribution (if E_lj != nullptr) with
 *  Lorentz-Berthelot combination rule.
 *  We use a single F+E kernel with conditional because the performance impact
 *  of this is pretty small and LB on the CPU is anyway very slow.
 */
static inline void calculate_lj_ewald_comb_LB_F_E(const DeviceAccessor<float, mode::read> a_nbfpComb,
                                                  const float sh_lj_ewald,
                                                  const int   typei,
                                                  const int   typej,
                                                  const float r2,
                                                  const float inv_r2,
                                                  const float lje_coeff2,
                                                  const float lje_coeff6_6,
                                                  const float int_bit,
                                                  float*      F_invr,
                                                  float*      E_lj)
{
    /* sigma and epsilon are scaled to give 6*C6 */
    const float c6_i  = a_nbfpComb[2 * typei];
    const float c12_i = a_nbfpComb[2 * typei + 1];
    const float c6_j  = a_nbfpComb[2 * typej];
    const float c12_j = a_nbfpComb[2 * typej + 1];

    const float sigma   = c6_i + c6_j;
    const float epsilon = c12_i * c12_j;

    const float sigma2 = sigma * sigma;
    const float c6grid = epsilon * sigma2 * sigma2 * sigma2;

    /* Recalculate inv_r6 without exclusion mask */
    const float inv_r6_nm = inv_r2 * inv_r2 * inv_r2;
    const float cr2       = lje_coeff2 * r2;
    const float expmcr2   = expf(-cr2);
    const float poly      = 1.0F + cr2 + 0.5F * cr2 * cr2;

    /* Subtract the grid force from the total LJ force */
    *F_invr += c6grid * (inv_r6_nm - expmcr2 * (inv_r6_nm * poly + lje_coeff6_6)) * inv_r2;

    if (E_lj != nullptr)
    {
        /* Shift should be applied only to real LJ pairs */
        const float sh_mask = sh_lj_ewald * int_bit;
        *E_lj += c_oneSixth * c6grid * (inv_r6_nm * (1.0F - expmcr2 * poly) + sh_mask);
    }
}

/*! Apply potential switch, force-only version. */
static inline void calculate_potential_switch_F(const switch_consts_t vdw_switch,
                                                const float           rVdwSwitch,
                                                const float           inv_r,
                                                const float           r2,
                                                float*                F_invr,
                                                float*                E_lj)
{
    /* potential switch constants */
    const float switch_V3 = vdw_switch.c3;
    const float switch_V4 = vdw_switch.c4;
    const float switch_V5 = vdw_switch.c5;
    const float switch_F2 = 3 * vdw_switch.c3;
    const float switch_F3 = 4 * vdw_switch.c4;
    const float switch_F4 = 5 * vdw_switch.c5;

    const float r       = r2 * inv_r;
    const float rSwitch = r - rVdwSwitch;

    // SYCL-TODO: The comment below is true for CUDA only
    // Unlike in the F+E kernel, conditional is faster here
    if (rSwitch > 0.0F)
    {
        const float sw =
                1.0F + (switch_V3 + (switch_V4 + switch_V5 * rSwitch) * rSwitch) * rSwitch * rSwitch * rSwitch;
        const float dsw = (switch_F2 + (switch_F3 + switch_F4 * rSwitch) * rSwitch) * rSwitch * rSwitch;

        *F_invr = (*F_invr) * sw - inv_r * (*E_lj) * dsw;
    }
}

/*! Apply potential switch, force + energy version. */
static inline void calculate_potential_switch_F_E(const switch_consts_t vdw_switch,
                                                  const float           rVdwSwitch,
                                                  float                 inv_r,
                                                  float                 r2,
                                                  float*                F_invr,
                                                  float*                E_lj)
{
    /* potential switch constants */
    const float switch_V3 = vdw_switch.c3;
    const float switch_V4 = vdw_switch.c4;
    const float switch_V5 = vdw_switch.c5;
    const float switch_F2 = 3 * vdw_switch.c3;
    const float switch_F3 = 4 * vdw_switch.c4;
    const float switch_F4 = 5 * vdw_switch.c5;

    const float r       = r2 * inv_r;
    float       rSwitch = r - rVdwSwitch;
    rSwitch             = rSwitch >= 0.0F ? rSwitch : 0.0F;

    const float sw =
            1.0F + (switch_V3 + (switch_V4 + switch_V5 * rSwitch) * rSwitch) * rSwitch * rSwitch * rSwitch;
    const float dsw = (switch_F2 + (switch_F3 + switch_F4 * rSwitch) * rSwitch) * rSwitch * rSwitch;

    *F_invr = (*F_invr) * sw - inv_r * (*E_lj) * dsw;
    *E_lj *= sw;
}


/*! Calculate analytical Ewald correction term. */
static inline float pmecorrF(const float z2)
{
    constexpr float FN6 = -1.7357322914161492954e-8f;
    constexpr float FN5 = 1.4703624142580877519e-6f;
    constexpr float FN4 = -0.000053401640219807709149f;
    constexpr float FN3 = 0.0010054721316683106153f;
    constexpr float FN2 = -0.019278317264888380590f;
    constexpr float FN1 = 0.069670166153766424023f;
    constexpr float FN0 = -0.75225204789749321333f;

    constexpr float FD4 = 0.0011193462567257629232f;
    constexpr float FD3 = 0.014866955030185295499f;
    constexpr float FD2 = 0.11583842382862377919f;
    constexpr float FD1 = 0.50736591960530292870f;
    constexpr float FD0 = 1.0f;

    float z4;
    float polyFN0, polyFN1, polyFD0, polyFD1;

    z4 = z2 * z2;

    polyFD0 = FD4 * z4 + FD2;
    polyFD1 = FD3 * z4 + FD1;
    polyFD0 = polyFD0 * z4 + FD0;
    polyFD0 = polyFD1 * z2 + polyFD0;

    polyFD0 = 1.0f / polyFD0;

    polyFN0 = FN6 * z4 + FN4;
    polyFN1 = FN5 * z4 + FN3;
    polyFN0 = polyFN0 * z4 + FN2;
    polyFN1 = polyFN1 * z4 + FN1;
    polyFN0 = polyFN0 * z4 + FN0;
    polyFN0 = polyFN1 * z2 + polyFN0;

    return polyFN0 * polyFD0;
}

/*! Linear interpolation using exactly two FMA operations.
 *
 *  Implements numeric equivalent of: (1-t)*d0 + t*d1.
 */
template<typename T>
static inline T lerp(T d0, T d1, T t)
{
    return fma(t, d1, fma(-t, d0, d0));
}

/*! Interpolate Ewald coulomb force correction using the F*r table.
 */
static inline float interpolate_coulomb_force_r(const DeviceAccessor<float, mode::read> a_coulombTab,
                                                const float coulombTabScale,
                                                const float r)
{
    const float normalized = coulombTabScale * r;
    const int   index      = (int)normalized;
    const float fraction   = normalized - index;

    const float left  = a_coulombTab[index];
    const float right = a_coulombTab[index + 1];

    return lerp(left, right, fraction);
}

/*! Final j-force reduction; this implementation only with power of two
 *  array sizes.
 */
static inline void reduce_force_j(cl::sycl::accessor<float, 1, mode::read_write, target::local> shmemBuf,
                                  const float3                                                  f,
                                  DeviceAccessor<float, mode::read_write> fout,
                                  const cl::sycl::nd_item<1>              itemIdx,
                                  const int                               tidxi,
                                  const int                               tidxj,
                                  const int                               aidx)
{
    // SYCL-TODO: Check for NTHREAD_Z > 1
    static constexpr int bufStride = c_clSize * c_clSize;
    const int            tid       = tidxi + tidxj * c_clSize; // itemIdx.get_local_linear_id();
    shmemBuf[tid]                  = f[0];
    shmemBuf[tid + bufStride]      = f[1];
    shmemBuf[tid + 2 * bufStride]  = f[2];
    // SYCL-TODO: Synchronizing only sub-group should be enough?
    itemIdx.barrier(fence_space::local_space);
    if (tidxi < 3)
    {
        float acc = 0.0F;
        for (int j = tidxj * c_clSize; j < (tidxj + 1) * c_clSize; j++)
        {
            acc += shmemBuf[bufStride * tidxi + j];
        }

        atomic_fetch_add(fout, 3 * aidx + tidxi, acc);
    }
    itemIdx.barrier(fence_space::local_space);
}

static constexpr int log2i(const int v)
{
    if (v == 1)
    {
        return 0;
    }
    else
    {
        assert(v % 2 == 0);
        return log2i(v / 2) + 1;
    }
}

/*! Final i-force reduction; this implementation works only with power of two
 *  array sizes.
 */
static inline void reduce_force_i_and_shift(cl::sycl::accessor<float, 1, mode::read_write, target::local> shmemBuf,
                                            const float3 fci_buf[c_nbnxnGpuNumClusterPerSupercluster],
                                            DeviceAccessor<float, mode::read_write> fout,
                                            const bool                              bCalcFshift,
                                            const cl::sycl::nd_item<1>              itemIdx,
                                            const int                               tidxi,
                                            const int                               tidxj,
                                            const int                               sci,
                                            const int                               shift,
                                            DeviceAccessor<float, mode::read_write> fshift)
{
    // SYCL-TODO: Check for NTHREAD_Z > 1
    static constexpr int bufStride  = c_clSize * c_clSize;
    static constexpr int clSizeLog2 = log2i(c_clSize);
    float                fshift_buf = 0;
    for (int ci_offset = 0; ci_offset < c_nbnxnGpuNumClusterPerSupercluster; ci_offset++)
    {
        int aidx = (sci * c_nbnxnGpuNumClusterPerSupercluster + ci_offset) * c_clSize + tidxi;
        int tidx = tidxi + tidxj * c_clSize;
        /* store i forces in shmem */
        shmemBuf[tidx]                 = fci_buf[ci_offset][0];
        shmemBuf[bufStride + tidx]     = fci_buf[ci_offset][1];
        shmemBuf[2 * bufStride + tidx] = fci_buf[ci_offset][2];
        itemIdx.barrier(fence_space::local_space);

        /* Reduce the initial CL_SIZE values for each i atom to half
         * every step by using CL_SIZE * i threads.
         * Can't just use i as loop variable because than nvcc refuses to unroll.
         */
        int i = c_clSize / 2;
        for (int j = clSizeLog2 - 1; j > 0; j--)
        {
            if (tidxj < i)
            {
                shmemBuf[tidxj * c_clSize + tidxi] += shmemBuf[(tidxj + i) * c_clSize + tidxi];
                shmemBuf[bufStride + tidxj * c_clSize + tidxi] +=
                        shmemBuf[bufStride + (tidxj + i) * c_clSize + tidxi];
                shmemBuf[2 * bufStride + tidxj * c_clSize + tidxi] +=
                        shmemBuf[2 * bufStride + (tidxj + i) * c_clSize + tidxi];
            }
            i >>= 1;
        }
        /* needed because
         * a) for c_clSize<8: id 2 (doing z in next block) is in 2nd warp
         * b) for all c_clSize a barrier is needed before f_buf is reused by next reduce_force_i call
         */
        itemIdx.barrier(fence_space::local_space);

        /* i == 1, last reduction step, writing to global mem */
        /* Split the reduction between the first 3 line threads
           Threads with line id 0 will do the reduction for (float3).x components
           Threads with line id 1 will do the reduction for (float3).y components
           Threads with line id 2 will do the reduction for (float3).z components. */
        if (tidxj < 3)
        {
            float f = shmemBuf[tidxj * bufStride + tidxi]
                      + shmemBuf[tidxj * bufStride + i * c_clSize + tidxi];

            atomic_fetch_add(fout, 3 * aidx + tidxj, f);

            if (bCalcFshift)
            {
                fshift_buf += f;
            }
        }
    }
    itemIdx.barrier(fence_space::local_space);
    /* add up local shift forces into global mem */
    if (bCalcFshift)
    {
        /* Only threads with tidxj < 3 will update fshift.
           The threads performing the update, must be the same as the threads
           storing the reduction result above.
         */
        if (tidxj < 3)
        {
            atomic_fetch_add(fshift, 3 * shift + tidxj, fshift_buf);
        }
    }
}

/*! \brief Main kernel for NBNXM.
 *
 */
template<bool doPruneNBL, bool doCalcEnergies, enum ElecType elecType, enum VdwType vdwType>
auto nbnxmKernel(cl::sycl::handler&                                        cgh,
                 DeviceAccessor<float4, mode::read>                        a_xq,
                 DeviceAccessor<float, mode::read_write>                   a_f,
                 DeviceAccessor<float3, mode::read>                        a_shiftVec,
                 DeviceAccessor<float, mode::read_write>                   a_fShift,
                 OptionalAccessor<float, mode::read_write, doCalcEnergies> a_elecEnergy,
                 OptionalAccessor<float, mode::read_write, doCalcEnergies> a_vdwEnergy,
                 DeviceAccessor<nbnxn_cj4_t, doPruneNBL ? mode::read_write : mode::read> a_plistCJ4,
                 DeviceAccessor<nbnxn_sci_t, mode::read>                                 a_plistSci,
                 DeviceAccessor<nbnxn_excl_t, mode::read>                    a_plistExcl,
                 OptionalAccessor<int, mode::read, !ljComb<vdwType>>         a_atomTypes,
                 OptionalAccessor<float2, mode::read, ljComb<vdwType>>       a_ljComb,
                 OptionalAccessor<float, mode::read, !ljComb<vdwType>>       a_nbfp,
                 OptionalAccessor<float, mode::read, ljEwald<vdwType>>       a_nbfpComb,
                 OptionalAccessor<float, mode::read, elecEwaldTab<elecType>> a_coulombType,
                 const float                                                 rCoulombSq,
                 const float                                                 rVdwSq,
                 const float                                                 twoKRf,
                 const float                                                 ewaldBeta,
                 const float                                                 rlistOuterSq,
                 const float                                                 ewaldShift,
                 const float                                                 epsFac,
                 const float                                                 ewaldCoeffLJ,
                 const int                                                   numTypes,
                 const float                                                 c_rf,
                 const shift_consts_t                                        dispersion_shift,
                 const shift_consts_t                                        repulsion_shift,
                 const switch_consts_t                                       vdw_switch,
                 const float                                                 rVdwSwitch,
                 const float                                                 sh_lj_ewald,
                 const float                                                 coulombTabScale,
                 const bool                                                  calcShift)
{
    static constexpr EnergyFunctionProperties<elecType, vdwType> props;

    cgh.require(a_xq);
    cgh.require(a_f);
    cgh.require(a_shiftVec);
    cgh.require(a_fShift);
    if constexpr (doCalcEnergies)
    {
        cgh.require(a_elecEnergy);
        cgh.require(a_vdwEnergy);
    }
    cgh.require(a_plistCJ4);
    cgh.require(a_plistSci);
    cgh.require(a_plistExcl);
    if constexpr (!props.vdwComb)
    {
        cgh.require(a_atomTypes);
        cgh.require(a_nbfp);
    }
    else
    {
        cgh.require(a_ljComb);
    }
    if constexpr (props.vdwEwald)
    {
        cgh.require(a_nbfpComb);
    }
    if constexpr (props.elecEwaldTab)
    {
        cgh.require(a_coulombType);
    }

    // shmem buffer for i x+q pre-loading
    cl::sycl::accessor<float4, 2, mode::read_write, target::local> xqib(
            cl::sycl::range<2>(c_nbnxnGpuNumClusterPerSupercluster, c_clSize), cgh);
    // shmem buffer for cj, for each warp separately
    // the cjs buffer's use expects a base pointer offset for pairs of warps in the j-concurrent execution
    cl::sycl::accessor<int, 2, mode::read_write, target::local> cjs(
            cl::sycl::range<2>(NTHREAD_Z, c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize), cgh);

    // shmem buffer for j- and i-forces
    // SYCL-TODO: Make into 3D
    cl::sycl::accessor<float, 1, mode::read_write, target::local> force_j_buf_shmem(
            cl::sycl::range<1>(c_clSize * c_clSize * NTHREAD_Z * 3), cgh); // 3 for float3

    auto atib = [&]() {
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

    auto ljcpib = [&]() {
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

    /* Flag to control the calculation of exclusion forces in the kernel
     * We do that with Ewald (elec/vdw) and RF. Cut-off only has exclusion
     * energy terms. */
    constexpr bool doExclusionForces =
            (props.elecEwald || props.elecRF || props.vdwEwald || (props.elecCutoff && doCalcEnergies));

    return [=](cl::sycl::nd_item<1> itemIdx) [[intel::reqd_sub_group_size(8)]]
    {
        /* thread/block/warp id-s */
        const cl::sycl::id<3> localId = unflattenId<c_clSize, c_clSize>(itemIdx.get_local_id());
        const unsigned        tidxi   = localId[0];
        const unsigned        tidxj   = localId[1];
        const cl::sycl::id<2> tidxji(localId[0], localId[1]);
        const unsigned        tidx = tidxj * itemIdx.get_group_range(0) + tidxi;
#if NTHREAD_Z == 1
        const unsigned tidxz = 0;
#else
        const unsigned tidxz = localId[2];
#endif
        const unsigned bidx = itemIdx.get_group(0);

        const sycl_pf::sub_group sg                         = itemIdx.get_sub_group();
        const unsigned           widx                       = sg.get_group_id(); // warp index
        float3 fci_buf[c_nbnxnGpuNumClusterPerSupercluster] = { { 0.0F, 0.0F, 0.0F } }; // i force buffer

        const nbnxn_sci_t nb_sci     = a_plistSci[bidx];
        const int         sci        = nb_sci.sci;
        const int         cij4_start = nb_sci.cj4_ind_start;
        const int         cij4_end   = nb_sci.cj4_ind_end;

        // Only needed if props.elecEwaldAna
        const float beta2 = ewaldBeta * ewaldBeta;
        const float beta3 = beta2 * ewaldBeta;

        bool doCalcShift = calcShift;

        float4 xqbuf;

        if (tidxz == 0)
        {
            for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i += c_clSize)
            {
                /* Pre-load i-atom x and q into shared memory */
                const int             ci = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxj + i;
                const int             ai = ci * c_clSize + tidxi;
                const cl::sycl::id<2> cacheIdx = cl::sycl::id<2>(tidxj + i, tidxi);

                float3 shift = a_shiftVec[nb_sci.shift];
                xqbuf        = a_xq[ai];
                xqbuf += float4(shift[0], shift[1], shift[2], 0.0F);
                xqbuf[3] *= epsFac;
                xqib[cacheIdx] = xqbuf;

                if constexpr (!props.vdwComb)
                {
                    // Pre-load the i-atom types into shared memory
                    atib[cacheIdx] = a_atomTypes[ai];
                }
                else
                {
                    // Pre-load the LJ combination parameters into shared memory
                    ljcpib[cacheIdx] = a_ljComb[ai];
                }
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
                        E_lj *= 0.5F * c_oneSixth * lje_coeff6_6; // c_OneTwelfth?
                    }
                    if constexpr (props.elecEwald || props.elecRF || props.elecCutoff)
                    {
                        // Correct for epsfac^2 due to adding qi^2 */
                        E_el /= epsFac * c_clSize * NTHREAD_Z;
                        if constexpr (props.elecRF || props.elecCutoff)
                        {
                            E_el *= -0.5F * c_rf;
                        }
                        else
                        {
                            E_el *= -ewaldBeta * c_OneOverSqrtPi; /* last factor 1/sqrt(pi) */
                        }
                    }
                } // (nb_sci.shift == CENTRAL && a_plistCJ4[cij4_start].cj[0] == sci * c_nbnxnGpuNumClusterPerSupercluster)
            }     // (doExclusionForces)
        }         // (doCalcEnergies)

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
                /*
                if ((tidxj == 0 || tidxj == 4) && (tidxi < c_nbnxnGpuJgroupSize))
                {
                    cjs[cl::sycl::id<2>(tidxz, tidxi + tidxj * c_nbnxnGpuJgroupSize /
                c_splitClSize)] = a_plistCJ4[j4].cj[tidxi];
                }
                sg.barrier();
                 */

                for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
                {
                    if (imask & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster)))
                    {
                        unsigned mask_ji = (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster));
                        /*
                        const int cj =
                                cjs[cl::sycl::id<2>(tidxz, jm + (tidxj & 4) * c_nbnxnGpuJgroupSize / c_splitClSize)];
                                */
                        const int cj = a_plistCJ4[j4].cj[jm];
                        const int aj = cj * c_clSize + tidxj;

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
                                    if (!sycl_pf::group_any_of(sg, r2 < rlistOuterSq))
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
                                                     * (c12 * (inv_r6 * inv_r6 + repulsion_shift.cpot) * c_oneTwelfth
                                                        - c6 * (inv_r6 + repulsion_shift.cpot) * c_oneSixth);
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
                                        F_invr = epsilon * sig_r6 * (sig_r6 - 1.0F) * inv_r2;
                                    } // (!props.vdwCombLB || doCalcEnergies)
                                    if constexpr (props.vdwFSwitch)
                                    {
                                        if constexpr (doCalcEnergies)
                                        {
                                            calculate_force_switch_F_E(
                                                    dispersion_shift, repulsion_shift, rVdwSwitch,
                                                    c6, c12, inv_r, r2, &F_invr, &E_lj_p);
                                        }
                                        else
                                        {
                                            calculate_force_switch_F(dispersion_shift,
                                                                     repulsion_shift, rVdwSwitch,
                                                                     c6, c12, inv_r, r2, &F_invr);
                                        }
                                    }
                                    if constexpr (props.vdwEwald)
                                    {
                                        if constexpr (props.vdwEwaldCombGeom)
                                        {
                                            if constexpr (doCalcEnergies)
                                            {
                                                calculate_lj_ewald_comb_geom_F_E(
                                                        a_nbfpComb, sh_lj_ewald, typei, typej, r2,
                                                        inv_r2, lje_coeff2, lje_coeff6_6, int_bit,
                                                        &F_invr, &E_lj_p);
                                            }
                                            else
                                            {
                                                calculate_lj_ewald_comb_geom_F(
                                                        a_nbfpComb, typei, typej, r2, inv_r2,
                                                        lje_coeff2, lje_coeff6_6, &F_invr);
                                            }
                                        }
                                        else if constexpr (props.vdwEwaldCombLB)
                                        {
                                            calculate_lj_ewald_comb_LB_F_E(
                                                    a_nbfpComb, sh_lj_ewald, typei, typej, r2,
                                                    inv_r2, lje_coeff2, lje_coeff6_6,
                                                    (doCalcEnergies ? int_bit : 0), &F_invr,
                                                    (doCalcEnergies ? &E_lj_p : nullptr));
                                        }
                                    } // (props.vdwEwald)
                                    if constexpr (props.vdwPSwitch)
                                    {
                                        if constexpr (doCalcEnergies)
                                        {
                                            calculate_potential_switch_F_E(vdw_switch, rVdwSwitch,
                                                                           inv_r, r2, &F_invr, &E_lj_p);
                                        }
                                        else
                                        {
                                            calculate_potential_switch_F(vdw_switch, rVdwSwitch,
                                                                         inv_r, r2, &F_invr, &E_lj_p);
                                        }
                                    }
                                    if constexpr (vdwCutoffCheck<elecType>)
                                    {
                                        /* Separate VDW cut-off check to enable twin-range cut-offs
                                         * (rvdw < rcoulomb <= rlist) */
                                        const float vdwInRange = (r2 < rVdwSq) ? 1.0F : 0.0F;
                                        F_invr *= vdwInRange;
                                        if constexpr (doCalcEnergies)
                                        {
                                            E_lj_p *= vdwInRange;
                                        }
                                    }
                                    if constexpr (doCalcEnergies)
                                    {
                                        E_lj += E_lj_p;
                                    }

                                    if constexpr (props.elecCutoff)
                                    {
                                        if constexpr (doExclusionForces)
                                        {
                                            F_invr += qi * qj_f * int_bit * inv_r2 * inv_r;
                                        }
                                        else
                                        {
                                            F_invr += qi * qj_f * inv_r2 * inv_r;
                                        }
                                    }
                                    if constexpr (props.elecRF)
                                    {
                                        F_invr += qi * qj_f * (int_bit * inv_r2 * inv_r - twoKRf);
                                    }
                                    if constexpr (props.elecEwaldAna)
                                    {
                                        F_invr += qi * qj_f
                                                  * (int_bit * inv_r2 * inv_r + pmecorrF(beta2 * r2) * beta3);
                                    }
                                    else if constexpr (props.elecEwaldTab)
                                    {
                                        F_invr += qi * qj_f
                                                  * (int_bit * inv_r2
                                                     - interpolate_coulomb_force_r(
                                                               a_coulombType, coulombTabScale, r2 * inv_r))
                                                  * inv_r;
                                    }

                                    if constexpr (doCalcEnergies)
                                    {
                                        if constexpr (props.elecCutoff)
                                        {
                                            E_el += qi * qj_f * (int_bit * inv_r - c_rf);
                                        }
                                        if constexpr (props.elecRF)
                                        {
                                            E_el += qi * qj_f
                                                    * (int_bit * inv_r + 0.5f * twoKRf * r2 - c_rf);
                                        }
                                        if constexpr (props.elecEwald)
                                        {
                                            E_el += qi * qj_f
                                                    * (inv_r * (int_bit - erff(r2 * inv_r * ewaldBeta))
                                                       - int_bit * ewaldShift);
                                        }
                                    }

                                    const float3 f_ij = rv * F_invr;

                                    /* accumulate j forces in registers */
                                    fcj_buf -= f_ij;

                                    /* accumulate i forces in registers */
                                    fci_buf[i] += f_ij;
                                } // (r2 < rCoulombSq) && notExcluded
                            }     // (imask & mask_ji)
                            /* shift the mask bit by 1 */
                            mask_ji += mask_ji;
                        } // for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
                        // Replace with group_reduce in SYCL2020
                        /* reduce j forces */
                        reduce_force_j(force_j_buf_shmem, fcj_buf, a_f, itemIdx, tidxi, tidxj, aj);
                    } // (imask & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster)))
                } // for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
                if constexpr (doPruneNBL)
                {
                    /* Update the imask with the new one which does not contain the
                     * out of range clusters anymore. */
                    a_plistCJ4[j4].imei[widx].imask = imask;
                }
            } // (doPruneNBL || imask)

            // avoid shared memory WAR hazards between loop iterations
            sg.barrier();
        } // for (int j4 = cij4_start + tidxz; j4 < cij4_end; j4 += NTHREAD_Z)

        /* skip central shifts when summing shift forces */

        if (nb_sci.shift == CENTRAL)
        {
            doCalcShift = false;
        }

        reduce_force_i_and_shift(force_j_buf_shmem, fci_buf, a_f, doCalcShift, itemIdx, tidxi,
                                 tidxj, sci, nb_sci.shift, a_fShift);

        if constexpr (doCalcEnergies)
        {
            const float E_lj_wg =
                    sycl_pf::group_reduce(itemIdx.get_group(), E_lj, 0.0F, sycl_pf::plus<float>());
            const float E_el_wg =
                    sycl_pf::group_reduce(itemIdx.get_group(), E_el, 0.0F, sycl_pf::plus<float>());
            if (tidx == 0)
            {
                atomic_fetch_add(a_vdwEnergy, 0, E_lj_wg);
                atomic_fetch_add(a_elecEnergy, 0, E_el_wg);
            }
        }
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
        cgh.parallel_for<kernelNameType>(flattenNDRange(range), kernel);
    });

    return e;
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

    // Casting to float simplifies using atomic ops in the kernel
    cl::sycl::buffer<float3, 1> f(*adat->f.buffer_);
    auto                        f_as_float = f.reinterpret<float, 1>(f.get_count() * 3);
    cl::sycl::buffer<float3, 1> fShift(*adat->fShift.buffer_);
    auto fShift_as_float = fShift.reinterpret<float, 1>(fShift.get_count() * 3);

    cl::sycl::event e = chooseAndLaunchNbnxmKernel(
            doPruneNBL, stepWork.computeEnergy, nbp->elecType, nbp->vdwType, deviceStream,
            plist->nsci, adat->xq, f_as_float, adat->shiftVec, fShift_as_float, adat->eElec,
            adat->eLJ, plist->cj4, plist->sci, plist->excl, adat->atomTypes, adat->ljComb, nbp->nbfp,
            nbp->nbfp_comb, nbp->coulomb_tab, nbp->rcoulomb_sq, nbp->rvdw_sq, nbp->two_k_rf,
            nbp->ewald_beta, nbp->rlistOuter_sq, nbp->sh_ewald, nbp->epsfac, nbp->ewaldcoeff_lj,
            adat->numTypes, nbp->c_rf, nbp->dispersion_shift, nbp->repulsion_shift, nbp->vdw_switch,
            nbp->rvdw_switch, nbp->sh_lj_ewald, nbp->coulomb_tab_scale, stepWork.computeVirial);

    e.wait_and_throw(); // SYCL-TODO: remove
}

} // namespace Nbnxm
