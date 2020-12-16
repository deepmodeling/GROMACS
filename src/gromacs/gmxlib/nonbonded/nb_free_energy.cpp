/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2017 by the GROMACS development team.
 * Copyright (c) 2018,2019,2020, by the GROMACS development team, led by
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
#include "gmxpre.h"

#include "nb_free_energy.h"

#include "config.h"

#include <cmath>

#include <algorithm>

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gmxlib/nonbonded/nb_kernel.h"
#include "gromacs/gmxlib/nonbonded/nonbonded.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/simd/simd.h"
#include "gromacs/simd/simd_math.h"
#include "gromacs/utility/fatalerror.h"

#include "nb_softcore.h"

//! Scalar (non-SIMD) data types.
struct ScalarDataTypes
{
    using RealType = real; //!< The data type to use as real.
    using IntType  = int;  //!< The data type to use as int.
    using BoolType = bool; //!< The data type to use as bool for real value comparison.
    static constexpr int simdRealWidth = 1; //!< The width of the RealType.
    static constexpr int simdIntWidth  = 1; //!< The width of the IntType.
};

#if GMX_SIMD_HAVE_REAL && GMX_SIMD_HAVE_INT32_ARITHMETICS
//! SIMD data types.
struct SimdDataTypes
{
    using RealType = gmx::SimdReal;  //!< The data type to use as real.
    using IntType  = gmx::SimdInt32; //!< The data type to use as int.
    using BoolType = gmx::SimdBool;  //!< The data type to use as bool for real value comparison.
    static constexpr int simdRealWidth = GMX_SIMD_REAL_WIDTH; //!< The width of the RealType.
#    if GMX_SIMD_HAVE_DOUBLE && GMX_DOUBLE
    static constexpr int simdIntWidth = GMX_SIMD_DINT32_WIDTH; //!< The width of the IntType.
#    else
    static constexpr int simdIntWidth = GMX_SIMD_FINT32_WIDTH; //!< The width of the IntType.
#    endif
};
#endif

/* Retrieve value (specified by index) from table if mask is true */
static inline real retrieveFromTable(const real* table, int index, bool mask)
{
    return (mask ? table[index] : 0.0F);
}

#if GMX_SIMD_HAVE_REAL && GMX_SIMD_HAVE_INT32_ARITHMETICS && GMX_USE_SIMD_KERNELS
/* Retrieve GMX_SIMD_REAL_WIDTH values (specified by index) from table if the corresponding mask values are true */
static inline gmx::SimdReal retrieveFromTable(const real* table, gmx::SimdInt32 index, const gmx::SimdBool mask)
{
#    if GMX_SIMD_HAVE_DOUBLE && GMX_DOUBLE
    GMX_ASSERT(GMX_SIMD_REAL_WIDTH == GMX_SIMD_DINT32_WIDTH,
               "Mismatch between SIMD real and integer sizes.");
#    else
    GMX_ASSERT(GMX_SIMD_REAL_WIDTH == GMX_SIMD_FINT32_WIDTH,
               "Mismatch between SIMD real and integer sizes.");
#    endif

    gmx::SimdReal res;
    std::int32_t  extractedIndex[GMX_SIMD_REAL_WIDTH];
    extractedIndex[0] = gmx::extract<0>(index);
#    if GMX_SIMD_REAL_WIDTH >= 2
    extractedIndex[1] = gmx::extract<1>(index);
#    endif
#    if GMX_SIMD_REAL_WIDTH >= 4
    extractedIndex[2] = gmx::extract<2>(index);
    extractedIndex[3] = gmx::extract<3>(index);
#    endif
#    if GMX_SIMD_REAL_WIDTH >= 6
    extractedIndex[4] = gmx::extract<4>(index);
    extractedIndex[5] = gmx::extract<5>(index);
#    endif
#    if GMX_SIMD_REAL_WIDTH >= 8
    extractedIndex[6] = gmx::extract<6>(index);
    extractedIndex[7] = gmx::extract<7>(index);
#    endif

    for (std::size_t i = 0; i < GMX_SIMD_REAL_WIDTH; i++)
    {
        res.simdInternal_[i] = mask.simdInternal_[i] ? table[extractedIndex[i]] : 0.0;
    }
    return res;
}
#endif

//! Computes r^(1/p) and 1/r^(1/p) for the standard p=6
template<class RealType, class BoolType>
static inline void pthRoot(const RealType r, RealType* pthRoot, RealType* invPthRoot, const BoolType mask)
{
    RealType cbrtRes = gmx::cbrt(r);
    *invPthRoot      = gmx::maskzInvsqrt(cbrtRes, mask);
    *pthRoot         = gmx::maskzInv(*invPthRoot, mask);
}

template<class RealType>
static inline RealType calculateRinv6(const RealType rInvV)
{
    RealType rInv6 = rInvV * rInvV;
    return (rInv6 * rInv6 * rInv6);
}

template<class RealType>
static inline RealType calculateVdw6(const RealType c6, const RealType rInv6)
{
    return (c6 * rInv6);
}

template<class RealType>
static inline RealType calculateVdw12(const RealType c12, const RealType rInv6)
{
    return (c12 * rInv6 * rInv6);
}

/* reaction-field electrostatics */
template<class RealType>
static inline RealType reactionFieldScalarForce(const RealType qq,
                                                const RealType rInv,
                                                const RealType r,
                                                const real     krf,
                                                const real     two)
{
    return (qq * (rInv - two * krf * r * r));
}
template<class RealType>
static inline RealType reactionFieldPotential(const RealType qq,
                                              const RealType rInv,
                                              const RealType r,
                                              const real     krf,
                                              const real     potentialShift)
{
    return (qq * (rInv + krf * r * r - potentialShift));
}

/* Ewald electrostatics */
template<class RealType>
static inline RealType ewaldScalarForce(const RealType coulomb, const RealType rInv)
{
    return (coulomb * rInv);
}
template<class RealType>
static inline RealType ewaldPotential(const RealType coulomb, const RealType rInv, const real potentialShift)
{
    return (coulomb * (rInv - potentialShift));
}

/* cutoff LJ */
template<class RealType>
static inline RealType lennardJonesScalarForce(const RealType v6, const RealType v12)
{
    return (v12 - v6);
}
template<class RealType>
static inline RealType lennardJonesPotential(const RealType v6,
                                             const RealType v12,
                                             const RealType c6,
                                             const RealType c12,
                                             const real     repulsionShift,
                                             const real     dispersionShift,
                                             const real     oneSixth,
                                             const real     oneTwelfth)
{
    return ((v12 + c12 * repulsionShift) * oneTwelfth - (v6 + c6 * dispersionShift) * oneSixth);
}

/* Ewald LJ */
template<class RealType>
static inline RealType ewaldLennardJonesGridSubtract(const RealType c6grid,
                                                     const real     potentialShift,
                                                     const real     oneSixth)
{
    return (c6grid * potentialShift * oneSixth);
}

/* LJ Potential switch */
template<class RealType, class BoolType>
static inline RealType potSwitchScalarForceMod(const RealType fScalarInp,
                                               const RealType potential,
                                               const RealType sw,
                                               const RealType r,
                                               const RealType dsw,
                                               const BoolType mask)
{
    /* The mask should select on rV < rVdw */
    return (gmx::selectByMask(fScalarInp * sw - r * potential * dsw, mask));
}
template<class RealType, class BoolType>
static inline RealType potSwitchPotentialMod(const RealType potentialInp, const RealType sw, const BoolType mask)
{
    /* The mask should select on rV < rVdw */
    return (gmx::selectByMask(potentialInp * sw, mask));
}


//! Templated free-energy non-bonded kernel
template<typename DataTypes, SoftcoreType softcoreType, bool scLambdasOrAlphasDiffer, bool vdwInteractionTypeIsEwald, bool elecInteractionTypeIsEwald, bool vdwModifierIsPotSwitch>
static void nb_free_energy_kernel(const t_nblist* gmx_restrict nlist,
                                  rvec* gmx_restrict         xx,
                                  gmx::ForceWithShiftForces* forceWithShiftForces,
                                  const t_forcerec* gmx_restrict fr,
                                  const t_mdatoms* gmx_restrict mdatoms,
                                  nb_kernel_data_t* gmx_restrict kernel_data,
                                  t_nrnb* gmx_restrict nrnb)
{
#define STATE_A 0
#define STATE_B 1
#define NSTATES 2

    using RealType = typename DataTypes::RealType;
    using IntType  = typename DataTypes::IntType;
    using BoolType = typename DataTypes::BoolType;

    constexpr real oneTwelfth = 1.0 / 12.0;
    constexpr real oneSixth   = 1.0 / 6.0;
    constexpr real zero       = 0.0;
    constexpr real half       = 0.5;
    constexpr real one        = 1.0;
    constexpr real two        = 2.0;
    constexpr int  one_i      = 1;
    constexpr int  two_i      = 2;
    constexpr int  four_i     = 4;

    /* Extract pointer to non-bonded interaction constants */
    const interaction_const_t* ic = fr->ic;

    // Extract pair list data
    const int  nri    = nlist->nri;
    const int* iinr   = nlist->iinr;
    const int* jindex = nlist->jindex;
    const int* jjnr   = nlist->jjnr;
    const int* shift  = nlist->shift;
    const int* gid    = nlist->gid;

    const real* shiftvec      = fr->shift_vec[0];
    const real* chargeA       = mdatoms->chargeA;
    const real* chargeB       = mdatoms->chargeB;
    real*       Vc            = kernel_data->energygrp_elec;
    const int*  typeA         = mdatoms->typeA;
    const int*  typeB         = mdatoms->typeB;
    const int   ntype         = fr->ntype;
    const real* nbfp          = fr->nbfp.data();
    const real* nbfp_grid     = fr->ljpme_c6grid;
    real*       Vv            = kernel_data->energygrp_vdw;
    const real  lambda_coul   = kernel_data->lambda[efptCOUL];
    const real  lambda_vdw    = kernel_data->lambda[efptVDW];
    real*       dvdl          = kernel_data->dvdl;
    const auto& scParams      = *ic->softCoreParameters;
    const real  alpha_coul    = scParams.alphaCoulomb;
    const real  alpha_vdw     = scParams.alphaVdw;
    const real  lam_power     = scParams.lambdaPower;
    const real  sigma6_def    = scParams.sigma6WithInvalidSigma;
    const real  sigma6_min    = scParams.sigma6Minimum;
    const bool  doForces      = ((kernel_data->flags & GMX_NONBONDED_DO_FORCE) != 0);
    const bool  doShiftForces = ((kernel_data->flags & GMX_NONBONDED_DO_SHIFTFORCE) != 0);
    const bool  doPotential   = ((kernel_data->flags & GMX_NONBONDED_DO_POTENTIAL) != 0);

    // Extract data from interaction_const_t
    const real facel           = ic->epsfac;
    const real rCoulomb        = ic->rcoulomb;
    const real krf             = ic->k_rf;
    const real crf             = ic->c_rf;
    const real shLjEwald       = ic->sh_lj_ewald;
    const real rVdw            = ic->rvdw;
    const real dispersionShift = ic->dispersion_shift.cpot;
    const real repulsionShift  = ic->repulsion_shift.cpot;

    // Note that the nbnxm kernels do not support Coulomb potential switching at all
    GMX_ASSERT(ic->coulomb_modifier != eintmodPOTSWITCH,
               "Potential switching is not supported for Coulomb with FEP");

    const real rVdwSwitch(ic->rvdw_switch);
    real       vdw_swV3, vdw_swV4, vdw_swV5, vdw_swF2, vdw_swF3, vdw_swF4;
    if (vdwModifierIsPotSwitch)
    {
        const real d = ic->rvdw - ic->rvdw_switch;
        vdw_swV3     = -10.0 / (d * d * d);
        vdw_swV4     = 15.0 / (d * d * d * d);
        vdw_swV5     = -6.0 / (d * d * d * d * d);
        vdw_swF2     = -30.0 / (d * d * d);
        vdw_swF3     = 60.0 / (d * d * d * d);
        vdw_swF4     = -30.0 / (d * d * d * d * d);
    }
    else
    {
        /* Avoid warnings from stupid compilers (looking at you, Clang!) */
        vdw_swV3 = vdw_swV4 = vdw_swV5 = vdw_swF2 = vdw_swF3 = vdw_swF4 = 0.0;
    }

    int icoul;
    if (ic->eeltype == eelCUT || EEL_RF(ic->eeltype))
    {
        icoul = GMX_NBKERNEL_ELEC_REACTIONFIELD;
    }
    else
    {
        icoul = GMX_NBKERNEL_ELEC_NONE;
    }

    real rcutoff_max2 = std::max(ic->rcoulomb, ic->rvdw);
    rcutoff_max2      = rcutoff_max2 * rcutoff_max2;

    const real* tab_ewald_F_lj           = nullptr;
    const real* tab_ewald_V_lj           = nullptr;
    const real* ewtab                    = nullptr;
    real        coulombTableScale        = 0;
    real        coulombTableScaleInvHalf = 0;
    real        vdwTableScale            = 0;
    real        vdwTableScaleInvHalf     = 0;
    real        sh_ewald                 = 0;
    if (elecInteractionTypeIsEwald || vdwInteractionTypeIsEwald)
    {
        sh_ewald = ic->sh_ewald;
    }
    if (elecInteractionTypeIsEwald)
    {
        const auto& coulombTables = *ic->coulombEwaldTables;
        ewtab                     = coulombTables.tableFDV0.data();
        coulombTableScale         = coulombTables.scale;
        coulombTableScaleInvHalf  = half / coulombTableScale;
    }
    if (vdwInteractionTypeIsEwald)
    {
        const auto& vdwTables = *ic->vdwEwaldTables;
        tab_ewald_F_lj        = vdwTables.tableF.data();
        tab_ewald_V_lj        = vdwTables.tableV.data();
        vdwTableScale         = vdwTables.scale;
        vdwTableScaleInvHalf  = half / vdwTableScale;
    }

    /* For Ewald/PME interactions we cannot easily apply the soft-core component to
     * reciprocal space. When we use non-switched Ewald interactions, we
     * assume the soft-coring does not significantly affect the grid contribution
     * and apply the soft-core only to the full 1/r (- shift) pair contribution.
     *
     * However, we cannot use this approach for switch-modified since we would then
     * effectively end up evaluating a significantly different interaction here compared to the
     * normal (non-free-energy) kernels, either by applying a cutoff at a different
     * position than what the user requested, or by switching different
     * things (1/r rather than short-range Ewald). For these settings, we just
     * use the traditional short-range Ewald interaction in that case.
     */
    GMX_RELEASE_ASSERT(!(vdwInteractionTypeIsEwald && vdwModifierIsPotSwitch),
                       "Can not apply soft-core to switched Ewald potentials");

    RealType dvdlCoul(zero);
    RealType dvdlVdw(zero);

    /* Lambda factor for state A, 1-lambda*/
    real LFC[NSTATES], LFV[NSTATES];
    LFC[STATE_A] = one - lambda_coul;
    LFV[STATE_A] = one - lambda_vdw;

    /* Lambda factor for state B, lambda*/
    LFC[STATE_B] = lambda_coul;
    LFV[STATE_B] = lambda_vdw;

    /*derivative of the lambda factor for state A and B */
    real DLF[NSTATES];
    DLF[STATE_A] = -one;
    DLF[STATE_B] = one;

    real           lFacCoul[NSTATES], dlFacCoul[NSTATES], lFacVdw[NSTATES], dlFacVdw[NSTATES];
    constexpr real sc_r_power = 6.0_real;
    for (int i = 0; i < NSTATES; i++)
    {
        lFacCoul[i]  = (lam_power == 2 ? (1 - LFC[i]) * (1 - LFC[i]) : (1 - LFC[i]));
        dlFacCoul[i] = DLF[i] * lam_power / sc_r_power * (lam_power == 2 ? (1 - LFC[i]) : 1);
        lFacVdw[i]   = (lam_power == 2 ? (1 - LFV[i]) * (1 - LFV[i]) : (1 - LFV[i]));
        dlFacVdw[i]  = DLF[i] * lam_power / sc_r_power * (lam_power == 2 ? (1 - LFV[i]) : 1);
    }

    // TODO: We should get rid of using pointers to real
    const real* x             = xx[0];
    real* gmx_restrict f      = &(forceWithShiftForces->force()[0][0]);
    real* gmx_restrict fshift = &(forceWithShiftForces->shiftForces()[0][0]);

    for (int n = 0; n < nri; n++)
    {
        int npair_within_cutoff = 0;

        const int  is3  = 3 * shift[n];
        const real shX  = shiftvec[is3];
        const real shY  = shiftvec[is3 + 1];
        const real shZ  = shiftvec[is3 + 2];
        const int  nj0  = jindex[n];
        const int  nj1  = jindex[n + 1];
        const int  ii   = iinr[n];
        const int  ii3  = 3 * ii;
        const real ix   = shX + x[ii3 + 0];
        const real iy   = shY + x[ii3 + 1];
        const real iz   = shZ + x[ii3 + 2];
        const real iqA  = facel * chargeA[ii];
        const real iqB  = facel * chargeB[ii];
        const int  ntiA = 2 * ntype * typeA[ii];
        const int  ntiB = 2 * ntype * typeB[ii];
        RealType   vCTot(0);
        RealType   vVTot(0);
        RealType   fIX(0);
        RealType   fIY(0);
        RealType   fIZ(0);

#if GMX_SIMD_HAVE_REAL
        alignas(GMX_SIMD_ALIGNMENT) int preloadIi[DataTypes::simdRealWidth];
        alignas(GMX_SIMD_ALIGNMENT) int preloadIs[DataTypes::simdRealWidth];
#else
        int preloadIi[DataTypes::simdRealWidth];
        int preloadIs[DataTypes::simdRealWidth];
#endif
        for (int s = 0; s < DataTypes::simdRealWidth; s++)
        {
            preloadIi[s] = ii;
            preloadIs[s] = shift[n];
        }
        IntType ii_s = gmx::load<IntType>(preloadIi);

        for (int k = nj0; k < nj1; k += DataTypes::simdRealWidth)
        {
            int      tj[NSTATES];
            RealType r, rInv;

#if GMX_SIMD_HAVE_REAL
            alignas(GMX_SIMD_ALIGNMENT) real preloadPairIncluded[DataTypes::simdRealWidth];
            alignas(GMX_SIMD_ALIGNMENT) int  preloadJnr[DataTypes::simdRealWidth];
            alignas(GMX_SIMD_ALIGNMENT) real preloadDx[DataTypes::simdRealWidth];
            alignas(GMX_SIMD_ALIGNMENT) real preloadDy[DataTypes::simdRealWidth];
            alignas(GMX_SIMD_ALIGNMENT) real preloadDz[DataTypes::simdRealWidth];
            alignas(GMX_SIMD_ALIGNMENT) real preloadC6[NSTATES][DataTypes::simdRealWidth],
                    preloadC12[NSTATES][DataTypes::simdRealWidth];
            alignas(GMX_SIMD_ALIGNMENT) real preloadQq[NSTATES][DataTypes::simdRealWidth];
            alignas(GMX_SIMD_ALIGNMENT) real preloadSigma6[NSTATES][DataTypes::simdRealWidth];
            alignas(GMX_SIMD_ALIGNMENT) real preloadAlphaVdwEff[DataTypes::simdRealWidth];
            alignas(GMX_SIMD_ALIGNMENT) real preloadAlphaCoulEff[DataTypes::simdRealWidth];
            alignas(GMX_SIMD_ALIGNMENT) real preloadLjPmeC6Grid[NSTATES][DataTypes::simdRealWidth];
#else
            real preloadPairIncluded[DataTypes::simdRealWidth];
            int  preloadJnr[DataTypes::simdRealWidth];
            real preloadDx[DataTypes::simdRealWidth];
            real preloadDy[DataTypes::simdRealWidth];
            real preloadDz[DataTypes::simdRealWidth];
            real preloadC6[NSTATES][DataTypes::simdRealWidth],
                    preloadC12[NSTATES][DataTypes::simdRealWidth];
            real preloadQq[NSTATES][DataTypes::simdRealWidth];
            real preloadSigma6[NSTATES][DataTypes::simdRealWidth];
            real preloadAlphaVdwEff[DataTypes::simdRealWidth];
            real preloadAlphaCoulEff[DataTypes::simdRealWidth];
            real preloadLjPmeC6Grid[NSTATES][DataTypes::simdRealWidth];
#endif
            for (int s = 0; s < DataTypes::simdRealWidth; s++)
            {
                if (k + s < nj1)
                {
                    /* Check if this pair on the exclusions list.*/
                    preloadPairIncluded[s] = (nlist->excl_fep == nullptr || nlist->excl_fep[k + s]);
                    const int jnr          = jjnr[k + s];
                    const int j3           = 3 * jnr;
                    preloadJnr[s]          = jnr;
                    preloadDx[s]           = ix - x[j3];
                    preloadDy[s]           = iy - x[j3 + 1];
                    preloadDz[s]           = iz - x[j3 + 2];
                    tj[STATE_A]            = ntiA + 2 * typeA[jnr];
                    tj[STATE_B]            = ntiB + 2 * typeB[jnr];
                    preloadQq[STATE_A][s]  = iqA * chargeA[jnr];
                    preloadQq[STATE_B][s]  = iqB * chargeB[jnr];

                    for (int i = 0; i < NSTATES; i++)
                    {
                        preloadC6[i][s]  = nbfp[tj[i]];
                        preloadC12[i][s] = nbfp[tj[i] + 1];
                        if (vdwInteractionTypeIsEwald)
                        {
                            preloadLjPmeC6Grid[i][s] = nbfp_grid[tj[i]];
                        }
                        else
                        {
                            preloadLjPmeC6Grid[i][s] = 0;
                        }
                        if (softcoreType == SoftcoreType::Beutler || softcoreType == SoftcoreType::Gapsys)
                        {
                            if ((preloadC6[i][s] > 0) && (preloadC12[i][s] > 0))
                            {
                                /* c12 is stored scaled with 12.0 and c6 is scaled with 6.0 - correct for this */
                                preloadSigma6[i][s] = 0.5 * preloadC12[i][s] / preloadC6[i][s];
                                if (preloadSigma6[i][s]
                                    < sigma6_min) /* for disappearing coul and vdw with soft core at the same time */
                                {
                                    preloadSigma6[i][s] = sigma6_min;
                                }
                            }
                            else
                            {
                                preloadSigma6[i][s] = sigma6_def;
                            }
                        }
                    }
                    if (softcoreType == SoftcoreType::Beutler || softcoreType == SoftcoreType::Gapsys)
                    {
                        /* only use softcore if one of the states has a zero endstate - softcore is for avoiding infinities!*/
                        if ((preloadC12[STATE_A][s] > 0) && (preloadC12[STATE_B][s] > 0))
                        {
                            preloadAlphaVdwEff[s]  = 0;
                            preloadAlphaCoulEff[s] = 0;
                        }
                        else
                        {
                            preloadAlphaVdwEff[s]  = alpha_vdw;
                            preloadAlphaCoulEff[s] = alpha_coul;
                        }
                    }
                }
                else
                {
                    preloadJnr[s] = 0;
                    /* These must be set to make sure that the distance is beyond the cutoff. */
                    preloadDx[s] = rcutoff_max2;
                    preloadDy[s] = rcutoff_max2;
                    preloadDz[s] = rcutoff_max2;
                    /* Only pairs that are included can be skipped completely. */
                    preloadPairIncluded[s] = true;
                    preloadAlphaVdwEff[s]  = 0;
                    preloadAlphaCoulEff[s] = 0;

                    for (int i = 0; i < NSTATES; i++)
                    {
                        preloadQq[i][s]     = 0;
                        preloadC6[i][s]     = 0;
                        preloadC12[i][s]    = 0;
                        preloadSigma6[i][s] = 0;
                    }
                }
            }
            const RealType pairIncluded     = gmx::load<RealType>(preloadPairIncluded);
            const BoolType bPairIncluded    = (pairIncluded != zero);
            const BoolType bPairNotIncluded = (pairIncluded == zero);

            const RealType dX  = gmx::load<RealType>(preloadDx);
            const RealType dY  = gmx::load<RealType>(preloadDy);
            const RealType dZ  = gmx::load<RealType>(preloadDz);
            const RealType rSq = dX * dX + dY * dY + dZ * dZ;

            BoolType withinCutoffMask = (rSq < rcutoff_max2);

            if (!gmx::anyTrue(withinCutoffMask || bPairNotIncluded))
            {
                /* We save significant time by skipping all code below.
                 * Note that with soft-core interactions, the actual cut-off
                 * check might be different. But since the soft-core distance
                 * is always larger than r, checking on r here is safe.
                 * Exclusions outside the cutoff can not be skipped as
                 * when using Ewald: the reciprocal-space
                 * Ewald component still needs to be subtracted.
                 */
                continue;
            }
            npair_within_cutoff++; /* It is not necessary to actually count how many are within cutoff as long as it gets > 0 */
            const IntType  jnr_s    = gmx::load<IntType>(preloadJnr);
            const BoolType bIiEqJnr = gmx::cvtIB2B(ii_s == jnr_s);

            RealType c6[NSTATES];
            RealType c12[NSTATES];
            RealType sigma6[NSTATES];
            RealType qq[NSTATES];
            RealType ljPmeC6Grid[NSTATES];
            RealType alphaVdwEff;
            RealType alphaCoulEff;
            for (int i = 0; i < NSTATES; i++)
            {
                c6[i]          = gmx::load<RealType>(preloadC6[i]);
                c12[i]         = gmx::load<RealType>(preloadC12[i]);
                qq[i]          = gmx::load<RealType>(preloadQq[i]);
                ljPmeC6Grid[i] = gmx::load<RealType>(preloadLjPmeC6Grid[i]);
                if (softcoreType == SoftcoreType::Beutler || softcoreType == SoftcoreType::Gapsys)
                {
                    sigma6[i] = gmx::load<RealType>(preloadSigma6[i]);
                }
            }
            if (softcoreType == SoftcoreType::Beutler || softcoreType == SoftcoreType::Gapsys)
            {
                alphaVdwEff  = gmx::load<RealType>(preloadAlphaVdwEff);
                alphaCoulEff = gmx::load<RealType>(preloadAlphaCoulEff);
            }

            BoolType rSqValid = (zero < rSq);

            /* The force at r=0 is zero, because of symmetry.
             * But note that the potential is in general non-zero,
             * since the soft-cored r will be non-zero.
             */
            rInv = gmx::maskzInvsqrt(rSq, rSqValid);
            r    = rSq * rInv;

            RealType rp, rpm2;
            if (softcoreType == SoftcoreType::Beutler)
            {
                rpm2 = rSq * rSq;  /* r4 */
                rp   = rpm2 * rSq; /* r6 */
            }
            else
            {
                /* The soft-core power p will not affect the results
                 * with not using soft-core, so we use power of 0 which gives
                 * the simplest math and cheapest code.
                 */
                rpm2 = rInv * rInv;
                rp   = one;
            }

            RealType fScal(0);

            /* The following block is masked to only calculate values having bPairIncluded. If
             * bPairIncluded is true then withinCutoffMask must also be true. */
            if (gmx::anyTrue(withinCutoffMask && bPairIncluded))
            {
                RealType fScalC[NSTATES], fScalV[NSTATES];
                RealType vCoul[NSTATES], vVdw[NSTATES];
                for (int i = 0; i < NSTATES; i++)
                {
                    fScalC[i] = zero;
                    fScalV[i] = zero;
                    vCoul[i]  = zero;
                    vVdw[i]   = zero;

                    RealType rInvC, rInvV, rC, rV, rPInvC, rPInvV;

                    /* The following block is masked to require (qq[i] != 0 || c6[i] != 0 || c12[i]
                     * != 0) in addition to bPairIncluded, which in turn requires withinCutoffMask. */
                    BoolType nonZeroState = ((qq[i] != 0 || c6[i] != 0 || c12[i] != 0)
                                             && bPairIncluded && withinCutoffMask);
                    if (gmx::anyTrue(nonZeroState))
                    {
                        if (softcoreType == SoftcoreType::Beutler)
                        {
                            RealType divisor      = (alphaCoulEff * lFacCoul[i] * sigma6[i] + rp);
                            BoolType validDivisor = (zero < divisor);
                            rPInvC                = gmx::maskzInv(divisor, validDivisor);
                            pthRoot(rPInvC, &rInvC, &rC, validDivisor);

                            if (scLambdasOrAlphasDiffer)
                            {
                                RealType divisor      = (alphaVdwEff * lFacVdw[i] * sigma6[i] + rp);
                                BoolType validDivisor = (zero < divisor);
                                rPInvV                = gmx::maskzInv(divisor, validDivisor);
                                pthRoot(rPInvV, &rInvV, &rV, validDivisor);
                            }
                            else
                            {
                                /* We can avoid one expensive pow and one / operation */
                                rPInvV = rPInvC;
                                rInvV  = rInvC;
                                rV     = rC;
                            }
                        }
                        else
                        {
                            rPInvC = one;
                            rInvC  = rInv;
                            rC     = r;

                            rPInvV = one;
                            rInvV  = rInv;
                            rV     = r;
                        }

                        /* Only process the coulomb interactions if we either
                         * include all entries in the list (no cutoff
                         * used in the kernel), or if we are within the cutoff.
                         */
                        BoolType computeElecInteraction;
                        if (elecInteractionTypeIsEwald)
                        {
                            computeElecInteraction = (r < rCoulomb && qq[i] != 0 && bPairIncluded);
                        }
                        else
                        {
                            computeElecInteraction = (rC < rCoulomb && qq[i] != 0 && bPairIncluded);
                        }
                        if (gmx::anyTrue(computeElecInteraction))
                        {
                            if (elecInteractionTypeIsEwald)
                            {
                                vCoul[i]  = ewaldPotential(qq[i], rInvC, sh_ewald);
                                fScalC[i] = ewaldScalarForce(qq[i], rInvC);
                            }
                            else
                            {
                                vCoul[i]  = reactionFieldPotential(qq[i], rInvC, rC, krf, crf);
                                fScalC[i] = reactionFieldScalarForce(qq[i], rInvC, rC, krf, two);
                            }

                            if (softcoreType == SoftcoreType::Gapsys)
                            {
                                if (elecInteractionTypeIsEwald)
                                {
                                    ewaldQuadraticPotential(qq[i], rC, LFC[i], DLF[i], sigma6[i],
                                                            alphaCoulEff, sh_ewald, &fScalC[i],
                                                            &vCoul[i], &dvdlCoul,
                                                            computeElecInteraction);
                                }
                                else
                                {
                                    reactionFieldQuadraticPotential(
                                            qq[i], rC, LFC[i], DLF[i], sigma6[i], alphaCoulEff,
                                            krf, crf, &fScalC[i], &vCoul[i], &dvdlCoul,
                                            computeElecInteraction);
                                }
                            }

                            vCoul[i]  = gmx::selectByMask(vCoul[i], computeElecInteraction);
                            fScalC[i] = gmx::selectByMask(fScalC[i], computeElecInteraction);
                        }

                        /* Only process the VDW interactions if we either
                         * include all entries in the list (no cutoff used
                         * in the kernel), or if we are within the cutoff.
                         */
                        BoolType computeVdwInteraction;
                        if (vdwInteractionTypeIsEwald)
                        {
                            computeVdwInteraction =
                                    (r < rVdw && (c6[i] != 0 || c12[i] != 0) && bPairIncluded);
                        }
                        else
                        {
                            computeVdwInteraction =
                                    (rV < rVdw && (c6[i] != 0 || c12[i] != 0) && bPairIncluded);
                        }
                        if (gmx::anyTrue(computeVdwInteraction))
                        {
                            RealType rInv6;
                            if (softcoreType == SoftcoreType::Beutler)
                            {
                                rInv6 = rPInvV;
                            }
                            else
                            {
                                rInv6 = calculateRinv6(rInvV);
                            }
                            RealType vVdw6  = calculateVdw6(c6[i], rInv6);
                            RealType vVdw12 = calculateVdw12(c12[i], rInv6);

                            vVdw[i] = lennardJonesPotential(
                                    vVdw6, vVdw12, c6[i], c12[i], repulsionShift, dispersionShift, oneSixth, oneTwelfth);
                            fScalV[i] = lennardJonesScalarForce(vVdw6, vVdw12);

                            if (softcoreType == SoftcoreType::Gapsys)
                            {
                                lennardJonesQuadraticPotential(c6[i], c12[i], r, rSq, LFV[i],
                                                               DLF[i], sigma6[i], alphaCoulEff,
                                                               repulsionShift, dispersionShift,
                                                               &fScalV[i], &vVdw[i], &dvdlVdw,
                                                               computeVdwInteraction);
                            }

                            if (vdwInteractionTypeIsEwald)
                            {
                                /* Subtract the grid potential at the cut-off */
                                vVdw[i] = vVdw[i]
                                          + gmx::selectByMask(ewaldLennardJonesGridSubtract(
                                                                      ljPmeC6Grid[i], shLjEwald, oneSixth),
                                                              computeVdwInteraction);
                            }

                            if (vdwModifierIsPotSwitch)
                            {
                                RealType d             = rV - rVdwSwitch;
                                BoolType zeroMask      = zero < d;
                                BoolType potSwitchMask = rV < rVdw;
                                d                      = gmx::selectByMask(d, zeroMask);
                                const RealType d2      = d * d;
                                const RealType sw =
                                        one + d2 * d * (vdw_swV3 + d * (vdw_swV4 + d * vdw_swV5));
                                const RealType dsw = d2 * (vdw_swF2 + d * (vdw_swF3 + d * vdw_swF4));

                                fScalV[i] = potSwitchScalarForceMod(
                                        fScalV[i], vVdw[i], sw, rV, dsw, potSwitchMask);
                                vVdw[i] = potSwitchPotentialMod(vVdw[i], sw, potSwitchMask);
                            }

                            vVdw[i]   = gmx::selectByMask(vVdw[i], computeVdwInteraction);
                            fScalV[i] = gmx::selectByMask(fScalV[i], computeVdwInteraction);
                        }

                        /* fScalC (and fScalV) now contain: dV/drC * rC
                         * Now we multiply by rC^-p, so it will be: dV/drC * rC^1-p
                         * Further down we first multiply by r^p-2 and then by
                         * the vector r, which in total gives: dV/drC * (r/rC)^1-p
                         */
                        fScalC[i] = fScalC[i] * rPInvC;
                        fScalV[i] = fScalV[i] * rPInvV;
                    } // end of block requiring nonZeroState
                }     // end for (int i = 0; i < NSTATES; i++)

                /* Assemble A and B states. */
                BoolType assembleStates = (bPairIncluded && withinCutoffMask);
                if (gmx::anyTrue(assembleStates))
                {
                    for (int i = 0; i < NSTATES; i++)
                    {
                        vCTot = vCTot + LFC[i] * vCoul[i];
                        vVTot = vVTot + LFV[i] * vVdw[i];

                        fScal = fScal + LFC[i] * fScalC[i] * rpm2;
                        fScal = fScal + LFV[i] * fScalV[i] * rpm2;

                        if (softcoreType == SoftcoreType::Beutler)
                        {
                            dvdlCoul = dvdlCoul + vCoul[i] * DLF[i]
                                       + LFC[i] * alphaCoulEff * dlFacCoul[i] * fScalC[i] * sigma6[i];
                            dvdlVdw = dvdlVdw + vVdw[i] * DLF[i]
                                      + LFV[i] * alphaVdwEff * dlFacVdw[i] * fScalV[i] * sigma6[i];
                        }
                        else
                        {
                            dvdlCoul = dvdlCoul + vCoul[i] * DLF[i];
                            dvdlVdw  = dvdlVdw + vVdw[i] * DLF[i];
                        }
                    }
                }
            } // end of block requiring bPairIncluded && withinCutoffMask
            /* In the following block bPairIncluded should be false in the masks. */
            if (icoul == GMX_NBKERNEL_ELEC_REACTIONFIELD)
            {
                const BoolType computeReactionField = bPairNotIncluded;

                if (gmx::anyTrue(computeReactionField))
                {
                    /* For excluded pairs, which are only in this pair list when
                     * using the Verlet scheme, we don't use soft-core.
                     * As there is no singularity, there is no need for soft-core.
                     */
                    const RealType FF = -two * krf;
                    RealType       VV = krf * rSq - crf;

                    /* If ii == jnr the i particle (ii) has itself (jnr)
                     * in its neighborlist. This can only happen with the Verlet
                     * scheme, and corresponds to a self-interaction that will
                     * occur twice. Scale it down by 50% to only include it once.
                     */
                    VV = VV * gmx::blend(one, half, bIiEqJnr);

                    for (int i = 0; i < NSTATES; i++)
                    {
                        vCTot = vCTot + gmx::selectByMask(LFC[i] * qq[i] * VV, computeReactionField);
                        fScal = fScal + gmx::selectByMask(LFC[i] * qq[i] * FF, computeReactionField);
                        dvdlCoul = dvdlCoul + gmx::selectByMask(DLF[i] * qq[i] * VV, computeReactionField);
                    }
                }
            }

            /* In the following block the mask should require (r < rCoulomb || !bPairIncluded) */
            const BoolType computeElecEwaldInteraction = (r < rCoulomb || bPairNotIncluded);
            if (elecInteractionTypeIsEwald && gmx::anyTrue(computeElecEwaldInteraction))
            {
                /* See comment in the preamble. When using Ewald interactions
                 * (unless we use a switch modifier) we subtract the reciprocal-space
                 * Ewald component here which made it possible to apply the free
                 * energy interaction to 1/r (vanilla coulomb short-range part)
                 * above. This gets us closer to the ideal case of applying
                 * the softcore to the entire electrostatic interaction,
                 * including the reciprocal-space component.
                 */
                RealType v_lr, f_lr;

                const RealType ewrt   = r * coulombTableScale;
                IntType        ewitab = gmx::cvttR2I(ewrt);
                const RealType eweps  = ewrt - gmx::cvtI2R(ewitab);
                ewitab                = four_i * ewitab;

                f_lr = retrieveFromTable(ewtab, ewitab, computeElecEwaldInteraction);
                /* f_lr = ewtab[ewitab] + eweps * ewtab[ewitab + 1] */
                f_lr = f_lr + eweps * retrieveFromTable(ewtab, ewitab + one_i, computeElecEwaldInteraction);

                /* term1 = ewtab[ewitab + 2] */
                RealType term1 = retrieveFromTable(ewtab, ewitab + two_i, computeElecEwaldInteraction);
                /* term2 = coulombTableScaleInvHalf * eweps * (ewtab[ewitab] + f_lr) */
                RealType term2 = coulombTableScaleInvHalf * eweps
                                 * (retrieveFromTable(ewtab, ewitab, computeElecEwaldInteraction) + f_lr);
                /* v_lr = (ewtab[ewitab + 2] - coulombTableScaleInvHalf * eweps * (ewtab[ewitab] + f_lr)) */
                v_lr = term1 - term2;

                f_lr = f_lr * rInv;

                /* Note that any possible Ewald shift has already been applied in
                 * the normal interaction part above.
                 */

                /* If ii == jnr the i particle (ii) has itself (jnr)
                 * in its neighborlist. This can only happen with the Verlet
                 * scheme, and corresponds to a self-interaction that will
                 * occur twice. Scale it down by 50% to only include it once.
                 */
                v_lr = v_lr * gmx::blend(one, half, bIiEqJnr);

                for (int i = 0; i < NSTATES; i++)
                {
                    vCTot = vCTot - gmx::selectByMask(LFC[i] * qq[i] * v_lr, computeElecEwaldInteraction);
                    fScal = fScal - gmx::selectByMask(LFC[i] * qq[i] * f_lr, computeElecEwaldInteraction);
                    dvdlCoul = dvdlCoul
                               - gmx::selectByMask(DLF[i] * qq[i] * v_lr, computeElecEwaldInteraction);
                }
            }

            /* In the following block the mask should require (r < rVdw) */
            const BoolType computeVdwEwaldInteraction = r < rVdw;
            if (vdwInteractionTypeIsEwald && gmx::anyTrue(computeVdwEwaldInteraction))
            {
                /* See comment in the preamble. When using LJ-Ewald interactions
                 * (unless we use a switch modifier) we subtract the reciprocal-space
                 * Ewald component here which made it possible to apply the free
                 * energy interaction to r^-6 (vanilla LJ6 short-range part)
                 * above. This gets us closer to the ideal case of applying
                 * the softcore to the entire VdW interaction,
                 * including the reciprocal-space component.
                 */
                /* We could also use the analytical form here
                 * iso a table, but that can cause issues for
                 * r close to 0 for non-interacting pairs.
                 */

                const RealType rs        = rSq * rInv * vdwTableScale;
                const IntType  ri        = gmx::cvttR2I(rs);
                const RealType frac      = rs - gmx::cvtI2R(ri);
                const RealType otherFrac = one - frac;

                /* term1 = (1 - frac) * tab_ewald_F_lj[ri] */
                RealType term1 =
                        otherFrac * retrieveFromTable(tab_ewald_F_lj, ri, computeVdwEwaldInteraction);
                /* term2 = frac * tab_ewald_F_lj[ri + 1] */
                RealType term2 =
                        frac * retrieveFromTable(tab_ewald_F_lj, ri + one_i, computeVdwEwaldInteraction);
                /* f_lr = (1 - frac) * tab_ewald_F_lj[ri] + frac * tab_ewald_F_lj[ri + 1] */
                const RealType f_lr = term1 + term2;
                /* TODO: Currently the Ewald LJ table does not contain
                 * the factor 1/6, we should add this.
                 */
                const RealType FF = f_lr * rInv * oneSixth;

                term1 = retrieveFromTable(tab_ewald_V_lj, ri, computeVdwEwaldInteraction);
                term2 = vdwTableScaleInvHalf * frac
                        * (retrieveFromTable(tab_ewald_F_lj, ri, computeVdwEwaldInteraction) + f_lr);

                /* VV = (tab_ewald_V_lj[ri] - vdwTableScaleInvHalf * frac * (tab_ewald_F_lj[ri] + f_lr)) / six */
                RealType VV = (term1 - term2) * oneSixth;

                /* If ii == jnr the i particle (ii) has itself (jnr)
                 * in its neighborlist. This can only happen with the Verlet
                 * scheme, and corresponds to a self-interaction that will
                 * occur twice. Scale it down by 50% to only include it once.
                 */
                VV = VV * gmx::blend(one, half, bIiEqJnr);

                for (int i = 0; i < NSTATES; i++)
                {
                    vVTot = vVTot + gmx::selectByMask(LFV[i] * ljPmeC6Grid[i] * VV, computeVdwEwaldInteraction);
                    fScal = fScal + gmx::selectByMask(LFV[i] * ljPmeC6Grid[i] * FF, computeVdwEwaldInteraction);
                    dvdlVdw = dvdlVdw + gmx::selectByMask(DLF[i] * ljPmeC6Grid[i] * VV, computeVdwEwaldInteraction);
                }
            }

            /* Avoid expensive restricted omp access by first checking if there are any operations
             * to do. This can improve the performance significantly. */
            if (doForces && gmx::anyTrue(fScal != zero))
            {
                const RealType tX = fScal * dX;
                const RealType tY = fScal * dY;
                const RealType tZ = fScal * dZ;
                fIX               = fIX + tX;
                fIY               = fIY + tY;
                fIZ               = fIZ + tZ;

#pragma omp critical
                gmx::transposeScatterDecrU<3>(reinterpret_cast<real*>(f), preloadJnr, tX, tY, tZ);
            }
        } // end for (int k = nj0; k < nj1; k += DataTypes::simdRealWidth)

        if (npair_within_cutoff > 0)
        {
#pragma omp critical
            /* Avoid expensive restricted omp access by first checking if there are any operations
             * to do. This check, and the ones below, do not save as much time as the one above. */
            if ((doForces || doShiftForces)
                && (gmx::anyTrue(fIX != zero) || gmx::anyTrue(fIY != zero) || gmx::anyTrue(fIZ != zero)))
            {
                if (doForces)
                {
                    gmx::transposeScatterIncrU<3>(reinterpret_cast<real*>(f), preloadIi, fIX, fIY, fIZ);
                }
                if (doShiftForces)
                {
                    gmx::transposeScatterIncrU<3>(
                            reinterpret_cast<real*>(fshift), preloadIs, fIX, fIY, fIZ);
                }
            }
            if (doPotential)
            {
                int ggid = gid[n];
                if (gmx::anyTrue(vCTot != zero))
                {
#pragma omp atomic
                    Vc[ggid] += gmx::reduce(vCTot);
                }
                if (gmx::anyTrue(vVTot != zero))
                {
#pragma omp atomic
                    Vv[ggid] += gmx::reduce(vVTot);
                }
            }
        }
    } // end for (int n = 0; n < nri; n++)

    if (gmx::anyTrue(dvdlCoul != zero))
    {
#pragma omp atomic
        dvdl[efptCOUL] += gmx::reduce(dvdlCoul);
    }
    if (gmx::anyTrue(dvdlVdw != zero))
    {
#pragma omp atomic
        dvdl[efptVDW] += gmx::reduce(dvdlVdw);
    }

    /* Estimate flops, average for free energy stuff:
     * 12  flops per outer iteration
     * 150 flops per inner iteration
     */
#pragma omp atomic
    inc_nrnb(nrnb, eNR_NBKERNEL_FREE_ENERGY, nlist->nri * 12 + nlist->jindex[nri] * 150);
}

typedef void (*KernelFunction)(const t_nblist* gmx_restrict nlist,
                               rvec* gmx_restrict         xx,
                               gmx::ForceWithShiftForces* forceWithShiftForces,
                               const t_forcerec* gmx_restrict fr,
                               const t_mdatoms* gmx_restrict mdatoms,
                               nb_kernel_data_t* gmx_restrict kernel_data,
                               t_nrnb* gmx_restrict nrnb);

template<SoftcoreType softcoreType, bool scLambdasOrAlphasDiffer, bool vdwInteractionTypeIsEwald, bool elecInteractionTypeIsEwald, bool vdwModifierIsPotSwitch>
static KernelFunction dispatchKernelOnUseSimd(const bool useSimd)
{
    if (useSimd)
    {
#if GMX_SIMD_HAVE_REAL && GMX_SIMD_HAVE_INT32_ARITHMETICS && GMX_USE_SIMD_KERNELS
        return (nb_free_energy_kernel<SimdDataTypes, softcoreType, scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, vdwModifierIsPotSwitch>);
#else
        return (nb_free_energy_kernel<ScalarDataTypes, softcoreType, scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, vdwModifierIsPotSwitch>);
#endif
    }
    else
    {
        return (nb_free_energy_kernel<ScalarDataTypes, softcoreType, scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, vdwModifierIsPotSwitch>);
    }
}

template<SoftcoreType softcoreType, bool scLambdasOrAlphasDiffer, bool vdwInteractionTypeIsEwald, bool elecInteractionTypeIsEwald>
static KernelFunction dispatchKernelOnVdwModifier(const bool vdwModifierIsPotSwitch, const bool useSimd)
{
    if (vdwModifierIsPotSwitch)
    {
        return (dispatchKernelOnUseSimd<softcoreType, scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, true>(
                useSimd));
    }
    else
    {
        return (dispatchKernelOnUseSimd<softcoreType, scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, false>(
                useSimd));
    }
}

template<SoftcoreType softcoreType, bool scLambdasOrAlphasDiffer, bool vdwInteractionTypeIsEwald>
static KernelFunction dispatchKernelOnElecInteractionType(const bool elecInteractionTypeIsEwald,
                                                          const bool vdwModifierIsPotSwitch,
                                                          const bool useSimd)
{
    if (elecInteractionTypeIsEwald)
    {
        return (dispatchKernelOnVdwModifier<softcoreType, scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, true>(
                vdwModifierIsPotSwitch, useSimd));
    }
    else
    {
        return (dispatchKernelOnVdwModifier<softcoreType, scLambdasOrAlphasDiffer, vdwInteractionTypeIsEwald, false>(
                vdwModifierIsPotSwitch, useSimd));
    }
}

template<SoftcoreType softcoreType, bool scLambdasOrAlphasDiffer>
static KernelFunction dispatchKernelOnVdwInteractionType(const bool vdwInteractionTypeIsEwald,
                                                         const bool elecInteractionTypeIsEwald,
                                                         const bool vdwModifierIsPotSwitch,
                                                         const bool useSimd)
{
    if (vdwInteractionTypeIsEwald)
    {
        return (dispatchKernelOnElecInteractionType<softcoreType, scLambdasOrAlphasDiffer, true>(
                elecInteractionTypeIsEwald, vdwModifierIsPotSwitch, useSimd));
    }
    else
    {
        return (dispatchKernelOnElecInteractionType<softcoreType, scLambdasOrAlphasDiffer, false>(
                elecInteractionTypeIsEwald, vdwModifierIsPotSwitch, useSimd));
    }
}

template<SoftcoreType softcoreType>
static KernelFunction dispatchKernelOnScLambdasOrAlphasDifference(const bool scLambdasOrAlphasDiffer,
                                                                  const bool vdwInteractionTypeIsEwald,
                                                                  const bool elecInteractionTypeIsEwald,
                                                                  const bool vdwModifierIsPotSwitch,
                                                                  const bool useSimd)
{
    if (scLambdasOrAlphasDiffer)
    {
        return (dispatchKernelOnVdwInteractionType<softcoreType, true>(
                vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, vdwModifierIsPotSwitch, useSimd));
    }
    else
    {
        return (dispatchKernelOnVdwInteractionType<softcoreType, false>(
                vdwInteractionTypeIsEwald, elecInteractionTypeIsEwald, vdwModifierIsPotSwitch, useSimd));
    }
}

static KernelFunction dispatchKernel(const bool                 scLambdasOrAlphasDiffer,
                                     const bool                 vdwInteractionTypeIsEwald,
                                     const bool                 elecInteractionTypeIsEwald,
                                     const bool                 vdwModifierIsPotSwitch,
                                     const bool                 useSimd,
                                     const interaction_const_t& ic)
{
    if (ic.softCoreParameters->alphaCoulomb == 0 && ic.softCoreParameters->alphaVdw == 0)
    {
        return (dispatchKernelOnScLambdasOrAlphasDifference<SoftcoreType::None>(scLambdasOrAlphasDiffer,
                                                                                vdwInteractionTypeIsEwald,
                                                                                elecInteractionTypeIsEwald,
                                                                                vdwModifierIsPotSwitch,
                                                                                useSimd));
    }
    else
    {
        if (ic.softCoreParameters->softcoreType == SoftcoreType::Beutler)
        {
            return (dispatchKernelOnScLambdasOrAlphasDifference<SoftcoreType::Beutler>(scLambdasOrAlphasDiffer,
                                                                                       vdwInteractionTypeIsEwald,
                                                                                       elecInteractionTypeIsEwald,
                                                                                       vdwModifierIsPotSwitch,
                                                                                       useSimd));
        }
        else
        {
            return (dispatchKernelOnScLambdasOrAlphasDifference<SoftcoreType::Gapsys>(scLambdasOrAlphasDiffer,
                                                                                      vdwInteractionTypeIsEwald,
                                                                                      elecInteractionTypeIsEwald,
                                                                                      vdwModifierIsPotSwitch,
                                                                                      useSimd));
        }
    }
}


void gmx_nb_free_energy_kernel(const t_nblist*            nlist,
                               rvec*                      xx,
                               gmx::ForceWithShiftForces* ff,
                               const t_forcerec*          fr,
                               const t_mdatoms*           mdatoms,
                               nb_kernel_data_t*          kernel_data,
                               t_nrnb*                    nrnb)
{
    const interaction_const_t& ic = *fr->ic;
    GMX_ASSERT(EEL_PME_EWALD(ic.eeltype) || ic.eeltype == eelCUT || EEL_RF(ic.eeltype),
               "Unsupported eeltype with free energy");
    GMX_ASSERT(ic.softCoreParameters, "We need soft-core parameters");

    const auto& scParams                   = *ic.softCoreParameters;
    const bool  vdwInteractionTypeIsEwald  = (EVDW_PME(ic.vdwtype));
    const bool  elecInteractionTypeIsEwald = (EEL_PME_EWALD(ic.eeltype));
    const bool  vdwModifierIsPotSwitch     = (ic.vdw_modifier == eintmodPOTSWITCH);
    bool        scLambdasOrAlphasDiffer    = true;
    const bool  useSimd                    = fr->use_simd_kernels;

    if (scParams.alphaCoulomb == 0 && scParams.alphaVdw == 0)
    {
        scLambdasOrAlphasDiffer = false;
    }
    else
    {
        if (kernel_data->lambda[efptCOUL] == kernel_data->lambda[efptVDW]
            && scParams.alphaCoulomb == scParams.alphaVdw)
        {
            scLambdasOrAlphasDiffer = false;
        }
    }

    KernelFunction kernelFunc;
    kernelFunc = dispatchKernel(scLambdasOrAlphasDiffer,
                                vdwInteractionTypeIsEwald,
                                elecInteractionTypeIsEwald,
                                vdwModifierIsPotSwitch,
                                useSimd,
                                ic);
    kernelFunc(nlist, xx, ff, fr, mdatoms, kernel_data, nrnb);
}
