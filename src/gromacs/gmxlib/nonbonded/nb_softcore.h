/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2016,2017 by the GROMACS development team.
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
#ifndef GMX_GMXLIB_NONBONDED_SOFTCORE_H
#define GMX_GMXLIB_NONBONDED_SOFTCORE_H

#include "config.h"

#include "gromacs/math/functions.h"
#include "gromacs/simd/simd.h"
#include "gromacs/simd/simd_math.h"

/* linearized electrostatics */
template<class RealType, class BoolType>
static inline void quadraticApproximationCoulomb(const RealType qq,
                                                 const RealType rInvQ,
                                                 const RealType r,
                                                 const real lambdaFac,
                                                 const real dLambdaFac,
                                                 RealType*      force,
                                                 RealType*      potential,
                                                 RealType*      dvdl,
                                                 BoolType       mask)
{
    RealType quadrFac, linFac, constFac;
    constFac = qq * rInvQ;
    linFac   = constFac * r * rInvQ;
    quadrFac = linFac * r * rInvQ;

    /* Computing Coulomb force and potential energy */
    *force = -2. * quadrFac + 3. * linFac;

    *potential = quadrFac - 3. * (linFac - constFac);

    *dvdl = gmx::selectByMask(
            dLambdaFac * 0.5 * (lambdaFac / (1. - lambdaFac)) * (quadrFac - 2. * linFac + constFac), mask);
}

/* reaction-field linearized electrostatics */
template<class RealType, class BoolType>
static inline void reactionFieldQuadraticPotential(const RealType qq,
                                                   const RealType r,
                                                   const real lambdaFac,
                                                   const real dLambdaFac,
                                                   const RealType sigma6,
                                                   const RealType alphaEff,
                                                   const real krf,
                                                   const real potentialShift,
                                                   RealType*      force,
                                                   RealType*      potential,
                                                   RealType*      dvdl,
                                                   BoolType       mask)
{
    /* check if we have to use the hardcore values */
    BoolType computeValues = mask && (lambdaFac < 1 && 0 < alphaEff);
    if (gmx::anyTrue(computeValues))
    {
        constexpr real c_twentySixSeventh = 26.0 / 7.0;
        RealType       rQ;

        RealType lambdaFacRev = gmx::selectByMask(1.0 - lambdaFac, computeValues);

        rQ = gmx::cbrt(c_twentySixSeventh * sigma6 * lambdaFacRev);
        rQ = gmx::sqrt(rQ);
        rQ = rQ * alphaEff;

        computeValues = (computeValues && r < rQ);
        if (gmx::anyTrue(computeValues))
        {
            RealType rInvQ, forceOut, potentialOut, dvdlOut;

            rInvQ = gmx::maskzInv(rQ, computeValues);
            quadraticApproximationCoulomb(
                    qq, rInvQ, r, lambdaFac, dLambdaFac, &forceOut, &potentialOut, &dvdlOut, computeValues);

            *force     = gmx::selectByMask(forceOut - (qq * 2.0 * krf * r * r), computeValues);
            *potential = gmx::selectByMask(potentialOut + (qq * (krf * r * r - potentialShift)),
                                           computeValues);
            *dvdl = *dvdl + dvdlOut;
        }
    }
}

/* ewald linearized electrostatics */
template<class RealType, class BoolType>
static inline void ewaldQuadraticPotential(const RealType qq,
                                           const RealType r,
                                           const real lambdaFac,
                                           const real dLambdaFac,
                                           const RealType sigma6,
                                           const RealType alphaEff,
                                           const real potentialShift,
                                           RealType*      force,
                                           RealType*      potential,
                                           RealType*      dvdl,
                                           BoolType       mask)
{

    /* check if we have to use the hardcore values */
    BoolType computeValues = mask && (lambdaFac < 1 && 0 < alphaEff);
    if (gmx::anyTrue(computeValues))
    {
        constexpr real c_twentySixSeventh = 26.0 / 7.0;
        RealType       rQ;

        RealType lambdaFacRev = gmx::selectByMask(1.0 - lambdaFac, computeValues);

        rQ = gmx::cbrt(c_twentySixSeventh * sigma6 * lambdaFacRev);
        rQ = gmx::sqrt(rQ);
        rQ = rQ * alphaEff;

        computeValues = (computeValues && r < rQ);
        if (gmx::anyTrue(computeValues))
        {
            RealType rInvQ, forceOut, potentialOut, dvdlOut;

            rInvQ = gmx::maskzInv(rQ, computeValues);
            quadraticApproximationCoulomb(
                    qq, rInvQ, r, lambdaFac, dLambdaFac, &forceOut, &potentialOut, &dvdlOut, computeValues);

            *force     = gmx::selectByMask(forceOut, computeValues);
            *potential = gmx::selectByMask(potentialOut - (qq * potentialShift), computeValues);
            *dvdl = *dvdl + gmx::selectByMask(dvdlOut, computeValues);
        }
    }


}

/* cutoff LJ with quadratic appximation of lj-potential */
template<class RealType, class BoolType>
static inline void lennardJonesQuadraticPotential(const RealType c6,
                                                  const RealType c12,
                                                  const RealType r,
                                                  const RealType rsq,
                                                  const real lambdaFac,
                                                  const real dLambdaFac,
                                                  const RealType sigma6,
                                                  const RealType alphaEff,
                                                  const real repulsionShift,
                                                  const real dispersionShift,
                                                  RealType*      force,
                                                  RealType*      potential,
                                                  RealType*      dvdl,
                                                  BoolType       mask)
{
    constexpr real c_twentySixSeventh = 26.0 / 7.0;
    constexpr real c_oneSixth         = 1.0 / 6.0;
    constexpr real c_oneTwelth        = 1.0 / 12.0;

    /* check if we have to use the hardcore values */
    BoolType computeValues = mask && (lambdaFac < 1 && 0 < alphaEff);
    if (gmx::anyTrue(computeValues))
    {
        RealType       rQ;

        RealType lambdaFacRev = gmx::selectByMask(1.0 - lambdaFac, computeValues);

        rQ = gmx::cbrt(c_twentySixSeventh * sigma6 * lambdaFacRev);
        rQ = gmx::sqrt(rQ);
        rQ = rQ * alphaEff;

        computeValues = (computeValues && r < rQ);
        if (gmx::anyTrue(computeValues))
        {
            // scaled values for c6 and c12
            RealType c6s, c12s;
            c6s  = c_oneSixth * c6;
            c12s = c_oneTwelth * c12;
            /* Temporary variables for inverted values */
            RealType rInvQ = gmx::maskzInv(rQ, computeValues);
            RealType rInv14C, rInv13C, rInv12C;
            RealType rInv8C, rInv7C, rInv6C;
            rInv6C  = rInvQ * rInvQ * rInvQ;
            rInv6C  = rInv6C * rInv6C;
            rInv7C  = rInv6C * rInvQ;
            rInv8C  = rInv7C * rInvQ;
            rInv14C = c12s * rInv7C * rInv7C * rsq;
            rInv13C = c12s * rInv7C * rInv6C * r;
            rInv12C = c12s * rInv6C * rInv6C;
            rInv8C  = rInv8C * c6s * rsq;
            rInv7C  = rInv7C * c6s * r;
            rInv6C  = rInv6C * c6s;

            /* Temporary variables for A and B */
            RealType quadrFac, linearFac, constFac;
            quadrFac  = 156. * rInv14C - 42. * rInv8C;
            linearFac = 168. * rInv13C - 48. * rInv7C;
            constFac  = 91. * rInv12C - 28. * rInv6C;

            /* Computing LJ force and potential energy*/
            *force = gmx::selectByMask(-quadrFac + linearFac, computeValues);

            *potential = gmx::selectByMask(0.5 * quadrFac - linearFac + constFac, computeValues);

            *dvdl = *dvdl + gmx::selectByMask(dLambdaFac * 28. * (lambdaFac / (1. - lambdaFac))
                        * ((6.5 * rInv14C - rInv8C) - (13. * rInv13C - 2. * rInv7C)
                           + (6.5 * rInv12C - rInv6C)), computeValues);

            *potential = *potential + gmx::selectByMask(((c12s * repulsionShift) - (c6s * dispersionShift)), computeValues);
        }
    }
}

#endif
