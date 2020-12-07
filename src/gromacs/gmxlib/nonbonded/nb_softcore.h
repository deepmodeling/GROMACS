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

/* linearized electrostatics */
template<class RealType>
static inline void quadraticApproximationCoulomb(const RealType qq,
                                                 const RealType rInvQ,
                                                 const RealType r,
                                                 const RealType lambdaFac,
                                                 const RealType dLambdaFac,
                                                 RealType*      force,
                                                 RealType*      potential,
                                                 RealType*      dvdl)
{
    RealType quadrFac, linFac, constFac;
    constFac = qq * rInvQ;
    linFac   = constFac * r * rInvQ;
    quadrFac = linFac * r * rInvQ;

    /* Computing Coulomb force and potential energy */
    *force = -2. * quadrFac + 3. * linFac;

    *potential = quadrFac - 3. * (linFac - constFac);

    *dvdl += dLambdaFac * 0.5 * (lambdaFac / (1. - lambdaFac)) * (quadrFac - 2. * linFac + constFac);
}

/* reaction-field linearized electrostatics */
template<class RealType>
static inline void reactionFieldQuadraticPotential(const RealType qq,
                                                   const RealType r,
                                                   const RealType lambdaFac,
                                                   const RealType dLambdaFac,
                                                   const RealType sigma6,
                                                   const RealType alphaEff,
                                                   const RealType krf,
                                                   const RealType potentialShift,
                                                   RealType*      force,
                                                   RealType*      potential,
                                                   RealType*      dvdl)
{
    /* check if we have to use the hardcore values */
    if ((lambdaFac < 1) && (alphaEff > 0))
    {
        constexpr RealType c_twentySixSeventh = 26.0 / 7.0;
        RealType           rQ;

        rQ = gmx::sixthroot(c_twentySixSeventh * sigma6 * (1.- lambdaFac));
        rQ *= alphaEff;

        if (r < rQ)
        {
            RealType rInvQ = 1.0/rQ;
            quadraticApproximationCoulomb(qq, rInvQ, r, lambdaFac, dLambdaFac, force, potential, dvdl);
            *force -= (qq * 2.0 * krf * r * r);
            *potential += (qq * (krf * r * r - potentialShift));
        }
    }
}

/* ewald linearized electrostatics */
template<class RealType>
static inline void ewaldQuadraticPotential(const RealType qq,
                                           const RealType r,
                                           const RealType lambdaFac,
                                           const RealType dLambdaFac,
                                           const RealType sigma6,
                                           const RealType alphaEff,
                                           const RealType potentialShift,
                                           RealType*      force,
                                           RealType*      potential,
                                           RealType*      dvdl)
{

    /* check if we have to use the hardcore values */
    if ((lambdaFac < 1) && (alphaEff > 0))
    {
        constexpr RealType c_twentySixSeventh = 26.0 / 7.0;
        RealType           rQ;

        rQ = gmx::sixthroot(c_twentySixSeventh * sigma6 * (1.- lambdaFac));
        rQ *= alphaEff;

        if (r < rQ)
        {
            RealType rInvQ = 1.0/rQ;
            quadraticApproximationCoulomb(qq, rInvQ, r, lambdaFac, dLambdaFac, force, potential, dvdl);

            *potential -= qq*potentialShift;
        }
    }
}

/* cutoff LJ with quadratic appximation of lj-potential */
template<class RealType>
static inline void lennardJonesQuadraticPotential(const RealType c6,
                                                  const RealType c12,
                                                  const RealType r,
                                                  const RealType rsq,
                                                  const RealType lambdaFac,
                                                  const RealType dLambdaFac,
                                                  const RealType sigma6,
                                                  const RealType alphaEff,
                                                  const RealType repulsionShift,
                                                  const RealType dispersionShift,
                                                  RealType*      force,
                                                  RealType*      potential,
                                                  RealType*      dvdl)
{
    constexpr RealType c_twentySixSeventh = 26.0 / 7.0;
    constexpr RealType c_oneSixth         = 1.0 / 6.0;
    constexpr RealType c_oneTwelth        = 1.0 / 12.0;

    /* check if we have to use the hardcore values */
    if ((lambdaFac < 1) && (alphaEff > 0))
    {
        RealType rQ;
        rQ = gmx::sixthroot(c_twentySixSeventh * sigma6 * (1.- lambdaFac));
        rQ *= alphaEff;

        // scaled values for c6 and c12
        RealType c6s, c12s;
        c6s  = c_oneSixth * c6;
        c12s = c_oneTwelth * c12;

        if (r < rQ)
        {
            /* Temporary variables for inverted values */
            RealType rInvQ = 1.0 / rQ;
            RealType rInv14C, rInv13C, rInv12C;
            RealType rInv8C, rInv7C, rInv6C;
            rInv6C = rInvQ * rInvQ * rInvQ;
            rInv6C *= rInv6C;
            rInv7C  = rInv6C * rInvQ;
            rInv8C  = rInv7C * rInvQ;
            rInv14C = c12s * rInv7C * rInv7C * rsq;
            rInv13C = c12s * rInv7C * rInv6C * r;
            rInv12C = c12s * rInv6C * rInv6C;
            rInv8C *= c6s * rsq;
            rInv7C *= c6s * r;
            rInv6C *= c6s;

            /* Temporary variables for A and B */
            RealType quadrFac, linearFac, constFac;
            quadrFac  = 156. * rInv14C - 42. * rInv8C;
            linearFac = 168. * rInv13C - 48. * rInv7C;
            constFac  = 91. * rInv12C - 28. * rInv6C;

            /* Computing LJ force and potential energy*/
            *force = -quadrFac + linearFac;

            *potential = 0.5 * quadrFac - linearFac + constFac;

            *dvdl += dLambdaFac * 28. * (lambdaFac / (1. - lambdaFac))
                        * ((6.5 * rInv14C - rInv8C) - (13. * rInv13C - 2. * rInv7C)
                           + (6.5 * rInv12C - rInv6C));

            *potential += ((c12s * repulsionShift) - (c6s * dispersionShift));
        }

    }
}

#endif
