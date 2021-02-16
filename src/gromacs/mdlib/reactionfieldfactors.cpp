/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2017,2018 by the GROMACS development team.
 * Copyright (c) 2019,2020,2021, by the GROMACS development team, led by
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

#include "reactionfieldfactors.h"

#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/md_enums.h"

namespace gmx
{

ReactionFieldCoefficients::ReactionFieldCoefficients(real dielectric,
                                                     real reactionFieldDielectric,
                                                     real rCoulomb,
                                                     bool useReactionField,
                                                     int  coulombModifier)
{
    if (useReactionField)
    {
        real rCoulombCubed = std::pow(rCoulomb, 3);
        dielectric_        = reactionFieldDielectric;

        /* eps == 0 signals infinite dielectric */
        if (reactionFieldDielectric == 0)
        {
            constant_ = 1 / (2 * rCoulombCubed);
        }
        else
        {
            constant_ = (reactionFieldDielectric - dielectric)
                        / (2 * reactionFieldDielectric + dielectric) / rCoulombCubed;
        }
        correction_ = 1 / rCoulomb + constant_ * rCoulomb * rCoulomb;
    }
    else
    {
        /* For plain cut-off we might use the reaction-field kernels */
        dielectric_ = dielectric;
        constant_   = 0;
        if (coulombModifier == eintmodPOTSHIFT)
        {
            correction_ = 1 / rCoulomb;
        }
        else
        {
            correction_ = 0;
        }
    }
}

void ReactionFieldCoefficients::ReactionFieldLog(FILE* fplog, real rCoulomb, real dielectric)
{
    fprintf(fplog,
            "%s:\n"
            "epsRF = %g, rc = %g, krf = %g, crf = %g, epsfac = %g\n",
            eel_names[eelRF],
            dielectric_,
            rCoulomb,
            constant_,
            correction_,
            ONE_4PI_EPS0 / dielectric);
    // Make sure we don't lose resolution in pow() by casting real arg to double
    real rmin = gmx::invcbrt(static_cast<double>(constant_ * 2.0));
    fprintf(fplog, "The electrostatics potential has its minimum at r = %g\n", rmin);
}

ReactionFieldCoefficients::~ReactionFieldCoefficients() = default;

} // namespace gmx
