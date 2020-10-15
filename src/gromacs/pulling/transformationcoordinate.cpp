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
#include "gmxpre.h"

#include "transformationcoordinate.h"

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#include "pull_internal.h"
#include "pullcoordexpressionparser.h"

namespace gmx
{

namespace
{

//! Calculates the value a for transformation pull coordinate
double getTransformationPullCoordinateValue(pull_coord_work_t* coord)
{
    const int transformationPullCoordinateIndex = coord->coordIndex;
    GMX_ASSERT(ssize(coord->transformationVariables) == transformationPullCoordinateIndex,
               "We need as many variables as the transformation pull coordinate index");
    double result = 0;
    try
    {
        result = coord->expressionParser.evaluate(coord->transformationVariables);
    }
    catch (mu::Parser::exception_type& e)
    {
        GMX_THROW(InternalError(
                formatString("failed to evaluate expression for transformation pull-coord%d: %s\n",
                             transformationPullCoordinateIndex + 1, e.GetMsg().c_str())));
    }
    catch (std::exception& e)
    {
        GMX_THROW(InternalError(formatString(
                "failed to evaluate expression for transformation pull-coord%d.\n"
                "Last variable pull-coord-index: %d.\n"
                "Message:  %s\n",
                transformationPullCoordinateIndex + 1, transformationPullCoordinateIndex + 1, e.what())));
    }
    return result;
}

} // namespace

double getTransformationPullCoordinateValue(pull_coord_work_t*                coord,
                                            ArrayRef<const pull_coord_work_t> variableCoords)
{
    GMX_ASSERT(ssize(variableCoords) == coord->coordIndex,
               "We need as many variables as the transformation pull coordinate index");
    int coordIndex = 0;
    for (const auto& variableCoord : variableCoords)
    {
        coord->transformationVariables[coordIndex++] = variableCoord.spatialData.value;
    }

    return getTransformationPullCoordinateValue(coord);
}

double computeForceFromTransformationPullCoord(pull_coord_work_t* coord, const int variablePcrdIndex)
{
    GMX_ASSERT(variablePcrdIndex >= 0 && variablePcrdIndex < coord->coordIndex,
               "The variable index should be in range of the transformation coordinate");

    // epsilon for numerical differentiation.
    const double epsilon                 = c_pullTransformationCoordinateDifferentationEpsilon;
    const double transformationPcrdValue = coord->spatialData.value;
    // Perform numerical differentiation of 1st order
    const double valueBackup = coord->transformationVariables[variablePcrdIndex];
    coord->transformationVariables[variablePcrdIndex] += epsilon;
    double transformationPcrdValueEps = getTransformationPullCoordinateValue(coord);
    double derivative = (transformationPcrdValueEps - transformationPcrdValue) / epsilon;
    // reset pull coordinate value
    coord->transformationVariables[variablePcrdIndex] = valueBackup;
    double result                                     = coord->scalarForce * derivative;
    if (debug)
    {
        fprintf(debug,
                "Distributing force %4.4f for transformation coordinate %d to coordinate %d with "
                "force "
                "%4.4f\n",
                coord->scalarForce, coord->coordIndex, variablePcrdIndex, result);
    }
    return result;
}

} // namespace gmx
