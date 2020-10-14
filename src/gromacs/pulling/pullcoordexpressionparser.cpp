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

#include "pullcoordexpressionparser.h"

#include "config.h"

#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#include "pull_internal.h"

void PullCoordExpressionParser::setVariable(int variableIdx, double value, int nVariables)
{
#if HAVE_MUPARSER
    if (!parser_)
    {
        initializeParser(nVariables);
    }
    if (gmx::ssize(variableValues_) <= variableIdx)
    {
        GMX_THROW(gmx::InvalidInputError("Variable index out of range for the expression"));
    }
    variableValues_[variableIdx] = value;
#else
    GMX_UNUSED_VALUE(variableIdx);
    GMX_UNUSED_VALUE(value);
    GMX_UNUSED_VALUE(nVariables);
    GMX_RELEASE_ASSERT(false, "Can not use transformation pull coordinate without muparser");
#endif
}

double PullCoordExpressionParser::eval()
{
#if HAVE_MUPARSER
    if (!parser_)
    {
        GMX_THROW(gmx::InvalidInputError("Tried to evaluate an uninitialized expression."));
    }
    return parser_->Eval();
#else
    return 0;

#endif
}

void PullCoordExpressionParser::initializeParser(int nVariables)
{
#if HAVE_MUPARSER
    parser_ = std::make_unique<mu::Parser>();
    parser_->SetExpr(expression_);
    variableValues_.resize(nVariables);
    for (int n = 0; n < nVariables; n++)
    {
        variableValues_[n] = 0;
        std::string name   = "x" + std::to_string(n + 1);
        parser_->DefineVar(name, &variableValues_[n]);
    }
#else
    GMX_UNUSED_VALUE(nVariables);
    GMX_RELEASE_ASSERT(false, "Can not use transformation pull coordinate without muparser");
#endif
}

double getTransformationPullCoordinateValue(pull_t* pull, int transformationPullCoordinateIndex)
{
#if HAVE_MUPARSER
    double             result = 0;
    pull_coord_work_t* coord  = &pull->coord[transformationPullCoordinateIndex];
    int                variablePcrdIndex;
    try
    {
        for (variablePcrdIndex = 0; variablePcrdIndex < transformationPullCoordinateIndex;
             variablePcrdIndex++)
        {
            pull_coord_work_t* variablePcrd = &pull->coord[variablePcrdIndex];
            coord->expressionParser.setVariable(variablePcrdIndex, variablePcrd->spatialData.value,
                                                transformationPullCoordinateIndex);
        }
        result = coord->expressionParser.eval();
    }
    catch (mu::Parser::exception_type& e)
    {
        GMX_THROW(gmx::InternalError(gmx::formatString(
                "failed to evaluate expression for transformation pull-coord%d: %s\n",
                transformationPullCoordinateIndex + 1, e.GetMsg().c_str())));
    }
    catch (std::exception& e)
    {
        GMX_THROW(gmx::InternalError(gmx::formatString(
                "failed to evaluate expression for transformation pull-coord%d.\n"
                "Last variable pull-coord-index: %d.\n"
                "Message:  %s\n",
                transformationPullCoordinateIndex + 1, variablePcrdIndex + 1, e.what())));
    }
    return result;
#else
    GMX_UNUSED_VALUE(pull);
    GMX_UNUSED_VALUE(transformationPullCoordinateIndex);
    return 0;
#endif
}
