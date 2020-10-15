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
 *
 *
 * \brief
 * Contains classes and methods related to use of MuParser in pulling
 *
 * \author Oliver Fleetwood <oliver.fleetwood@gmail.com>
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \author Joe Jordan <ejjordan@kth.se>
 *
 */
#ifndef GMX_PULL_PULLCOORDEXPRESSIONPARSER_H
#define GMX_PULL_PULLCOORDEXPRESSIONPARSER_H

#include "config.h"

#include <memory>
#include <string>
#include <vector>

#if HAVE_MUPARSER
#    include <muParser.h>
#else
namespace mu
{
//! Defines a dummy Parser type to reduce use of the preprocessor.
using Parser = std::false_type;
} // namespace mu
#endif

struct pull_coord_work_t;

namespace gmx
{
template<typename>
class ArrayRef;

/*! \brief Class with a mathematical expression and parser.
 * \internal
 *
 * The class handles parser instantiation from an mathematical expression, e.g. 'x1*x2',
 * and evaluates the expression given the variables' numerical values.
 *
 * Note that for performance reasons you should not create a new PullCoordExpressionParser
 * for every evaluation.
 * Instead, instantiate one PullCoordExpressionParser per expression,
 * then update the variables before the next evaluation.
 *
 */
class PullCoordExpressionParser
{
public:
    //! Constructor which takes a mathematical expression as argument.
    PullCoordExpressionParser(const std::string& expression) : expression_(expression) {}

    //! Evaluates the expression with the numerical values passed in \p variables.
    double evaluate(ArrayRef<const double> variables);

private:
    /*! \brief
     * Prepares the expressionparser to bind muParser to n_variables.
     *
     * There's a performance gain by doing it this way since muParser will convert the expression
     * to bytecode the first time it's initialized. Then the subsequent evaluations are much faster.
     */
    void initializeParser(int numVariables);

    /*! \brief The mathematical expression, e.g. 'x1*x2' */
    std::string expression_;

    /*! \brief A vector containing the numerical values of the variables before parser evaluation.
     *
     * muParser compiles the expression to bytecode, then binds to the memory address
     * of these vector elements, making the evaluations fast and memory efficient.
     * */
    std::vector<double> variableValues_;

    /*! \brief The parser_ which compiles and evaluates the mathematical expression */
    std::unique_ptr<mu::Parser> parser_;
};

/*! \brief Calculates pull->coord[coord_ind].spatialData.value for transformation pull coordinates
 *
 * This requires the values of the pull coordinates of lower indices to be set
 * \param[in] coord  The (transformation) coordinate to compute the value for
 * \param[in] variableCoords  Pull coordinates used as variables, entries 0 to coord->coordIndex
 * will be used \returns Transformation value for pull coordinate.
 */
double getTransformationPullCoordinateValue(pull_coord_work_t*                coord,
                                            ArrayRef<const pull_coord_work_t> variableCoords);

} // namespace gmx

#endif // GMX_PULL_PULLCOORDEXPRESSIONPARSER_H
