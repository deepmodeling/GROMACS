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

/*! \libinternal \file
 *
 * \brief
 * This file contains the definitions of classes and structs for user supplied VdW tables
 *
 * \author Berk Hess
 *
 * \inlibraryapi
 * \ingroup module_mdtypes
 */

#ifndef GMX_MDTYPES_USERVDWTABLES_H
#define GMX_MDTYPES_USERVDWTABLES_H

#include <memory>
#include <vector>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

class ISerializer;

/*! \libinternal
 * \brief User supplied Van der Waals tables for disperion and repulsion
 *
 * For both dispersion and repulsion the potential is tabulated and optionally
 * the derivative. All tables have the same size, except that both derivatives
 * are empty when \p haveDerivativeTables = false.
 */
class UserVdwTable
{

public:
    /*! \brief Constructor
     * \param[in] spacing  Spacing between table points
     * \param[in] dispersionPotential  Potential for dispersion
     * \param[in] dispersionDerivative  Derivative for dispersion, so have same size as dispersionPotential or be empty
     * \param[in] repulsionPotential  Potential for repulsion, should have same size as dispersionPotential
     * \param[in] repulsionDerivative  Derivative for repulsion, so have same size as repulsionPotential or be empty
     */
    UserVdwTable(const double           spacing,
                 ArrayRef<const double> dispersionPotential,
                 ArrayRef<const double> dispersionDerivative,
                 ArrayRef<const double> repulsionPotential,
                 ArrayRef<const double> repulsionDerivative);

    //! Spacing between consecutive table points
    const double spacing;
    //! Whether we have dispersionDerivative and repulsionDerivative
    const bool haveDerivativeTables;
    //! The dispersion potential table
    const std::vector<double> dispersionPotential;
    //! The dispersion derivative table, empty with haveDerivativeTables=true
    const std::vector<double> dispersionDerivative;
    //! The repulsion potential table
    const std::vector<double> repulsionPotential;
    //! The repulsion derivative table, empty with haveDerivativeTables=true
    const std::vector<double> repulsionDerivative;
};

/*! \libinternal
 * \brief Holds a UserVdwTable for an energy group pair
 */
struct UserVdwEnergyGroupPairTable
{
    //! Energy group index of the first group in the energy group pair
    int energyGroupI;
    //! Energy group index of the second group in the energy group pair
    int energyGroupJ;
    //! The VdW tables
    UserVdwTable table;
};

/*! \libinternal
 * \brief Holds all the user Van der Waals tables for the system
 */
struct UserVdwTableCollection
{
    //! The default table used for energy group pairs without specific pair table, can be empty
    std::unique_ptr<UserVdwTable> defaultTable;
    //! List of energy group pair tables
    std::vector<UserVdwEnergyGroupPairTable> energyGroupPairTables;
};

//! Serializes a UserVdwTableCollection
void serializeUserVdwTableCollection(ISerializer* serializer, UserVdwTableCollection* userVdwTableCollection);

} // namespace gmx

#endif // GMX_MDTYPES_USERVDWTABLES_H
