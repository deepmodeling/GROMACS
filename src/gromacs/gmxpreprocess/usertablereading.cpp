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

#include "usertablereading.h"

#include "gromacs/fileio/filetypes.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/uservdwtables.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

int checkDistancesAndReturnNumPoints(ArrayRef<const double> distances,
                                     const real             vdwCutoffDistance,
                                     const std::string&     filename)
{
    if (distances.size() < 2)
    {
        GMX_THROW(InvalidInputError(
                formatString("Need at least two points in file '%s'", filename.c_str())));
    }
    if (distances[0] != 0)
    {
        GMX_THROW(InvalidInputError(formatString(
                "The table in file '%s' should start at distance 0", filename.c_str())));
    }

    int    numPoints    = 0;
    double prevDistance = -1;
    for (const double distance : distances)
    {
        if (distance <= prevDistance)
        {
            GMX_THROW(InvalidInputError(formatString(
                    "The distances in the table file '%s' should be increasing", filename.c_str())));
        }
        if (distance <= vdwCutoffDistance * (1 + GMX_REAL_EPS))
        {
            numPoints++;
        }
    }

    if (numPoints < 2)
    {
        GMX_THROW(InvalidInputError(formatString(
                "Found less than 2 distances within the cutoff distance %g in table file '%s'",
                vdwCutoffDistance, filename.c_str())));
    }

    return numPoints;
}

UserVdwTable readUserVdwTable(const std::string& filename, const double vdwCutoffDistance)
{
    MultiDimArray<std::vector<double>, dynamicExtents2D> xvgData    = readXvgData(filename);
    const int                                            numColumns = xvgData.extent(0);

    if (numColumns != 7)
    {
        GMX_THROW(InvalidInputError(formatString("Expected %d columns in file '%s', but found %d",
                                                 7, filename.c_str(), numColumns)));
    }

    auto columns = xvgData.asView();

    const int numPoints = checkDistancesAndReturnNumPoints(
            constArrayRefFromArray(columns[0].data(), xvgData.extent(1)), vdwCutoffDistance, filename);

    const double spacing = (columns[0][numPoints - 1] - columns[0][0]) / (numPoints - 1);

    // Change signs of the derivative as the user supplies the negative of the derivative
    for (int i = 0; i < numPoints; i++)
    {
        columns[4][i] = -columns[4][i];
        columns[6][i] = -columns[6][i];
    }

    return UserVdwTable(spacing, constArrayRefFromArray(columns[3].data(), numPoints),
                        constArrayRefFromArray(columns[4].data(), numPoints),
                        constArrayRefFromArray(columns[5].data(), numPoints),
                        constArrayRefFromArray(columns[6].data(), numPoints));
}

std::vector<EnergyGroupPairTablesData> getEnergyGroupPairTablesFilenames(int numNonbondedEnergyGroupPairs,
                                                                         int* energyGroupPairFlags,
                                                                         int  numEnergyGroups,
                                                                         const SimulationGroups& groups,
                                                                         const std::string& tableBaseFilename)
{
    std::vector<EnergyGroupPairTablesData> energyGroupPairTablesData;
    for (int egi = 0; egi < numNonbondedEnergyGroupPairs; egi++)
    {
        for (int egj = egi; egj < numNonbondedEnergyGroupPairs; egj++)
        {
            const int egpFlags = energyGroupPairFlags[GID(egi, egj, numEnergyGroups)];
            if (egpFlags & EGP_TABLE)
            {
                const std::string filename =
                        tableBaseFilename + "_"
                        + *groups.groupNames[groups.groupNumbers[SimulationAtomGroupType::EnergyOutput][egi]]
                        + "_"
                        + *groups.groupNames[groups.groupNumbers[SimulationAtomGroupType::EnergyOutput][egj]];
                energyGroupPairTablesData.push_back(EnergyGroupPairTablesData{ egi, egj, filename });
            }
        }
    }
    return energyGroupPairTablesData;
}

UserVdwTableCollectionBuilder::UserVdwTableCollectionBuilder(const t_inputrec&  ir,
                                                             const std::string& tableBaseFilename,
                                                             const SimulationGroups& groups) :
    vdwCutoffDistance_(ir.rvdw),
    numNonbondedEnergyGroupPairs_(ir.opts.ngener - ir.nwall),
    defaultTableFilename_(tableBaseFilename + "." + ftp2ext(efXVG))
{
    energyGroupPairTablesData_ = getEnergyGroupPairTablesFilenames(
            numNonbondedEnergyGroupPairs_, ir.opts.egp_flags, ir.opts.ngener, groups, tableBaseFilename);
}

UserVdwTableCollection UserVdwTableCollectionBuilder::build()
{
    UserVdwTableCollection tableCollection;
    // If not all energy group pairs use a table, we need the default table
    if (int(energyGroupPairTablesData_.size())
        < ((numNonbondedEnergyGroupPairs_ + 1) * numNonbondedEnergyGroupPairs_) / 2)
    {
        tableCollection.defaultTable = std::make_unique<UserVdwTable>(
                readUserVdwTable(defaultTableFilename_, vdwCutoffDistance_));
    }

    for (const auto& tablesData : energyGroupPairTablesData_)
    {
        tableCollection.energyGroupPairTables.push_back(
                { tablesData.EnergyGroupPairsI, tablesData.EnergyGroupPairsJ,
                  readUserVdwTable(tablesData.EnergyGroupPairsFilename, vdwCutoffDistance_) });
    }
    return tableCollection;
}

} // namespace gmx
