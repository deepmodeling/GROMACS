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

#include "uservdwtables.h"

#include "gromacs/utility/inmemoryserializer.h"

namespace gmx
{

UserVdwTable::UserVdwTable(const double           spacing,
                           ArrayRef<const double> dispersionPotential,
                           ArrayRef<const double> dispersionDerivative,
                           ArrayRef<const double> repulsionPotential,
                           ArrayRef<const double> repulsionDerivative) :
    spacing(spacing),
    haveDerivativeTables(!dispersionDerivative.empty()),
    dispersionPotential(dispersionPotential.begin(), dispersionPotential.end()),
    dispersionDerivative(dispersionDerivative.begin(), dispersionDerivative.end()),
    repulsionPotential(repulsionPotential.begin(), repulsionPotential.end()),
    repulsionDerivative(repulsionDerivative.begin(), repulsionDerivative.end())
{
    GMX_RELEASE_ASSERT(repulsionPotential.size() == dispersionPotential.size(),
                       "Tables should have equal size");
    if (haveDerivativeTables)
    {
        GMX_RELEASE_ASSERT(dispersionDerivative.size() == dispersionPotential.size(),
                           "Tables should have equal size");
        GMX_RELEASE_ASSERT(repulsionDerivative.size() == dispersionPotential.size(),
                           "Tables should have equal size");
    }
    else
    {
        GMX_RELEASE_ASSERT(repulsionDerivative.empty(),
                           "Should either have both or no derivative tables");
    }
}

namespace
{

std::vector<double> readDoubleVector(ISerializer* serializer)
{
    int numElements;
    serializer->doInt(&numElements);
    std::vector<double> v(numElements);
    serializer->doDoubleArray(v.data(), numElements);

    return v;
}

void writeDoubleArray(ISerializer* serializer, ArrayRef<const double> v)
{
    int numElements = v.size();
    serializer->doInt(&numElements);
    serializer->doDoubleArray(const_cast<double*>(v.data()), numElements);
}

UserVdwTable readSerializeUserVdwTable(ISerializer* serializer)
{
    double spacing;
    serializer->doDouble(&spacing);

    std::vector<double> dispersionPotential  = readDoubleVector(serializer);
    std::vector<double> dispersionDerivative = readDoubleVector(serializer);
    std::vector<double> repulsionPotential   = readDoubleVector(serializer);
    std::vector<double> repulsionDerivative  = readDoubleVector(serializer);

    return UserVdwTable(spacing, dispersionPotential, dispersionDerivative, repulsionPotential,
                        repulsionDerivative);
}

void writeSerializeUserVdwTable(ISerializer* serializer, const UserVdwTable& userVdwTable)
{
    serializer->doDouble(const_cast<double*>(&userVdwTable.spacing));
    writeDoubleArray(serializer, userVdwTable.dispersionPotential);
    writeDoubleArray(serializer, userVdwTable.dispersionDerivative);
    writeDoubleArray(serializer, userVdwTable.repulsionPotential);
    writeDoubleArray(serializer, userVdwTable.repulsionDerivative);
}

} // namespace

void serializeUserVdwTableCollection(ISerializer* serializer, UserVdwTableCollection* userVdwTableCollection)
{
    bool haveDefaultTable = (userVdwTableCollection->defaultTable != nullptr);
    serializer->doBool(&haveDefaultTable);
    if (haveDefaultTable)
    {
        if (serializer->reading())
        {
            userVdwTableCollection->defaultTable =
                    std::make_unique<UserVdwTable>(readSerializeUserVdwTable(serializer));
        }
        else
        {
            writeSerializeUserVdwTable(serializer, *userVdwTableCollection->defaultTable);
        }
    }

    int numEnerGroupPairTables = userVdwTableCollection->energyGroupPairTables.size();
    serializer->doInt(&numEnerGroupPairTables);
    for (int i = 0; i < numEnerGroupPairTables; i++)
    {
        if (serializer->reading())
        {
            int egi;
            serializer->doInt(&egi);
            int egj;
            serializer->doInt(&egj);
            userVdwTableCollection->energyGroupPairTables.push_back(
                    { egi, egj, readSerializeUserVdwTable(serializer) });
        }
        else
        {
            const auto& energyGroupPairTable = userVdwTableCollection->energyGroupPairTables[i];
            serializer->doInt(const_cast<int*>(&energyGroupPairTable.energyGroupI));
            serializer->doInt(const_cast<int*>(&energyGroupPairTable.energyGroupJ));
            writeSerializeUserVdwTable(serializer, energyGroupPairTable.table);
        }
    }
}

} // namespace gmx
