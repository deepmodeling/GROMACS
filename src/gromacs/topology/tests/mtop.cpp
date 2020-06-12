/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020, by the GROMACS development team, led by
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
 * \brief
 * Implements test of mtop routines
 *
 * \author Roland Schulz <roland.schulz@intel.com>
 * \ingroup module_topology
 */
#include "gmxpre.h"

#include <gtest/gtest.h>

#include "gromacs/topology/mtop_util.h"
#include "gromacs/utility/basedefinitions.h"

namespace gmx
{
namespace
{

/*! \brief Initializes a basic topology with 9 atoms with settle*/
void createBasicTop(gmx_mtop_t* mtop)
{
    gmx_moltype_t moltype;
    moltype.atoms.nr             = NRAL(F_SETTLE);
    std::vector<int>& iatoms     = moltype.ilist[F_SETTLE].iatoms;
    const int         settleType = 0;
    iatoms.push_back(settleType);
    iatoms.push_back(0);
    iatoms.push_back(1);
    iatoms.push_back(2);
    mtop->moltype.push_back(moltype);

    mtop->molblock.resize(1);
    mtop->molblock[0].type = 0;
    mtop->molblock[0].nmol = 3;
    mtop->natoms           = moltype.atoms.nr * mtop->molblock[0].nmol;
    gmx_mtop_finalize(mtop);
}

/*! \brief
 * Creates dummy topology with two differently sized residues.
 *
 * Residue begin and end are set to allow checking routines
 * that make use of the boundaries.
 *
 * \returns The residue ranges.
 */
std::vector<gmx::Range<int>> createTwoResidueTopology(gmx_mtop_t* mtop)
{
    auto& moltype        = mtop->moltype.emplace_back();
    int   residueOneSize = 5;
    int   residueTwoSize = 4;
    moltype.atoms.nr     = residueOneSize + residueTwoSize;
    snew(moltype.atoms.atom, residueOneSize + residueTwoSize);
    for (int i = 0; i < residueOneSize; i++)
    {
        moltype.atoms.atom[i].resind = 0;
    }
    for (int i = residueOneSize; i < residueOneSize + residueTwoSize; i++)
    {
        moltype.atoms.atom[i].resind = 1;
    }

    mtop->molblock.resize(1);
    mtop->molblock[0].type = 0;
    mtop->molblock[0].nmol = 1;
    mtop->natoms           = moltype.atoms.nr * mtop->molblock[0].nmol;
    gmx_mtop_finalize(mtop);
    std::vector<gmx::Range<int>> residueRange;
    residueRange.emplace_back(0, residueOneSize);
    residueRange.emplace_back(residueOneSize, residueOneSize + residueTwoSize);
    return residueRange;
}

TEST(MtopTest, RangeBasedLoop)
{
    gmx_mtop_t mtop;
    createBasicTop(&mtop);
    int count = 0;
    for (const AtomProxy atomP : AtomRange(mtop))
    {
        EXPECT_EQ(atomP.globalAtomNumber(), count);
        ++count;
    }
    EXPECT_EQ(count, 9);
}

TEST(MtopTest, Operators)
{
    gmx_mtop_t mtop;
    createBasicTop(&mtop);
    AtomIterator it(mtop);
    AtomIterator otherIt(mtop);
    EXPECT_EQ((*it).globalAtomNumber(), 0);
    EXPECT_EQ(it->globalAtomNumber(), 0);
    EXPECT_TRUE(it == otherIt);
    EXPECT_FALSE(it != otherIt);
    ++it;
    EXPECT_EQ(it->globalAtomNumber(), 1);
    it++;
    EXPECT_EQ(it->globalAtomNumber(), 2);
    EXPECT_TRUE(it != otherIt);
    EXPECT_FALSE(it == otherIt);
}

TEST(MtopTest, CanFindResidueStartAndEndAtoms)
{
    gmx_mtop_t mtop;
    auto       expectedResidueRange = createTwoResidueTopology(&mtop);

    auto atomRanges = atomRangeOfEachResidue(mtop.moltype[0]);
    ASSERT_EQ(atomRanges.size(), expectedResidueRange.size());
    for (gmx::index i = 0; i < gmx::ssize(atomRanges); i++)
    {
        EXPECT_EQ(atomRanges[i].begin(), expectedResidueRange[i].begin());
        EXPECT_EQ(atomRanges[i].end(), expectedResidueRange[i].end());
        ASSERT_EQ(atomRanges[i].size(), expectedResidueRange[i].size());
    }
    int rangeIndex = 0;
    for (const auto& range : atomRanges)
    {
        auto referenceRangeIt = expectedResidueRange[rangeIndex].begin();
        for (int i : range)
        {
            EXPECT_EQ(i, *referenceRangeIt);
            referenceRangeIt++;
        }
        rangeIndex++;
    }
}

} // namespace

} // namespace gmx
