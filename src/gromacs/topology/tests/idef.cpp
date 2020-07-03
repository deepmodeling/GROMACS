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
 * \brief
 * Implements test of InteractionList routines
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_topology
 */
#include "gmxpre.h"

#include "gromacs/topology/idef.h"

#include <array>

#include <gtest/gtest.h>

namespace gmx
{
namespace
{

TEST(InteractionListTest, EmptyWorks)
{
    InteractionList ilist(F_ANGLES);
    EXPECT_TRUE(ilist.empty());
    EXPECT_EQ(ilist.rawIndices().size(), 0);
}

TEST(InteractionListTest, CanAddInteractionArray)
{
    InteractionList    ilist(F_FBPOSRES);
    int                parameterType = 0;
    std::array<int, 1> atomList      = { 1 };
    ilist.push_back(parameterType, atomList);
    EXPECT_FALSE(ilist.empty());
    EXPECT_EQ(ilist.rawIndices().size(), 2);
    EXPECT_EQ(ilist.rawIndices()[0], parameterType);
    EXPECT_EQ(ilist.rawIndices()[1], 1);
}

TEST(InteractionListTest, CanAddInteractionArrayMultipleAtoms)
{
    InteractionList    ilist(F_ANGLES);
    int                parameterType = 0;
    std::array<int, 3> atomList      = { 1, 2, 3 };
    ilist.push_back(parameterType, atomList);
    EXPECT_FALSE(ilist.empty());
    EXPECT_EQ(ilist.rawIndices().size(), 4);
    EXPECT_EQ(ilist.rawIndices()[0], parameterType);
    EXPECT_EQ(ilist.rawIndices()[1], 1);
    EXPECT_EQ(ilist.rawIndices()[2], 2);
    EXPECT_EQ(ilist.rawIndices()[3], 3);
}

TEST(InteractionListTest, CanAddInteractionPointer)
{
    InteractionList    ilist(F_FBPOSRES);
    int                parameterType  = 0;
    std::array<int, 1> singleAtomList = { 1 };
    ilist.push_back(parameterType, singleAtomList.size(), singleAtomList.data());
    EXPECT_FALSE(ilist.empty());
    EXPECT_EQ(ilist.rawIndices().size(), 2);
    EXPECT_EQ(ilist.rawIndices()[0], parameterType);
    EXPECT_EQ(ilist.rawIndices()[1], 1);
}

TEST(InteractionListTest, CanAddListToOtherList)
{
    InteractionList firstList(F_FBPOSRES);
    int             firstParameterType = 0;
    {
        std::array<int, 1> singleAtomList = { 1 };
        firstList.push_back(firstParameterType, singleAtomList);
        EXPECT_FALSE(firstList.empty());
        EXPECT_EQ(firstList.rawIndices().size(), 2);
        EXPECT_EQ(firstList.rawIndices()[0], firstParameterType);
        EXPECT_EQ(firstList.rawIndices()[1], 1);
    }
    InteractionList secondList(F_FBPOSRES);
    int             secondParameterType = 2;
    {
        std::array<int, 1> atomList = { 3 };
        secondList.push_back(secondParameterType, atomList);
        EXPECT_FALSE(secondList.empty());
        EXPECT_EQ(secondList.rawIndices().size(), 2);
        EXPECT_EQ(secondList.rawIndices()[0], secondParameterType);
        EXPECT_EQ(secondList.rawIndices()[1], 3);
    }
    firstList.append(secondList);
    EXPECT_EQ(firstList.rawIndices().size(), 4);
    EXPECT_EQ(firstList.rawIndices()[2], secondParameterType);
    EXPECT_EQ(firstList.rawIndices()[3], 3);
}

TEST(InteractionListTest, ClearingWorks)
{
    InteractionList    ilist(F_CONSTR);
    int                parameterType = 0;
    std::array<int, 2> atomList      = { 1, 2 };
    ilist.push_back(parameterType, atomList);
    EXPECT_FALSE(ilist.empty());
    EXPECT_EQ(ilist.rawIndices().size(), 3);
    EXPECT_EQ(ilist.rawIndices()[0], parameterType);
    EXPECT_EQ(ilist.rawIndices()[1], 1);
    EXPECT_EQ(ilist.rawIndices()[2], 2);
    ilist.clear();
    EXPECT_TRUE(ilist.empty());
    EXPECT_EQ(ilist.rawIndices().size(), 0);
}

} // namespace

} // namespace gmx
