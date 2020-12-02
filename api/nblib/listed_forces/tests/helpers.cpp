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
 * This implements basic nblib utility tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include "gmxpre.h"

#include <gtest/gtest.h>

#include "nblib/listed_forces/helpers.hpp"
#include "nblib/listed_forces/traits.h"

#include "testutils/testasserts.h"


namespace nblib
{
namespace test
{
namespace
{

TEST(NBlibTest, CanSplitListedWork)
{
    ListedInteractionData interactions;

    DefaultAngle     angle(Degrees(1), 1);
    HarmonicBondType bond(1, 1);

    int largestIndex = 20;
    int nSplits      = 3; // split ranges: [0,5], [6,11], [12, 19]

    std::vector<InteractionIndex<HarmonicBondType>> bondIndices{
        { 0, 1, 0 }, { 0, 6, 0 }, { 11, 12, 0 }, { 18, 19, 0 }
    };
    std::vector<InteractionIndex<DefaultAngle>> angleIndices{
        { 0, 1, 2, 0 }, { 0, 6, 7, 0 }, { 11, 12, 13, 0 }, { 17, 19, 18, 0 }
    };

    pickType<HarmonicBondType>(interactions).indices = bondIndices;
    pickType<DefaultAngle>(interactions).indices     = angleIndices;

    std::vector<ListedInteractionData> splitInteractions =
            splitListedWork(interactions, largestIndex, nSplits);

    std::vector<InteractionIndex<HarmonicBondType>> refBondIndices0{ { 0, 1, 0 }, { 0, 6, 0 } };
    std::vector<InteractionIndex<DefaultAngle>> refAngleIndices0{ { 0, 1, 2, 0 }, { 0, 6, 7, 0 } };
    std::vector<InteractionIndex<HarmonicBondType>> refBondIndices1{ { 11, 12, 0 } };
    std::vector<InteractionIndex<DefaultAngle>>     refAngleIndices1{ { 11, 12, 13, 0 } };
    std::vector<InteractionIndex<HarmonicBondType>> refBondIndices2{ { 18, 19, 0 } };
    std::vector<InteractionIndex<DefaultAngle>>     refAngleIndices2{ { 17, 19, 18, 0 } };

    EXPECT_EQ(refBondIndices0, pickType<HarmonicBondType>(splitInteractions[0]).indices);
    EXPECT_EQ(refBondIndices1, pickType<HarmonicBondType>(splitInteractions[1]).indices);
    EXPECT_EQ(refBondIndices2, pickType<HarmonicBondType>(splitInteractions[2]).indices);

    EXPECT_EQ(refAngleIndices0, pickType<DefaultAngle>(splitInteractions[0]).indices);
    EXPECT_EQ(refAngleIndices1, pickType<DefaultAngle>(splitInteractions[1]).indices);
    EXPECT_EQ(refAngleIndices2, pickType<DefaultAngle>(splitInteractions[2]).indices);
}


TEST(NBlibTest, ListedForceBuffer)
{
    using T     = gmx::RVec;
    int ncoords = 20;

    T              vzero{ 0, 0, 0 };
    std::vector<T> masterBuffer(ncoords, vzero);

    // the ForceBuffer is going to access indices [10-15) through the masterBuffer
    // and the outliers internally
    int rangeStart = 10;
    int rangeEnd   = 15;

    ForceBuffer<T> forceBuffer(masterBuffer.data(), rangeStart, rangeEnd);

    // in range
    T internal1{ 1, 2, 3 };
    T internal2{ 4, 5, 6 };
    forceBuffer[10] = internal1;
    forceBuffer[14] = internal2;

    // outliers
    T outlier{ 0, 1, 2 };
    forceBuffer[5] = outlier;

    std::vector<T> refMasterBuffer(ncoords, vzero);
    refMasterBuffer[10] = internal1;
    refMasterBuffer[14] = internal2;

    for (size_t i = 0; i < masterBuffer.size(); ++i)
    {
        for (size_t m = 0; m < dimSize; ++m)
        {
            EXPECT_REAL_EQ_TOL(refMasterBuffer[i][m], masterBuffer[i][m],
                               gmx::test::defaultRealTolerance());
        }
    }

    for (size_t m = 0; m < dimSize; ++m)
    {
        EXPECT_REAL_EQ_TOL(outlier[m], forceBuffer[5][m], gmx::test::defaultRealTolerance());
        EXPECT_REAL_EQ_TOL(vzero[m], forceBuffer[4][m], gmx::test::defaultRealTolerance());
    }
}

} // namespace
} // namespace test
} // namespace nblib
