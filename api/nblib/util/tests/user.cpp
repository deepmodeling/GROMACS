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
#include <vector>

#include "nblib/tests/testhelpers.h"
#include "nblib/util/user.h"

#include "testutils/testasserts.h"


namespace nblib
{
namespace test
{
namespace
{

TEST(NBlibTest, isRealValued)
{
    std::vector<Vec3> vec;
    vec.emplace_back(1., 1., 1.);
    vec.emplace_back(2., 2., 2.);

    bool ret = isRealValued(vec);
    EXPECT_EQ(ret, true);
}

TEST(NBlibTest, checkNumericValuesHasNan)
{
    std::vector<Vec3> vec;
    vec.emplace_back(1., 1., 1.);
    vec.emplace_back(2., 2., 2.);

    vec.emplace_back(NAN, NAN, NAN);

    bool ret = isRealValued(vec);
    EXPECT_EQ(ret, false);
}

TEST(NBlibTest, checkNumericValuesHasInf)
{
    std::vector<Vec3> vec;
    vec.emplace_back(1., 1., 1.);
    vec.emplace_back(2., 2., 2.);

    vec.emplace_back(INFINITY, INFINITY, INFINITY);

    bool ret = isRealValued(vec);
    EXPECT_EQ(ret, false);
}


TEST(NBlibTest, GeneratedVelocitiesAreCorrect)
{
    constexpr size_t  N = 10;
    std::vector<real> masses(N, 1.0);
    std::vector<Vec3> velocities;
    velocities = generateVelocity(300.0, 1, masses);

    Vector3DTest velocitiesTest;
    velocitiesTest.testVectors(velocities, "generated-velocities");
}
TEST(NBlibTest, generateVelocitySize)
{
    constexpr int     N = 10;
    std::vector<real> masses(N, 1.0);
    auto              out = generateVelocity(300.0, 1, masses);
    EXPECT_EQ(out.size(), N);
}

TEST(NBlibTest, generateVelocityCheckNumbers)
{
    constexpr int     N = 10;
    std::vector<real> masses(N, 1.0);
    auto              out = generateVelocity(300.0, 1, masses);
    bool              ret = isRealValued(out);
    EXPECT_EQ(ret, true);
}

} // namespace
} // namespace test
} // namespace nblib
