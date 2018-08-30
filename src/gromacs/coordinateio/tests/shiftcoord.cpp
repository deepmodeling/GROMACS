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
/*!\file
 * \internal
 * \brief
 * Tests for frameconverter coordinate shift method.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */

#include "gmxpre.h"

#include <numeric>

#include "gromacs/coordinateio/frameconverters/register.h"
#include "gromacs/coordinateio/frameconverters/shiftcoord.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/trajectory/trajectoryframe.h"

#include "gromacs/coordinateio/tests/frameconverter.h"

namespace gmx
{

namespace test
{

/*! \brief
 * Helper to compare result of changes to RVec.
 *
 * \param[in] expectedOutcome If the inputs should match or not.
 * \param[in] first One RVec to compare.
 * \param[in] second Other RVec to compare.
 */
static void compareRVec(bool expectedOutcome, const RVec& first, const RVec& second)
{
    for (int d = 0; d < DIM; d++)
    {
        EXPECT_EQ(expectedOutcome, first[d] == second[d]);
    }
}

/*! \brief
 * Test fixture to prepare coordinate frame for coordinate manipulation.
 */
class ShiftCoordTest : public FrameconverterTestBase
{
public:
    ShiftCoordTest()
    {
        RVec value(1, 2, 3);
        RVec inc(1, 1, 1);
        for (int i = 0; i < frame()->natoms; i++)
        {
            copy_rvec(value, x()[i]);
            rvec_inc(value, inc);
        }
    }
    /*! \brief
     *  Run the test.
     *
     *  \param[in] shift How to shift coordinates.
     *  \param[in] addSelection if a selection should be used for shift.
     */
    void runTest(RVec shift, bool addSelection);
    //! Get access to selection.
    const Selection& selection() { return sel_; }

private:
    //! Selection to use for tests.
    Selection sel_;
};

void ShiftCoordTest::runTest(const RVec shift, bool addSelection)
{
    if (!addSelection)
    {
        method()->addFrameConverter(std::make_unique<ShiftCoord>(shift, selection()));
        setNewFrame(method()->prepareAndTransformCoordinates(frame()));
    }
}

TEST_F(ShiftCoordTest, AllAtomShiftWorks)
{
    EXPECT_EQ(frame()->x, x());

    RVec shift(-1, -2, -3);
    runTest(shift, false);
    for (int i = 0; i < frame()->natoms; i++)
    {
        compareRVec(false, newFrame()->x[i], frame()->x[i]);
    }
    std::vector<RVec> expectedFinalVector;
    for (int i = 0; i < frame()->natoms; i++)
    {
        expectedFinalVector.emplace_back();
        rvec_add(frame()->x[i], shift, expectedFinalVector.back());
    }

    for (int i = 0; i < frame()->natoms; i++)
    {
        compareRVec(true, newFrame()->x[i], expectedFinalVector[i]);
    }
}

} // namespace test

} // namespace gmx
