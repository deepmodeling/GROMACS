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
 * Tests for frameconverter method to remove pbc jumps in coordinate frames.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */

#include "gmxpre.h"

#include <numeric>

#include "gromacs/coordinateio/frameconverters/removejump.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/trajectory/trajectoryframe.h"

#include "gromacs/coordinateio/tests/frameconverter.h"

namespace gmx
{

namespace test
{

/*! \brief
 * Test fixture to prepare coordinate frame for coordinate manipulation.
 */
class RemoveJumpTest : public FrameconverterTestBase
{
public:
    RemoveJumpTest()
    {
        RVec value(1, 2, 3);
        RVec inc(1, 1, 1);
        reference_.resize(frame()->natoms);
        for (int i = 0; i < frame()->natoms; i++)
        {
            copy_rvec(value, x()[i]);
            clear_rvec(reference_[i]);
            rvec_dec(reference_[i], value);
            rvec_inc(value, inc);
        }
    }
    /*! \brief
     *  Run the test.
     *
     *  \param[in] reference Pointer to the reference coordiantes to use.
     *  \param[in] box A box to test for removing PBC jumps.
     */
    void runTest(gmx::ArrayRef<const RVec> reference, matrix box);
    //! Access to underlying reference coordinates.
    const gmx::ArrayRef<const RVec> reference() { return reference_; }

private:
    //! Reference coordinates to use.
    std::vector<RVec> reference_;
};

void RemoveJumpTest::runTest(const gmx::ArrayRef<const RVec> reference, matrix box)
{
    method()->addFrameConverter(std::make_unique<RemoveJump>(reference, box));
    setNewFrame(method()->prepareAndTransformCoordinates(frame()));
}

TEST_F(RemoveJumpTest, Works)
{
    EXPECT_EQ(frame()->x, x());

    matrix box;
    clear_mat(box);
    for (int d1 = 0; d1 < DIM; d1++)
    {
        box[d1][d1] = 1;
    }
    runTest(reference(), box);
    std::vector<RVec> expectedFinalVector;
    RVec              endVal(0, -1, -2);
    for (int i = 0; i < frame()->natoms; i++)
    {
        compareRVec(false, newFrame()->x[i], frame()->x[i]);
        expectedFinalVector.emplace_back(endVal);
        rvec_dec(endVal, RVec(1, 1, 1));
        compareRVec(true, newFrame()->x[i], expectedFinalVector[i]);
    }
}

} // namespace test

} // namespace gmx
