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
 * \brief Tests for virial calculation.
 *
 * \author Joe Jordan <ejjordan@kth.se>
 */
#include "gmxpre.h"

#include "gromacs/mdlib/calcvir.h"

#include <gtest/gtest.h>

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{


class CalcvirTest : public ::testing::Test
{
public:
    TestReferenceData      refData_;
    TestReferenceChecker   checker_;
    std::vector<gmx::RVec> coordinates_;
    std::vector<gmx::RVec> forces_;
    int                    numVirialAtoms_;
    tensor                 virial_;

    CalcvirTest() :
        checker_(refData_.rootChecker()),
        coordinates_{ { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } },
        forces_{ { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } },
        numVirialAtoms_(coordinates_.size())
    {
    }

private:
};

TEST_F(CalcvirTest, CanCalculateVirial)
{

    const matrix box = { { 4, 0, 0 }, { 0, 4, 0 }, { 0, 0, 4 } };
    ;
    calc_vir(numVirialAtoms_, as_rvec_array(coordinates_.data()), as_rvec_array(forces_.data()), virial_, false, box);

    checker_.checkVector(virial_[0], "Virial x");
    checker_.checkVector(virial_[1], "Virial y");
    checker_.checkVector(virial_[2], "Virial z");
}

TEST_F(CalcvirTest, CanCalculateVirialWithPbc)
{

    const matrix box = { { 2.5, 0, 0 }, { 0, 2.5, 0 }, { 0, 0, 2.5 } };
    ;
    calc_vir(numVirialAtoms_, as_rvec_array(coordinates_.data()), as_rvec_array(forces_.data()), virial_, false, box);

    checker_.checkVector(virial_[0], "Virial x");
    checker_.checkVector(virial_[1], "Virial y");
    checker_.checkVector(virial_[2], "Virial z");
}

} // namespace
} // namespace test
} // namespace gmx
