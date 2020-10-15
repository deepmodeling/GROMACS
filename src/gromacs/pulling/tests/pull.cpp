/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2018,2019,2020, by the GROMACS development team, led by
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
 * Implements test of some pulling routines
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_pulling
 */
#include "gmxpre.h"

#include "gromacs/pulling/pull.h"

#include "config.h"

#include <cmath>

#include <algorithm>

#include <gtest/gtest.h>

#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pulling/pull_internal.h"
#include "gromacs/utility/smalloc.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"

namespace gmx
{
namespace
{

using gmx::test::defaultRealTolerance;

class PullTest : public ::testing::Test
{
protected:
    PullTest() {}

    static void test(PbcType pbcType, matrix box)
    {
        t_pbc pbc;

        // PBC stuff
        set_pbc(&pbc, pbcType, box);

        GMX_ASSERT(pbc.ndim_ePBC >= 1 && pbc.ndim_ePBC <= DIM,
                   "Tests only support PBC along at least x and at most x, y, and z");

        real boxSizeZSquared;
        if (pbc.ndim_ePBC > ZZ)
        {
            boxSizeZSquared = gmx::square(box[ZZ][ZZ]);
        }
        else
        {
            boxSizeZSquared = GMX_REAL_MAX;
        }

        {
            // Distance pulling in all 3 dimensions
            t_pull_coord params;
            params.eGeom   = epullgDIST;
            params.dim[XX] = 1;
            params.dim[YY] = 1;
            params.dim[ZZ] = 1;
            pull_coord_work_t pcrd(params, 0);
            clear_dvec(pcrd.spatialData.vec);

            real minBoxSize2 = GMX_REAL_MAX;
            for (int d = 0; d < pbc.ndim_ePBC; d++)
            {
                minBoxSize2 = std::min(minBoxSize2, norm2(box[d]));
            }
            EXPECT_REAL_EQ_TOL(0.25 * minBoxSize2, max_pull_distance2(&pcrd, &pbc),
                               defaultRealTolerance());
        }

        {
            // Distance pulling along Z
            t_pull_coord params;
            params.eGeom   = epullgDIST;
            params.dim[XX] = 0;
            params.dim[YY] = 0;
            params.dim[ZZ] = 1;
            pull_coord_work_t pcrd(params, 0);
            clear_dvec(pcrd.spatialData.vec);
            EXPECT_REAL_EQ_TOL(0.25 * boxSizeZSquared, max_pull_distance2(&pcrd, &pbc),
                               defaultRealTolerance());
        }

        {
            // Directional pulling along Z
            t_pull_coord params;
            params.eGeom   = epullgDIR;
            params.dim[XX] = 1;
            params.dim[YY] = 1;
            params.dim[ZZ] = 1;
            pull_coord_work_t pcrd(params, 0);
            clear_dvec(pcrd.spatialData.vec);
            pcrd.spatialData.vec[ZZ] = 1;
            EXPECT_REAL_EQ_TOL(0.25 * boxSizeZSquared, max_pull_distance2(&pcrd, &pbc),
                               defaultRealTolerance());
        }

        {
            // Directional pulling along X
            t_pull_coord params;
            params.eGeom   = epullgDIR;
            params.dim[XX] = 1;
            params.dim[YY] = 1;
            params.dim[ZZ] = 1;
            pull_coord_work_t pcrd(params, 0);
            clear_dvec(pcrd.spatialData.vec);
            pcrd.spatialData.vec[XX] = 1;

            real minDist2 = square(box[XX][XX]);
            for (int d = XX + 1; d < DIM; d++)
            {
                minDist2 -= square(box[d][XX]);
            }
            EXPECT_REAL_EQ_TOL(0.25 * minDist2, max_pull_distance2(&pcrd, &pbc), defaultRealTolerance());
        }
    }
};

TEST_F(PullTest, MaxPullDistanceXyzScrewBox)
{
    matrix box = { { 10, 0, 0 }, { 0, 10, 0 }, { 0, 0, 10 } };

    test(PbcType::Screw, box);
}

TEST_F(PullTest, MaxPullDistanceXyzCubicBox)
{
    matrix box = { { 10, 0, 0 }, { 0, 10, 0 }, { 0, 0, 10 } };

    test(PbcType::Xyz, box);
}

TEST_F(PullTest, MaxPullDistanceXyzTricBox)
{
    matrix box = { { 10, 0, 0 }, { 3, 10, 0 }, { 3, 4, 10 } };

    test(PbcType::Xyz, box);
}

TEST_F(PullTest, MaxPullDistanceXyzLongBox)
{
    matrix box = { { 10, 0, 0 }, { 0, 10, 0 }, { 0, 0, 30 } };

    test(PbcType::Xyz, box);
}

TEST_F(PullTest, MaxPullDistanceXySkewedBox)
{
    matrix box = { { 10, 0, 0 }, { 5, 8, 0 }, { 0, 0, 0 } };

    test(PbcType::XY, box);
}

#if HAVE_MUPARSER
TEST_F(PullTest, TransformationCoord)
{
    t_pbc pbc;

    // PBC stuff
    matrix box = { { 10, 0, 0 }, { 0, 10, 0 }, { 0, 0, 10 } };
    set_pbc(&pbc, PbcType::Xyz, box);

    pull_t pull;

    // Create standard pull coordinate
    t_pull_coord params;
    params.eGeom = epullgDIST;
    pull.coord.emplace_back(params, 0);

    // Create transformation pull coordinate
    params.eGeom           = epullgTRANSFORMATION;
    std::string expression = "x1^2 + 3";
    params.expression      = expression;
    pull.coord.emplace_back(params, 1);

    for (double v = 0; v < 10; v += 1)
    {
        // transformation pull coord value
        pull.coord[0].spatialData.value = v;
        pull.coord[1].spatialData.value = getTransformationPullCoordinateValue(
                &pull.coord[1], constArrayRefFromArray(pull.coord.data(), 1));
        // Since we perform numerical differentiation and floating point operations
        // we only expect the results below to be approximately equal
        double expected = v * v + 3;
        EXPECT_REAL_EQ_TOL(pull.coord[1].spatialData.value, expected, defaultRealTolerance());

        // force and derivative
        double transformationForce = v + 0.5;
        pull.coord[1].scalarForce  = transformationForce;
        double variableForce       = computeForceFromTransformationPullCoord(&pull, 1, 0);
        expected                   = 2 * v * transformationForce;
        double finiteDiffInputSize = square(v + c_pullTransformationCoordinateDifferentationEpsilon) + 3;
        EXPECT_REAL_EQ_TOL(variableForce, expected,
                           test::relativeToleranceAsFloatingPoint(
                                   finiteDiffInputSize,
                                   1e-15 / c_pullTransformationCoordinateDifferentationEpsilon));
    }
}
#endif // HAVE_MUPARSER

} // namespace

} // namespace gmx
