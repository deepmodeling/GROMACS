/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2016,2017 by the GROMACS development team.
 * Copyright (c) 2018,2019,2020, by the GROMACS development team, led by
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
 * Tests utilities for nonbonded fep kernel (softcore).
 *
 * \author Sebastian Kehl <sebastian.kehl@mpcdf.mpg.de>
 * \ingroup module_gmxlib
 */
#include "gmxpre.h"

#include <string>

#include <gtest/gtest.h>

#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "gromacs/math/units.h"

#include "gromacs/gmxlib/nonbonded/nb_softcore.h"

namespace
{

struct InputData
{
    std::vector<real> distance_ = { 0.1, 0.3};
    std::vector<real> lambda_   = { 0.0, 0.4};
    std::vector<real> alpha_    = { 0.35, 0.85, 1.0 };
};

struct ResultData
{
    struct ReactionField
    {
        std::vector<real> force_ = { 45.44971727401715,  7.9564798871433124, 5.618549907138807,
                                     53.141497616431444, 9.501381806342767,  6.753006140951377,
                                     82.40735912670932,  16.22643413780952,  10.553537362667253,
                                     90.99882133196249,  19.849951123589747, 13.335909759612157 };

        std::vector<real> potential_ = { 153.4963391711208,  54.74482632376834,  42.48368423849913,
                                         165.76344640639638, 61.831019988681454, 48.705535430478285,
                                         80.57750218820057,  41.379535057355184, 33.34728854380457,
                                         82.28900834597745,  45.71298780780877,  37.50722603632285 };

        std::vector<real> dvdl_      = {
            0.0, 0.0, 0.0, 16.257088585414177, 9.557942962344592, 8.404930863143283,
            0.0, 0.0, 0.0, 1.8495453371238524, 5.690973078257637, 5.503004144090047
        };
    };

    struct EwaldCoulomb
    {
        std::vector<real> force_     = { 46.14439456223906,  8.651157175365222,  6.313227195360717,
                                     53.836174904653355, 10.196059094564678, 7.447683429173287,
                                     88.6594547207065,   22.47852973180671,  16.805632956664443,
                                     97.25091692595967,  26.102046717586937, 19.588005353609347 };

        std::vector<real> potential_ = { 153.14900052700983, 54.39748767965739, 42.136345594388175,
                                         165.41610776228544, 61.4836813445705,  48.35819678636733,
                                         77.45145439120198,  38.25348726035659, 30.221240746805975,
                                         79.16296054897886,  42.58694001081018, 34.38117823932426 };

        std::vector<real> dvdl_ = {
            0.0, 0.0, 0.0, 16.257088585414177, 9.557942962344592, 8.404930863143283,
            0.0, 0.0, 0.0, 1.8495453371238524, 5.690973078257637, 5.503004144090047
        };
    };

    struct LennardJones
    {
        std::vector<real> force_     = {1592504.8443630151, 22.52700221389511, -0.3994048768740422,
            3768632.8377578724, 83.77558926399867, 5.030256018969582, 0.0, 8.57813764570534,
            -1.198214630622123, 0.0, 0.0, 2.265650434405245};
        std::vector<real> potential_ = {347006.9938992501, 25.324325293355464, -1.4866674448593462,
            614549.4305951666, 87.10756169609895, 5.416621259131128, 0.0, -0.062056135774753114,
            -0.687857691111263, 0.0, 0.0, -0.36885157130686763};
        std::vector<real> dvdl_      = {0.0, 0.0, 0.0, 401503.1459288585, 129.0495668164257,
            16.110255786271328, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5018934377109587};
    };

    ReactionField reactionField_;
    EwaldCoulomb ewald_;
    LennardJones lennardJones_;
};

static const InputData s_data;
static const ResultData s_results;

struct Constants
{
    // coulomb parameters (with q=0.5)
    real qq_             = ONE_4PI_EPS0 * 0.25_real;
    real potentialShift_ = 1.0_real;
    real forceShift_     = 1.0_real;
    real ewaldShift_     = 1.0_real;

    // lennard-jones parameters (with eps=0.5, sigma=0.3)
    real c12_             = 1.062882e-6_real * 12;
    real c6_              = 1.458e-3_real * 6;
    real sigma_           = 0.5_real * c12_ / c6_;
    real repulsionShift_  = 1.0_real;
    real dispersionShift_ = 1.0_real;

    // softcore parameters
    real dLambda_ = 1.0_real;
};


class SoftcoreGapsysTest :
    public ::testing::TestWithParam<std::tuple<real, real, real>>
{
public:
    SoftcoreGapsysTest() :
        tolerance_(gmx::test::relativeToleranceAsPrecisionDependentFloatingPoint(1, 1.0e-4, 1.0e-12)),
        facel_(ONE_4PI_EPS0)
    {
    }

protected:
    void SetUp() override
    {
        force_     = 0.0_real;
        potential_ = 0.0_real;
        dvdl_      = 0.0_real;

        // get test index
        std::string name = testing::UnitTest::GetInstance()->current_test_info()->name();
        std::size_t found = name.find_last_of("/");
        idx_ = std::stoi(name.substr(found+1));

        // get input parameters
        std::tie(r_, lambda_, alpha_) = GetParam();
        rsq_                          = r_ * r_;
        rInv_                         = rsq_ > 0 ? 1.0_real / r_ : 0.0_real;
    }

    void reactionField()
    {
        reactionFieldQuadraticPotential(params_.qq_, facel_, r_, lambda_, params_.dLambda_,
                                        alpha_, params_.forceShift_, params_.potentialShift_,
                                        &force_, &potential_, &dvdl_);
    }

    void ewaldCoulomb()
    {
        ewaldQuadraticPotential(params_.qq_, facel_, r_, lambda_, params_.dLambda_, alpha_,
                                params_.ewaldShift_, &force_, &potential_, &dvdl_);
    }

    void lennardJones()
    {
        lennardJonesQuadraticPotential(params_.c6_, params_.c12_, r_, rsq_, lambda_, params_.dLambda_,
                                       params_.sigma_, alpha_, params_.repulsionShift_,
                                       params_.dispersionShift_, &force_, &potential_, &dvdl_);
    }

    // test setup
    int idx_;
    gmx::test::FloatingPointTolerance tolerance_;

    // fixed test parameters
    Constants params_;
    const real facel_;

    // input data this test class is supposed to use
    real r_;
    real rInv_;
    real rsq_;
    real lambda_;
    real alpha_;

    // output values
    real force_;
    real potential_;
    real dvdl_;

};

TEST_P(SoftcoreGapsysTest, reactionField)
{
    reactionField();

    EXPECT_REAL_EQ_TOL(force_, s_results.reactionField_.force_[idx_], tolerance_);
    EXPECT_REAL_EQ_TOL(potential_, s_results.reactionField_.potential_[idx_], tolerance_);
    EXPECT_REAL_EQ_TOL(dvdl_, s_results.reactionField_.dvdl_[idx_], tolerance_);
}

TEST_P(SoftcoreGapsysTest, ewaldCoulomb)
{
    ewaldCoulomb();

    EXPECT_REAL_EQ_TOL(force_, s_results.ewald_.force_[idx_], tolerance_);
    EXPECT_REAL_EQ_TOL(potential_, s_results.ewald_.potential_[idx_], tolerance_);
    EXPECT_REAL_EQ_TOL(dvdl_, s_results.ewald_.dvdl_[idx_], tolerance_);
}

TEST_P(SoftcoreGapsysTest, lennardJones)
{
    lennardJones();

    EXPECT_REAL_EQ_TOL(force_, s_results.lennardJones_.force_[idx_], tolerance_);
    EXPECT_REAL_EQ_TOL(potential_, s_results.lennardJones_.potential_[idx_], tolerance_);
    EXPECT_REAL_EQ_TOL(dvdl_, s_results.lennardJones_.dvdl_[idx_], tolerance_);
}

INSTANTIATE_TEST_CASE_P(CheckValues, SoftcoreGapsysTest,
                        ::testing::Combine(::testing::ValuesIn(s_data.distance_),
                                           ::testing::ValuesIn(s_data.lambda_),

                                           ::testing::ValuesIn(s_data.alpha_)));

class SoftcoreGapsysEvalZeroTest : public SoftcoreGapsysTest
{};

TEST_P(SoftcoreGapsysEvalZeroTest, reactionField)
{
    reactionField();

    EXPECT_EQ(force_, 0.0);
    EXPECT_EQ(potential_, 0.0);
    EXPECT_EQ(dvdl_, 0.0);
}

TEST_P(SoftcoreGapsysEvalZeroTest, ewaldCoulomb)
{
    ewaldCoulomb();

    EXPECT_EQ(force_, 0.0);
    EXPECT_EQ(potential_, 0.0);
    EXPECT_EQ(dvdl_, 0.0);
}

TEST_P(SoftcoreGapsysEvalZeroTest, lennardJones)
{
    lennardJones();

    EXPECT_EQ(force_, 0.0);
    EXPECT_EQ(potential_, 0.0);
    EXPECT_EQ(dvdl_, 0.0);
}

INSTANTIATE_TEST_CASE_P(CheckZeros, SoftcoreGapsysEvalZeroTest,
                        ::testing::Values(std::make_tuple(0.1, 1.0, 0.35), std::make_tuple(0.1, 0.4, 0.0)));
} // namespace