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
        std::vector<real> force_     = {298.2262673248765, 81.04208675312732, 60.716019343653784, 320.9082469989308, 93.92455579266444, 70.70915256867373, 0.0, 108.52859863380814, 97.87991749360009, 0.0, 0.0, 104.55063947421152};
        std::vector<real> potential_ = {308.46161910449194, 201.37114113756252, 176.63680068879958, 311.34337586881185, 214.50646301939787, 189.50482873889663, 0.0, 84.15285483983247, 83.29414218061244, 0.0, 0.0, 83.94546301215242};
        std::vector<real> dvdl_      = {0.0, 0.0, 0.0, 2.6807752609604263, 17.141528411721787, 16.943058294331728, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5278382842267321};
    };

    struct EwaldCoulomb
    {
        std::vector<real> force_     = {298.9209446130984, 81.73676404134923, 61.410696631875695, 321.6029242871527, 94.61923308088635, 71.40382985689564, 0.0, 114.78069422780533, 104.13201308759727, 0.0, 0.0, 110.8027350682087};
        std::vector<real> potential_ = {308.114280460381, 201.02380249345157, 176.2894620446886, 310.9960372247009, 214.15912437528692, 189.15749009478566, 0.0, 81.02680704283388, 80.16809438361385, 0.0, 0.0, 80.81941521515382};
        std::vector<real> dvdl_      = {0.0, 0.0, 0.0, 2.6807752609604263, 17.141528411721787, 16.943058294331728, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5278382842267321};
    };

    struct LennardJones
    {
        std::vector<real> force_     = {1592504.8443630151, 22.52700221389511, -0.3994048768740422, 3768632.8377578724, 83.77558926399867, 5.030256018969582, 0.0, 8.57813764570534, -1.198214630622123, 0.0, 0.0, 2.265650434405245};
        std::vector<real> potential_ = {347006.9938992501, 25.324325293355464, -1.4866674448593462, 614549.4305951666, 87.10756169609895, 5.416621259131128, 0.0, -0.062056135774753114, -0.687857691111263, 0.0, 0.0, -0.36885157130686763};
        std::vector<real> dvdl_      = {0.0, 0.0, 0.0, 401503.1459288585, 129.0495668164257, 16.110255786271328, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5018934377109587};
    };

    ReactionField rf_;
    EwaldCoulomb ewald_;
    LennardJones lj_;
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


class SoftcoreGapsys :
    public ::testing::TestWithParam<std::tuple<real, real, real>>
{
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

    // test case index
    int idx_;

    // fixed test parameters
    Constants params_;

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

TEST_P(SoftcoreGapsys, reactionField)
{
    reactionFieldQuadraticPotential(params_.qq_, r_, lambda_, params_.dLambda_, params_.sigma_,
                                    alpha_, params_.forceShift_, params_.potentialShift_, &force_,
                                    &potential_, &dvdl_);

    gmx::test::FloatingPointTolerance tolerance(
            gmx::test::relativeToleranceAsPrecisionDependentFloatingPoint(1, 1.0e-4, 1.0e-12));

    EXPECT_REAL_EQ_TOL(force_, s_results.rf_.force_[idx_], tolerance);
    EXPECT_REAL_EQ_TOL(potential_, s_results.rf_.potential_[idx_], tolerance);
    EXPECT_REAL_EQ_TOL(dvdl_, s_results.rf_.dvdl_[idx_], tolerance);
}

TEST_P(SoftcoreGapsys, ewaldCoulomb)
{
    ewaldQuadraticPotential(params_.qq_, r_, lambda_, params_.dLambda_, params_.sigma_, alpha_,
                            params_.ewaldShift_, &force_, &potential_, &dvdl_);

    gmx::test::FloatingPointTolerance tolerance(
            gmx::test::relativeToleranceAsPrecisionDependentFloatingPoint(1, 1.0e-4, 1.0e-12));

    EXPECT_REAL_EQ_TOL(force_, s_results.ewald_.force_[idx_], tolerance);
    EXPECT_REAL_EQ_TOL(potential_, s_results.ewald_.potential_[idx_], tolerance);
    EXPECT_REAL_EQ_TOL(dvdl_, s_results.ewald_.dvdl_[idx_], tolerance);
}

TEST_P(SoftcoreGapsys, lennardJones)
{
    lennardJonesQuadraticPotential(params_.c6_, params_.c12_, r_, rsq_, lambda_, params_.dLambda_,
                                   params_.sigma_, alpha_, params_.repulsionShift_,
                                   params_.dispersionShift_, &force_, &potential_, &dvdl_);

    gmx::test::FloatingPointTolerance tolerance(
            gmx::test::relativeToleranceAsPrecisionDependentFloatingPoint(1, 1.0e-4, 1.0e-12));

    EXPECT_REAL_EQ_TOL(force_, s_results.lj_.force_[idx_], tolerance);
    EXPECT_REAL_EQ_TOL(potential_, s_results.lj_.potential_[idx_], tolerance);
    EXPECT_REAL_EQ_TOL(dvdl_, s_results.lj_.dvdl_[idx_], tolerance);
}

INSTANTIATE_TEST_CASE_P(CheckValues, SoftcoreGapsys,
                        ::testing::Combine(::testing::ValuesIn(s_data.distance_),
                                           ::testing::ValuesIn(s_data.lambda_),
                                           ::testing::ValuesIn(s_data.alpha_)));
} // namespace
