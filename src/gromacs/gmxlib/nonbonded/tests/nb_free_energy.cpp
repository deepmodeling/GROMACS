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

#include <gtest/gtest.h>

#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "gromacs/math/units.h"

#include "gromacs/gmxlib/nonbonded/nb_softcore.h"

namespace
{

struct InputData
{
    std::vector<double> distance_ = { 0.0, 0.1, 0.3};
    std::vector<double> lambda_   = { 0.0, 0.4, 1.0};
    std::vector<double> alpha_    = { 0.35, 0.85, 1.0 };
};

static InputData s_data;

template <class RealType>
struct Constants
{
    // coulomb parameters (with q=0.5)
    RealType qq_             = ONE_4PI_EPS0 * 0.25;
    RealType potentialShift_ = 1.0;
    RealType forceShift_     = 1.0;
    RealType ewaldShift_     = 1.0;

    // lennard-jones parameters (with eps=0.5, sigma=0.3)
    RealType c12_             = 1.062882e-6 * 12;
    RealType c6_              = 1.458e-3 * 6;
    RealType sigma_           = 0.5 * c12_ / c6_;
    RealType repulsionShift_  = 1.0;
    RealType dispersionShift_ = 1.0;

    // softcore parameters
    RealType dLambda_ = 1.0;
};

class NonbondedFepTest : public ::testing::Test
{
public:
    NonbondedFepTest() : checker_(data_.rootChecker()) {}

    gmx::test::TestReferenceData data_;
    gmx::test::TestReferenceChecker checker_;
};

template<class RealType>
class SoftcoreGapsys :
    public NonbondedFepTest,
    public ::testing::WithParamInterface<std::tuple<double, double, double>>
{
protected:
    void SetUp() override
    {
        force_     = 0.0;
        potential_ = 0.0;
        dvdl_      = 0.0;

        // get input parameters
        std::tie(r_, lambda_, alpha_) = GetParam();
        rsq_                          = r_ * r_;
        rInv_                         = rsq_ > 0 ? 1.0 / r_ : 0.0;

        // set up name for data checker
        name_ = "_r_" + std::to_string(r_) + "_lam_" + std::to_string(lambda_) + "_alp_"
                + std::to_string(alpha_);
    }

    // fixed test parameters
    Constants<RealType> params_;

    // input data this test class is supposed to use
    RealType r_;
    RealType rInv_;
    RealType rsq_;
    RealType lambda_;
    RealType alpha_;

    // output values
    RealType force_;
    RealType potential_;
    RealType dvdl_;
    std::string name_;

};

using SoftcoreGapsysReal = SoftcoreGapsys<real>;

TEST_P(SoftcoreGapsysReal, reactionField)
{
    reactionFieldQuadraticPotential(params_.qq_, r_, lambda_, params_.dLambda_, params_.sigma_,
                                    alpha_, params_.forceShift_, params_.potentialShift_, &force_,
                                    &potential_, &dvdl_);

    checker_.checkValue(force_, ("force" + name_).c_str());
    checker_.checkValue(potential_, ("potential" + name_).c_str());
    checker_.checkValue(dvdl_, ("dvdl" + name_).c_str());
}

TEST_P(SoftcoreGapsysReal, ewaldCoulomb)
{
    ewaldQuadraticPotential(params_.qq_, r_, lambda_, params_.dLambda_, params_.sigma_, alpha_,
                            params_.ewaldShift_, &force_, &potential_, &dvdl_);

    checker_.checkValue(force_, ("force" + name_).c_str());
    checker_.checkValue(potential_, ("potential" + name_).c_str());
    checker_.checkValue(dvdl_, ("dvdl" + name_).c_str());
}

TEST_P(SoftcoreGapsysReal, lennardJones)
{
    lennardJonesQuadraticPotential(params_.c6_, params_.c12_, r_, rsq_, lambda_, params_.dLambda_,
                                   params_.sigma_, alpha_, params_.repulsionShift_,
                                   params_.dispersionShift_, &force_, &potential_, &dvdl_);

    checker_.checkValue(force_, ("force" + name_).c_str());
    checker_.checkValue(potential_, ("potential" + name_).c_str());
    checker_.checkValue(dvdl_, ("dvdl" + name_).c_str());
}

INSTANTIATE_TEST_CASE_P(CheckValues, SoftcoreGapsysReal,
                        ::testing::Combine(::testing::ValuesIn(s_data.distance_),
                                           ::testing::ValuesIn(s_data.lambda_),
                                           ::testing::ValuesIn(s_data.alpha_)));

} // namespace
