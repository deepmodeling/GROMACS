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

#include "testutils/testasserts.h"
#include "gromacs/simd/simd.h"

#include "gromacs/gmxlib/nonbonded/nb_softcore.h"
#include "data.h"

namespace
{

//! Scalar (non-SIMD) data types.
struct ScalarDataTypes
{
    using RealType = real; //!< The data type to use as real.
    using BoolType = bool; //!< The data type to use as boolean
    static constexpr int simdRealWidth = 1; //!< The width of the RealType.
};

#if GMX_SIMD_HAVE_REAL
//! SIMD data types.
struct SimdDataTypes
{
    using RealType = gmx::SimdReal;  //!< The data type to use as real.
    using BoolType = gmx::SimdBool;  //!< The data type to use as boolean
    static constexpr int simdRealWidth = GMX_SIMD_REAL_WIDTH; //!< The width of the RealType.
};
#endif

class InputDataGenerator
{
public:
    InputDataGenerator(InputData input) :
      distance_(input.distance_),
      lambda_(input.lambda_),
      alpha_(input.alpha_) {}

    std::vector<std::tuple<real, real, real, int>> getData() const
    {
        size_t numTuples = distance_.size() * lambda_.size() * alpha_.size();

        std::vector<std::tuple<real, real, real, int>> data;
        real distance, lambda, alpha;
        for (size_t i=0; i<numTuples; i++)
        {
            std::tie(distance, lambda, alpha) = getTuple(i);
            data.push_back(std::make_tuple(distance, lambda, alpha, i));
        }

        return data;
    }

protected:
    std::tuple<real, real, real> getTuple(int i) const
    {
        int idxDistance =  i / (lambda_.size() * alpha_.size());
        int idxLambda   =  i / alpha_.size();
        int idxAlpha    =  i % alpha_.size();

        // wrap around
        idxDistance = idxDistance % distance_.size();
        idxLambda   = idxLambda % lambda_.size();
        return std::make_tuple(distance_[idxDistance],
                               lambda_[idxLambda],
                               alpha_[idxAlpha]);
    }

    std::vector<real> distance_;
    std::vector<real> lambda_;
    std::vector<real> alpha_;
};

static const InputData s_input;
static const ResultData s_results;
static const InputDataGenerator s_data(s_input);

template<class DataType>
class SoftcoreGapsys
{
    using RealType = typename DataType::RealType;
    using BoolType = typename DataType::BoolType;

public:
    void SetParams(const std::tuple<real, real, real, int> params)
    {
        std::tie(r_, lambda_, alpha_, idx_) = params;

        rsq_          = r_ * r_;
        BoolType mask = 0.0 < rsq_;
        rInv_         = gmx::maskzInv(r_, mask);

        force_ = 0.0;
        potential_ = 0.0;
        dvdl_ = 0.0;
    }

    std::vector<real> toVector(RealType data)
    {
#if GMX_SIMD_HAVE_REAL
        alignas(GMX_SIMD_ALIGNMENT) real mem[DataType::simdRealWidth];
#else
        real mem[DataType::simdRealWidth];
#endif
        gmx::store(mem, data);
        std::vector<real> vecData(mem, mem+DataType::simdRealWidth);
        return vecData;
    }

    void reactionField()
    {
        reactionFieldQuadraticPotential(params_.qq_, r_, lambda_, params_.dLambda_, params_.sigma_,
                                        alpha_, params_.forceShift_, params_.potentialShift_,
                                        &force_, &potential_, &dvdl_, computeMask_);
    }

    void ewaldCoulomb()
    {
        ewaldQuadraticPotential(params_.qq_, r_, lambda_, params_.dLambda_, params_.sigma_, alpha_,
                                params_.ewaldShift_, &force_, &potential_, &dvdl_, computeMask_);
    }

    void lennardJones()
    {
        lennardJonesQuadraticPotential(params_.c6_, params_.c12_, r_, rsq_, lambda_, params_.dLambda_,
                                       params_.sigma_, alpha_, params_.repulsionShift_,
                                       params_.dispersionShift_, &force_, &potential_, &dvdl_, computeMask_);
    }

    // fixed parameters
    const Constants<RealType> params_;
    const BoolType computeMask_ = true;

    // input data
    RealType r_;
    RealType rInv_;
    RealType rsq_;
    RealType alpha_;
    real lambda_;
    int idx_;

    // output values
    RealType force_;
    RealType potential_;
    RealType dvdl_;

};

class SoftcoreGapsysTest :
    public ::testing::TestWithParam<std::tuple<real, real, real, int>>
{
public:
    SoftcoreGapsysTest() :
        tolerance_(
          gmx::test::relativeToleranceAsPrecisionDependentFloatingPoint(
            1, 1.0e-4, 1.0e-12))
    {}

    void SetUp() override
    {
        std::tuple<real, real, real, int> params = GetParam();
        softcore_.SetParams(params);
    }

    void compareVectorToReal(std::vector<real> vec, real value)
    {
        for (auto it=vec.begin(); it!=vec.end(); it++)
        {
            EXPECT_REAL_EQ_TOL(*it, value, tolerance_);
        }
    }

    template<class ReferenceData>
    void compareToReference(ReferenceData refData)
    {
        // transform multidata to vector
        std::vector<real> force(softcore_.toVector(softcore_.force_));
        std::vector<real> potential(softcore_.toVector(softcore_.potential_));
        std::vector<real> dvdl(softcore_.toVector(softcore_.dvdl_));

        // get reference values
        int idx           = softcore_.idx_;
        real refForce     = refData.force_[idx];
        real refPotential = refData.potential_[idx];
        real refDvdl      = refData.dvdl_[idx];

        compareVectorToReal(force, refForce);
        compareVectorToReal(potential, refPotential);
        compareVectorToReal(dvdl, refDvdl);
    }

    void compareToZero()
    {
        // transform multidata to vector
        real force     = gmx::reduce(softcore_.force_);
        real potential = gmx::reduce(softcore_.potential_);
        real dvdl      = gmx::reduce(softcore_.dvdl_);

        EXPECT_EQ(force, 0.0_real);
        EXPECT_EQ(potential, 0.0_real);
        EXPECT_EQ(dvdl, 0.0_real);
    }

#if GMX_SIMD_HAVE_REAL
    SoftcoreGapsys<SimdDataTypes> softcore_;
#else
    SoftcoreGapsys<ScalarDataTypes> softcore_;
#endif

    gmx::test::FloatingPointTolerance tolerance_;
};


TEST_P(SoftcoreGapsysTest, reactionField)
{
    softcore_.reactionField();
    compareToReference(s_results.reactionField_);
}

TEST_P(SoftcoreGapsysTest, ewaldCoulomb)
{
    softcore_.ewaldCoulomb();
    compareToReference(s_results.ewald_);
}

TEST_P(SoftcoreGapsysTest, lennardJones)
{
    softcore_.lennardJones();
    compareToReference(s_results.lennardJones_);
}

INSTANTIATE_TEST_CASE_P(CheckValues, SoftcoreGapsysTest,
                        ::testing::ValuesIn(s_data.getData()));


class SoftcoreGapsysZeroTest : public SoftcoreGapsysTest
{};

TEST_P(SoftcoreGapsysZeroTest, reactionField)
{
    softcore_.reactionField();
    compareToZero();

}

TEST_P(SoftcoreGapsysZeroTest, ewaldCoulomb)
{
    softcore_.ewaldCoulomb();
    compareToZero();
}

TEST_P(SoftcoreGapsysZeroTest, lennardJones)
{
    softcore_.lennardJones();
    compareToZero();
}

INSTANTIATE_TEST_CASE_P(CheckZeros, SoftcoreGapsysZeroTest,
                        ::testing::Values(std::make_tuple(0.1, 1.0, 0.35, 0), std::make_tuple(0.1, 0.4, 0.0, 0)));

} // namespace
