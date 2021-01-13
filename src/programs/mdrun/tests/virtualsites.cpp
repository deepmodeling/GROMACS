/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021, by the GROMACS development team, led by
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
 * \brief End-to-end tests checking sanity of results of simulations
 *        containing virtual sites
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrun_integration_tests
 */
#include <testutils/testmatchers.h>
#include "gmxpre.h"

#include "config.h"

#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/mpitest.h"
#include "testutils/simulationdatabase.h"

#include "gromacs/utility/strconvert.h"

#include "moduletest.h"
#include "simulatorcomparison.h"
#include "trajectoryreader.h"

namespace gmx::test
{
namespace
{
using VirtualSiteVelocityTestParams = std::tuple<std::string, std::string, std::string>;
class VirtualSiteVelocityTest :
    public MdrunTestFixture,
    public ::testing::WithParamInterface<VirtualSiteVelocityTestParams>
{
public:
    //! Check that the velocities of virtual sites follow expected behavior
    static void checkVirtualSiteVelocities(const std::string&           trajectoryName,
                                           ArrayRef<const unsigned int> virtualSites,
                                           real                         timeStep,
                                           const TrajectoryTolerances&  tolerances)
    {
        auto [virtualPositions, virtualVelocities] =
                getVirtualPositionsAndVelocities(trajectoryName, virtualSites);
        GMX_RELEASE_ASSERT(virtualPositions.size() == virtualVelocities.size(),
                           "Position and velocity trajectory don't have the same length.");
        const auto trajectorySize = virtualPositions.size();

        checkFirstFrame(virtualVelocities[0], tolerances);
        for (auto frameIdx = decltype(trajectorySize){ 1 }; frameIdx < trajectorySize; frameIdx++)
        {
            SCOPED_TRACE(formatString("Checking frame %lu", frameIdx + 1));
            checkFrame(virtualVelocities[frameIdx],
                       virtualPositions[frameIdx],
                       virtualPositions[frameIdx - 1],
                       tolerances,
                       timeStep);
        }
    }

    static std::tuple<std::vector<std::vector<RVec>>, std::vector<std::vector<RVec>>>
    getVirtualPositionsAndVelocities(const std::string& trajectoryName, ArrayRef<const unsigned int> virtualSites)
    {
        std::vector<std::vector<RVec>> positions;
        std::vector<std::vector<RVec>> velocities;

        TrajectoryFrameReader trajectoryFrameReader(trajectoryName);
        while (trajectoryFrameReader.readNextFrame())
        {
            const auto frame = trajectoryFrameReader.frame();
            positions.emplace_back();
            velocities.emplace_back();
            for (const auto& index : virtualSites)
            {
                positions.back().emplace_back(frame.x().at(index));
                velocities.back().emplace_back(frame.v().at(index));
            }
        }

        return { std::move(positions), std::move(velocities) };
    }

    //! Check that first frame velocities are zero
    static void checkFirstFrame(ArrayRef<const RVec> velocities, const TrajectoryTolerances& tolerances)
    {
        SCOPED_TRACE("Checking first frame");
        std::vector<RVec> zeroVelocities(velocities.size(), RVec{ 0, 0, 0 });
        EXPECT_THAT(velocities, Pointwise(RVecEq(tolerances.velocities), zeroVelocities));
    }

    //! Check that velocities are equal to the (current positions - old positions)/dt
    static void checkFrame(ArrayRef<const RVec>        velocities,
                           ArrayRef<const RVec>        positions,
                           ArrayRef<const RVec>        previousPositions,
                           const TrajectoryTolerances& tolerances,
                           real                        timeStep)
    {
        std::vector<RVec> referenceVSiteVelocities = {};
        GMX_RELEASE_ASSERT(positions.size() == previousPositions.size(),
                           "Length of positions vector differs between time steps.");
        const auto frameSize = positions.size();
        for (auto frameIdx = decltype(frameSize){ 0 }; frameIdx < frameSize; frameIdx++)
        {
            referenceVSiteVelocities.emplace_back((positions[frameIdx] - previousPositions[frameIdx])
                                                  / timeStep);
        }
        EXPECT_THAT(velocities, Pointwise(RVecEq(tolerances.velocities), referenceVSiteVelocities));
    }
};

TEST_P(VirtualSiteVelocityTest, WithinTolerances)
{
    const auto&                            params         = GetParam();
    const auto&                            integrator     = std::get<0>(params);
    const auto&                            tcoupling      = std::get<1>(params);
    const auto&                            pcoupling      = std::get<2>(params);
    const real                             timeStep       = 0.001;
    const auto&                            simulationName = "alanine_vsite_vacuo";
    constexpr std::array<unsigned int, 15> virtualSites   = { 2,  3,  4,  5,  7,  10, 11, 12,
                                                            13, 17, 19, 22, 23, 24, 25 };

    if (integrator == "md-vv" && pcoupling == "parrinello-rahman")
    {
        // Parrinello-Rahman is not implemented in md-vv
        return;
    }

    // Prepare mdp input
    auto mdpFieldValues = prepareMdpFieldValues(simulationName, integrator, tcoupling, pcoupling);
    mdpFieldValues["nsteps"]  = "8";
    mdpFieldValues["nstxout"] = "1";
    mdpFieldValues["nstvout"] = "1";
    mdpFieldValues["dt"]      = toString(timeStep);

    // Run grompp
    runner_.useTopGroAndNdxFromDatabase(simulationName);
    runner_.useStringAsMdpFile(prepareMdpFileContents(mdpFieldValues));
    runGrompp(&runner_);
    // Run mdrun
    runMdrun(&runner_);

    // Check virtual site velocities
    checkVirtualSiteVelocities(runner_.fullPrecisionTrajectoryFileName_,
                               virtualSites,
                               timeStep,
                               TrajectoryComparison::s_defaultTrajectoryTolerances);
}

INSTANTIATE_TEST_CASE_P(
        VelocitiesConformToExpectations,
        VirtualSiteVelocityTest,
        ::testing::Combine(::testing::Values("md", "md-vv", "sd", "bd"),
                           ::testing::Values("no", "v-rescale", "nose-hoover"),
                           ::testing::Values("no", "c-rescale", "parrinello-rahman")));

} // namespace
} // namespace gmx::test
