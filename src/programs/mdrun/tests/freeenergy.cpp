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
 * Tests to compare free energy simulations to reference
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/mpitest.h"
#include "testutils/refdata.h"
#include "testutils/setenv.h"
#include "testutils/simulationdatabase.h"

#include "moduletest.h"
#include "simulatorcomparison.h"

namespace gmx
{
namespace test
{
namespace
{

/*! \brief Test fixture base for two equivalent simulators
 *
 * This test ensures that two simulator code paths (called via different mdp
 * options and/or environment variables) yield identical coordinate, velocity,
 * box, force and energy trajectories, up to some (arbitrary) precision.
 *
 * These tests are useful to check that re-implementations of existing simulators
 * are correct, and that different code paths expected to yield identical results
 * are equivalent.
 */
using FreeEnergyReferenceTestParams = std::string;
class FreeEnergyReferenceTest :
    public MdrunTestFixture,
    public ::testing::WithParamInterface<FreeEnergyReferenceTestParams>
{
};

TEST_P(FreeEnergyReferenceTest, WithinTolerances)
{
    auto simulationName = GetParam();

    // TODO In similar tests, we are checking if the tests
    //      can be run with the number of MPI ranks available

    SCOPED_TRACE(formatString("Comparing FEP simulation '%s' to reference", simulationName.c_str()));

    // TODO: These are the regression test tolerances. Think about them and justify them!
    const auto defaultRegressionEnergyTolerance =
            FloatingPointTolerance(0.05, 0.05, 0.001, 0.001, UINT64_MAX, UINT64_MAX, false);
    const auto gmx_unused defaultRegressionVirialTolerance =
            FloatingPointTolerance(0.1, 0.1, 0.01, 0.01, UINT64_MAX, UINT64_MAX, false);

    // TODO: Regression tests only test Epot. Add other energy terms to be tested here.
    EnergyTermsToCompare energyTermsToCompare{ {
            { interaction_function[F_EPOT].longname, defaultRegressionEnergyTolerance },
    } };

    // TODO: Regression tests only tests forces. Add checks for rest of trajectory here.
    // Specify how trajectory frame matching must work.
    TrajectoryFrameMatchSettings trajectoryMatchSettings{ false,
                                                          false,
                                                          false,
                                                          ComparisonConditions::NoComparison,
                                                          ComparisonConditions::NoComparison,
                                                          ComparisonConditions::MustCompare };
    TrajectoryTolerances trajectoryTolerances = TrajectoryComparison::s_defaultTrajectoryTolerances;

    // Build the functor that will compare reference and test
    // trajectory frames in the chosen way.
    TrajectoryComparison trajectoryComparison{ trajectoryMatchSettings, trajectoryTolerances };

    // Set simulation file names
    auto simulationTrajectoryFileName = fileManager_.getTemporaryFilePath("simulation.trr");
    auto simulationEdrFileName        = fileManager_.getTemporaryFilePath("simulation.edr");

    // Run grompp
    runner_.tprFileName_ = fileManager_.getTemporaryFilePath("sim.tpr");
    runner_.useTopGroAndMdpFromFepTestDatabase(simulationName);
    // TODO: maxwarn because we copied these systems from regressiontests -
    //       we want systems that run without warnings, or else carefully
    //       document why these are needed!
    runGrompp(&runner_, { SimulationOptionTuple("-maxwarn", "3") });

    // Do mdrun
    runner_.fullPrecisionTrajectoryFileName_ = simulationTrajectoryFileName;
    runner_.edrFileName_                     = simulationEdrFileName;
    runMdrun(&runner_);

    // Compare simulation results
    TestReferenceData    refData;
    TestReferenceChecker rootChecker(refData.rootChecker());
    // Check the energies agree with the refdata within tolerance.
    checkEnergiesAgainstReferenceData(simulationEdrFileName, energyTermsToCompare, &rootChecker);
    // Check the trajectories agree with the refdata within tolerance.
    checkTrajectoryAgainstReferenceData(simulationTrajectoryFileName, trajectoryComparison, &rootChecker);
}

// TODO: The time for OpenCL kernel compilation means these tests time
//       out. Once that compilation is cached for the whole process, these
//       tests can run in such configurations.
#if GMX_GPU != GMX_GPU_OPENCL
INSTANTIATE_TEST_CASE_P(FreeEnergyCalculationsAreEquivalentToReference,
                        FreeEnergyReferenceTest,
                        ::testing::Values("coulandvdwsequential_coul",
                                          "coulandvdwsequential_vdw",
                                          "coulandvdwtogether",
                                          "expanded",
                                          "relative",
                                          "relative-position-restraints",
                                          "restraints",
                                          "simtemp",
                                          "transformAtoB",
                                          "vdwalone"));
#else
INSTANTIATE_TEST_CASE_P(DISABLED_FreeEnergyCalculationsAreEquivalentToReference,
                        FreeEnergyReferenceTest,
                        ::testing::Values("vdwalone"));
#endif

} // namespace
} // namespace test
} // namespace gmx
