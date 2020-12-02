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
 * This implements topology setup tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#include <gtest/gtest.h>

#include "gromacs/topology/exclusionblocks.h"
#include "nblib/forcecalculator.h"
#include "nblib/gmxsetup.h"
#include "nblib/integrator.h"
#include "nblib/tests/testhelpers.h"
#include "nblib/tests/testsystems.h"
#include "nblib/topology.h"

namespace nblib
{
namespace test
{
namespace
{

// This is defined in src/gromacs/mdtypes/forcerec.h but there is also a
// legacy C6 macro defined there that conflicts with the nblib C6 type.
// Todo: Once that C6 has been refactored into a regular function, this
//       file can just include forcerec.h
//! Macro to set Van der Waals interactions to atoms
#define SET_CGINFO_HAS_VDW(cgi) (cgi) = ((cgi) | (1 << 23))

TEST(NBlibTest, SpcMethanolForcesAreCorrect)
{
    auto options        = NBKernelOptions();
    options.nbnxmSimd   = SimdKernels::SimdNo;
    options.coulombType = CoulombType::Cutoff;

    SpcMethanolSimulationStateBuilder spcMethanolSystemBuilder;

    auto simState        = spcMethanolSystemBuilder.setupSimulationState();
    auto forceCalculator = ForceCalculator(simState, options);

    gmx::ArrayRef<Vec3> forces(simState.forces());
    ASSERT_NO_THROW(forceCalculator.compute(simState.coordinates(), forces));

    /* Use higher-than-usual tolerance for forces. Some of the particles in the test systems are
     * very close to each other, and, for example, the distance between the first two particles
     * is approx. 0.13 and already has relative uncertainty around 1e-6. */
    gmx::test::FloatingPointTolerance forceTolerance(1.0e-5, 1.0e-9, 1e-4, 1.0e-9, 1000, 1000, true);

    Vector3DTest forcesOutputTest(forceTolerance);
    forcesOutputTest.testVectors(forces, "SPC-methanol forces");
}

TEST(NBlibTest, ExpectedNumberOfForces)
{
    auto options      = NBKernelOptions();
    options.nbnxmSimd = SimdKernels::SimdNo;

    SpcMethanolSimulationStateBuilder spcMethanolSystemBuilder;

    auto simState        = spcMethanolSystemBuilder.setupSimulationState();
    auto forceCalculator = ForceCalculator(simState, options);

    gmx::ArrayRef<Vec3> forces(simState.forces());
    forceCalculator.compute(simState.coordinates(), forces);
    EXPECT_EQ(simState.topology().numParticles(), forces.size());
}

TEST(NBlibTest, CanIntegrateSystem)
{
    auto options          = NBKernelOptions();
    options.nbnxmSimd     = SimdKernels::SimdNo;
    options.numIterations = 1;

    SpcMethanolSimulationStateBuilder spcMethanolSystemBuilder;

    auto simState        = spcMethanolSystemBuilder.setupSimulationState();
    auto forceCalculator = ForceCalculator(simState, options);

    LeapFrog integrator(simState.topology(), simState.box());

    for (int iter = 0; iter < options.numIterations; iter++)
    {
        gmx::ArrayRef<Vec3> forces(simState.forces());
        forceCalculator.compute(simState.coordinates(), simState.forces());
        EXPECT_NO_THROW(integrator.integrate(1.0, simState.coordinates(), simState.velocities(),
                                             simState.forces()));
    }
}

/*!
 * Check if the following aspects of the ForceCalculator and
 * LeapFrog (integrator) work as expected:
 *
 * 1. Calling the ForceCalculator::compute() function makes no change
 *    to the internal representation of the system. Calling it repeatedly
 *    without update should return the same vector of forces.
 *
 * 2. Once the LeapFrog objects integrates for the given time using the
 *    forces, there the coordinates in SimulationState must change.
 *    Calling the compute() function must now generate a new set of forces.
 *
 */
TEST(NBlibTest, UpdateChangesForces)
{
    auto options          = NBKernelOptions();
    options.nbnxmSimd     = SimdKernels::SimdNo;
    options.numIterations = 1;

    SpcMethanolSimulationStateBuilder spcMethanolSystemBuilder;

    auto simState        = spcMethanolSystemBuilder.setupSimulationState();
    auto forceCalculator = ForceCalculator(simState, options);

    LeapFrog integrator(simState.topology(), simState.box());

    // step 1
    gmx::ArrayRef<Vec3> forces(simState.forces());
    forceCalculator.compute(simState.coordinates(), simState.forces());

    // copy computed forces to another array
    std::vector<Vec3> forces_1(forces.size());
    std::copy(forces.begin(), forces.end(), begin(forces_1));

    // zero original force buffer
    zeroCartesianArray(forces);

    // check if forces change without update step
    forceCalculator.compute(simState.coordinates(), forces);

    // check if forces change without update
    for (size_t i = 0; i < forces_1.size(); i++)
    {
        for (int j = 0; j < dimSize; j++)
        {
            EXPECT_EQ(forces[i][j], forces_1[i][j]);
        }
    }

    // update
    integrator.integrate(1.0, simState.coordinates(), simState.velocities(), simState.forces());

    // zero original force buffer
    zeroCartesianArray(forces);

    // step 2
    forceCalculator.compute(simState.coordinates(), forces);
    std::vector<Vec3> forces_2(forces.size());
    std::copy(forces.begin(), forces.end(), begin(forces_2));

    // check if forces change after update
    for (size_t i = 0; i < forces_1.size(); i++)
    {
        for (int j = 0; j < dimSize; j++)
        {
            EXPECT_NE(forces_1[i][j], forces_2[i][j]);
        }
    }
}

TEST(NBlibTest, ArgonForcesAreCorrect)
{
    auto options        = NBKernelOptions();
    options.nbnxmSimd   = SimdKernels::SimdNo;
    options.coulombType = CoulombType::Cutoff;

    ArgonSimulationStateBuilder argonSystemBuilder;

    auto simState        = argonSystemBuilder.setupSimulationState();
    auto forceCalculator = ForceCalculator(simState, options);

    gmx::ArrayRef<Vec3> testForces(simState.forces());
    forceCalculator.compute(simState.coordinates(), simState.forces());

    Vector3DTest forcesOutputTest;
    forcesOutputTest.testVectors(testForces, "Argon forces");
}

} // namespace
} // namespace test
} // namespace nblib
