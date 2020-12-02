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
/*! \libinternal \file
 * \brief Translation layer to GROMACS data structures for force calculations.
 *
 * Implements the translation layer between the user scope and
 * GROMACS data structures for force calculations. Sets up the
 * non-bonded verlet.
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */

#ifndef NBLIB_GMXSETUP_H
#define NBLIB_GMXSETUP_H

#include "nblib/gmxcalculator.h"
#include "nblib/simulationstate.h"

namespace Nbnxm
{
struct KernelSetup;
}

namespace nblib
{

/*! \brief Sets up the GROMACS data structures for the non-bonded force calculator
 *
 * This data structure initializes the GmxForceCalculator object which internally
 * contains various objects needed to perform non-bonded force calculations using
 * the internal representation for the problem as required for GROMACS.
 *
 * The public functions of this class basically translate the problem description
 * specified by the user in NBLIB. This ultimately returns the GmxForceCalculator
 * object which is used by the ForceCalculator object in the user-facing library.
 *
 */
class NbvSetupUtil final
{
public:
    NbvSetupUtil();

    //! Sets hardware params from the execution context
    void setExecutionContext(const NBKernelOptions& options);

    //! Sets non-bonded parameters to be used to build GMX data structures
    void setNonBondedParameters(const std::vector<ParticleType>& particleTypes,
                                const NonBondedInteractionMap&   nonBondedInteractionMap);

    //! Marks particles to have Van der Waals interactions
    void setParticleInfoAllVdv(size_t numParticles);

    //! Returns the kernel setup
    Nbnxm::KernelSetup getKernelSetup(const NBKernelOptions& options);

    //! Set up StepWorkload data
    void setupStepWorkload(const NBKernelOptions& options);

    //! Return an interaction constants struct with members set appropriately
    void setupInteractionConst(const NBKernelOptions& options);

    //! Sets Particle Types and Charges and VdW params
    void setAtomProperties(const std::vector<int>&  particleTypeIdOfAllParticles,
                           const std::vector<real>& charges);

    //! Sets up non-bonded verlet on the GmxForceCalculator
    void setupNbnxmInstance(size_t numParticleTypes, const NBKernelOptions& options);

    //! Puts particles on a grid based on bounds specified by the box
    void setParticlesOnGrid(const std::vector<Vec3>& coordinates, const Box& box);

    //! Constructs pair lists
    void constructPairList(const gmx::ListOfLists<int>& exclusions);

    //! Sets up t_forcerec object on the GmxForceCalculator
    void setupForceRec(const matrix& box);

    //! Returns a unique pointer a GmxForceCalculator object
    std::unique_ptr<GmxForceCalculator> getGmxForceCalculator()
    {
        return std::move(gmxForceCalculator_);
    }

private:
    //! Storage for parameters for short range interactions.
    std::vector<real> nonbondedParameters_;

    //! Particle info where all particles are marked to have Van der Waals interactions
    std::vector<int> particleInfoAllVdw_;

    //! GROMACS force calculator to compute forces
    std::unique_ptr<GmxForceCalculator> gmxForceCalculator_;
};

/*! \brief Calls the setup utilities needed to initialize a GmxForceCalculator object
 *
 * The GmxSetupDirector encapsulates the multi-stage setup of the GmxForceCalculator which
 * is done using the public functions of the NbvSetupUtil. This separation ensures that the
 * NbvSetupUtil object is temporary in scope. The function definition makes it easy for the
 * developers to follow the sequence of calls and the dataflow involved in setting up
 * the non-bonded force calculation backend. This is the only function needed to be called
 * from the ForceCalculator during construction.
 *
 */
class GmxSetupDirector
{
public:
    //! Sets up and returns a GmxForceCalculator
    static std::unique_ptr<GmxForceCalculator> setupGmxForceCalculator(const SimulationState& system,
                                                                       const NBKernelOptions& options);
};

} // namespace nblib
#endif // NBLIB_GMXSETUP_H
