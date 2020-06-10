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
 * Declares parameters needed to evaluate forces and energies for constant pH
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_CONSTANTPHPARAMETERS_H
#define GMX_APPLIED_FORCES_CONSTANTPHPARAMETERS_H

#include <array>
#include <string>
#include <vector>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

namespace gmx
{

//! Holds state dependent values for a lambda coordinate.
struct LambdaCoordinateState
{
    //! Lambda coordinate position.
    real position = 0.0;
    //! Lambda coordinate velcocity.
    real velocity = 0.0;
};

//! Holds all current values of a lambda.
struct LambdaState
{
    //! State at start of simulation.
    LambdaCoordinateState initial;
    //! Previous coordinate state.
    LambdaCoordinateState previous;
    //! Current coordinate state.
    LambdaCoordinateState current;
    //! Lambda Kinetic energy.
    real kineticEnergy = 0.0;
    //! Lambda Temperature.
    real temperature = 0.0;
    //! Lambda dv/dl.
    real dvdl = 0.0;
};

/*!\internal
 * \brief
 * Defines a collection of atoms under control of a lambda.
 *
 * Should later be constructed and updated with information from domdec.
 */
struct LambdaAtomCollection
{
    //! Current state of the lambda for this collection.
    LambdaState state;
    //! Name for our collection.
    std::string name;
    //! The atoms controlled by this lambda.
    std::vector<int> atoms;

    //! Check if this collection controls the same atoms as another.
    bool operator==(const LambdaAtomCollection& other)
    {
        return name == other.name && atoms == other.atoms;
    }

    //! Check if this collection controls the same atoms as another.
    bool operator!=(const LambdaAtomCollection& other) { return !(*this == other); }
};

/*!\internal
 * \brief
 * Data needed to define a residue in cpHMD
 *
 * Used together with LambdaAtomCollections to provide current
 * charges for residues under lambda control.
 */
struct constantpHResidue
{
    //! The mapping to the collection of lambda controlled atoms that define this residue.
    std::vector<int> atoms_;
    //! Global residue index for nice reporting.
    int globalResidueIndex_ = -1;
    //! pKa value for this residue.
    real residuepKa = 0.0;
    //! Vector of calibration coefficients.
    std::vector<real> dvdlCalibrationCoefficients;
    //! Coefficients for double well barrier potential.
    std::array<real, 15> doubleWellBarrierPotentialCoefficients;
    //! Charge for A state.
    std::vector<real> chargeA_;
    //! Charge for B state.
    std::vector<real> chargeB_;
};

/*!\internal
 * \brief Holding all directly user-provided parameters for constant pH.
 *
 * Also used for setting all default parameters.
 */
struct ConstantpHParameters
{
    //! Indicate if density fitting is active
    bool active_ = false;
    //! Mass of lambda particles.
    double mass_ = 10.0;
    //! Collection of residues.
    std::vector<constantpHResidue> cpHMDresdiues_;
    //! The pH value for the simulation.
    real simulationpH_ = 0.0;
    //! Coupling constant for lambdas.
    real couplingConstant_ = 0.0;
};

} // namespace gmx

#endif // GMX_APPLIED_FORCES_CONSTANTPHPARAMETERS_H
