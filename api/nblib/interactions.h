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
/*! \inpublicapi \file
 * \brief
 * Implements nblib particle-types interactions
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#ifndef GMX_NBLIB_INTERACTIONS_H
#define GMX_NBLIB_INTERACTIONS_H

#include <map>
#include <tuple>
#include <unordered_map>

#include "nblib/kerneloptions.h"
#include "nblib/particletype.h"

namespace nblib
{

//! Alias for specifying particle type name
using ParticleTypeName = std::string;
//! Shorthand for a map used for looking up non-bonded parameters using particle types
//! Alias for the C6 parameter in the Lennard-Jones potential
using C6 = real;
//! Alias for the C12 parameter in the Lennard-Jones potential
using C12 = real;
using NonBondedInteractionMapImpl =
        std::map<std::tuple<ParticleTypeName, ParticleTypeName>, std::tuple<C6, C12>>;

class NonBondedInteractionMap : public NonBondedInteractionMapImpl
{
public:
    C6  getC6(const ParticleTypeName&, const ParticleTypeName&) const;
    C12 getC12(const ParticleTypeName&, const ParticleTypeName&) const;
};

namespace detail
{

//! Combines the non-bonded parameters from two particles for pairwise interactions
real combineNonbondedParameters(real v, real w, CombinationRule combinationRule);

} // namespace detail

/*! \brief Non-Bonded Interactions between Particle Types
 *
 * \inpublicapi
 * \ingroup nblib
 *
 * A class to hold a mapping between pairs of particle types and the non-bonded
 * interactions between them. One may add the non-bonded parameters, namely the
 * C6/C12 params for each particle type individually and construct a pair-wise
 * mapping using combination rules or manually specify the parameters between
 * a specific pair.
 *
 */
class ParticleTypesInteractions
{
public:
    //! Initialized with the default geometric combination rule
    explicit ParticleTypesInteractions(CombinationRule = CombinationRule::Geometric);

    //! Specify non-bonded params of a particle type
    void add(const ParticleTypeName& particleTypeName, C6 c6, C12 c12);

    //! Specify the non-bonded params of a specific pair of particle types
    void add(const ParticleTypeName& particleTypeName1, const ParticleTypeName& particleTypeName2, C6 c6, C12 c12);

    //! Generate table based on the parameters stored
    NonBondedInteractionMap generateTable();

    //! Get combination rule enabled in this object
    CombinationRule getCombinationRule() const;

    //! Merge with the information stored in another ParticleTypesInteractions object
    void merge(const ParticleTypesInteractions&);

private:
    CombinationRule combinationRule_;

    std::unordered_map<ParticleTypeName, std::tuple<C6, C12>> singleParticleInteractionsMap_;
    std::map<std::tuple<ParticleTypeName, ParticleTypeName>, std::tuple<C6, C12>> twoParticlesInteractionsMap_;
};

} // namespace nblib
#endif // GMX_NBLIB_INTERACTIONS_H
