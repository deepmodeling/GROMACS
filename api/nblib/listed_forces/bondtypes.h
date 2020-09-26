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
 * Implements nblib supported bondtypes
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#ifndef NBLIB_LISTEDFORCES_BONDTYPES_H
#define NBLIB_LISTEDFORCES_BONDTYPES_H

#include <array>

#include "nblib/particletype.h"
#include "nblib/ppmap.h"
#include "nblib/util/user.h"

namespace nblib
{
using Name          = std::string;
using ForceConstant = real;
using EquilDistance = real;
using Exponent      = real;

using Degrees = StrongType<real, struct DegreeParameter>;
using Radians = StrongType<real, struct RadianParameter>;

#define CREATE_MEMBER(NAME, INDEX)                               \
    inline auto&       NAME() { return std::get<INDEX>(*this); } \
    inline const auto& NAME() const { return std::get<INDEX>(*this); }
#define NAMED_MEMBERS(...) MAP_ENUMERATE(CREATE_MEMBER, __VA_ARGS__)

/* NAMED_MEMBERS example expansion
   NAMED_MEMBERS(forceConstant, equilDistance) will expand to

   inline auto& forceConstant() { return std::get<0>(*this); }
   inline auto& equilDistance() { return std::get<1>(*this); }

   + const versions
   Note that the index for each member name is inferred from its position in the argument list
 */


/*! \brief Harmonic bond type
 *
 *  It represents the interaction of the form
 *  V(r; forceConstant, equilDistance) = 0.5 * forceConstant * (r - equilDistance)^2
 */
struct HarmonicBondType : public std::tuple<real, real>
{
    HarmonicBondType() = default;
    HarmonicBondType(ForceConstant f, EquilDistance d) : std::tuple<real, real>{ f, d } {}

    NAMED_MEMBERS(forceConstant, equilDistance)
};


/*! \brief GROMOS bond type
 *
 * It represents the interaction of the form
 * V(r; forceConstant, equilDistance) = 0.25 * forceConstant * (r^2 - equilDistance^2)^2
 */
struct G96BondType : public std::tuple<real, real>
{
    G96BondType() = default;
    G96BondType(ForceConstant f, EquilDistance d) : std::tuple<real, real>{ f, d } {}

    NAMED_MEMBERS(forceConstant, equilDistance)
};


/*! \brief Cubic bond type
 *
 * It represents the interaction of the form
 * V(r; quadraticForceConstant, cubicForceConstant, equilDistance) = quadraticForceConstant * (r -
 * equilDistance)^2 + quadraticForceConstant * cubicForceConstant * (r - equilDistance)
 */
struct CubicBondType : public std::tuple<real, real, real>
{
    CubicBondType() = default;
    CubicBondType(ForceConstant fq, ForceConstant fc, EquilDistance d) :
        std::tuple<real, real, real>{ fq, fc, d }
    {
    }

    NAMED_MEMBERS(quadraticForceConstant, cubicForceConstant, equilDistance)
};


/*! \brief FENE bond type
 *
 * It represents the interaction of the form
 * V(r; forceConstant, equilDistance) = - 0.5 * forceConstant * equilDistance^2 * log( 1 - (r / equilDistance)^2)
 */
struct FENEBondType : public std::tuple<real, real>
{
    FENEBondType() = default;
    FENEBondType(ForceConstant f, EquilDistance d) : std::tuple<real, real>{ f, d } {}

    NAMED_MEMBERS(forceConstant, equilDistance)
};


/*! \brief Morse bond type
 *
 * It represents the interaction of the form
 * V(r; forceConstant, exponent, equilDistance) = forceConstant * ( 1 - exp( -exponent * (r - equilDistance))
 */
struct MorseBondType : public std::tuple<real, real, real>
{
    MorseBondType() = default;
    MorseBondType(ForceConstant f, Exponent e, EquilDistance d) :
        std::tuple<real, real, real>{ f, e, d }
    {
    }

    NAMED_MEMBERS(forceConstant, exponent, equilDistance)
};


/*! \brief Half-attractive quartic bond type
 *
 * It represents the interaction of the form
 * V(r; forceConstant, equilDistance) = 0.5 * forceConstant * (r - equilDistance)^4
 */
struct HalfAttractiveQuarticBondType : public std::tuple<real, real>
{
    HalfAttractiveQuarticBondType() = default;
    HalfAttractiveQuarticBondType(ForceConstant f, EquilDistance d) : std::tuple<real, real>{ f, d }
    {
    }

    NAMED_MEMBERS(forceConstant, equilDistance)
};


/*! \brief default angle type
 *
 * Note: the angle is always stored as radians internally
 */
struct DefaultAngle : public std::tuple<real, real>
{
    DefaultAngle() = default;
    //! \brief construct from angle given in radians
    DefaultAngle(Radians angle, ForceConstant f) : std::tuple<real, real>{ angle, f } {}

    //! \brief construct from angle given in degrees
    DefaultAngle(Degrees angle, ForceConstant f) : std::tuple<real, real>{ angle * DEG2RAD, f } {}

    NAMED_MEMBERS(equilDistance, forceConstant)
};


/*! \brief Proper Dihedral Implementation
 */
struct ProperDihedral : public std::tuple<real, real, int>
{
    using Multiplicity = int;

    ProperDihedral() = default;
    ProperDihedral(Radians phi, ForceConstant f, Multiplicity m) :
        std::tuple<real, real, int>{ phi, f, m }
    {
    }
    ProperDihedral(Degrees phi, ForceConstant f, Multiplicity m) :
        std::tuple<real, real, int>{ phi * DEG2RAD, f, m }
    {
    }

    NAMED_MEMBERS(equilDistance, forceConstant, multiplicity)
};


/*! \brief Improper Dihedral Implementation
 */
struct ImproperDihedral : public std::tuple<real, real>
{
    ImproperDihedral() = default;
    ImproperDihedral(Radians phi, ForceConstant f) : std::tuple<real, real>{ phi, f } {}
    ImproperDihedral(Degrees phi, ForceConstant f) : std::tuple<real, real>{ phi * DEG2RAD, f } {}

    NAMED_MEMBERS(equilDistance, forceConstant)
};

/*! \brief Ryckaert-Belleman Dihedral Implementation
 */
struct RyckaertBellemanDihedral : public std::array<real, 6>
{
};


/*! \brief Type for 5-center interaction (C-MAP)
 */
struct Default5Center : public std::tuple<real, real, real, real>
{
    using Multiplicity = int;

    Default5Center() = default;
    Default5Center(Radians phi, Radians psi, ForceConstant fphi, ForceConstant fpsi) :
        std::tuple<real, real, real, real>{ phi, psi, fphi, fpsi }
    {
    }

    NAMED_MEMBERS(phi, psi, forceConstantPhi, forceConstantPsi)
};


} // namespace nblib
#endif // NBLIB_LISTEDFORCES_BONDTYPES_H
