/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2012,2014,2015,2018,2019, The GROMACS development team.
 * Copyright (c) 2020,2021, by the GROMACS development team, led by
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
#ifndef GMX_MATH_UNITS_H
#define GMX_MATH_UNITS_H

#include <cmath>

/*
 * Physical constants to be used in Gromacs.
 * No constants (apart from 0, 1 or 2) should
 * be anywhere else in the code.
 */

namespace gmx
{

constexpr double ANGSTROM  = 1e-10;
constexpr double KILO      = 1e3;
constexpr double NANO      = 1e-9;
constexpr double PICO      = 1e-12;
constexpr double NM2A      = NANO / ANGSTROM;
constexpr double RAD2DEG   = 180.0 / M_PI;
constexpr double DEG2RAD   = M_PI / 180.0;
constexpr double CAL2JOULE = 4.184;           /* Exact definition of the calorie */
constexpr double E_CHARGE  = 1.602176634e-19; /* Exact definition, Coulomb NIST 2018 CODATA */

constexpr double AMU       = 1.66053906660e-27;    /* kg, NIST 2018 CODATA  */
constexpr double BOLTZMANN = 1.380649e-23;         /* (J/K, Exact definition, NIST 2018 CODATA */
constexpr double AVOGADRO  = 6.02214076e23;        /* 1/mol, Exact definition, NIST 2018 CODATA */
constexpr double RGAS      = BOLTZMANN * AVOGADRO; /* (J/(mol K))  */
constexpr double BOLTZ     = RGAS / KILO;          /* (kJ/(mol K)) */
constexpr double FARADAY   = E_CHARGE * AVOGADRO;  /* (C/mol)      */
constexpr double PLANCK1   = 6.62607015e-34;       /* J/Hz, Exact definition, NIST 2018 CODATA */
constexpr double PLANCK    = (PLANCK1 * AVOGADRO / (PICO * KILO)); /* (kJ/mol) ps */

constexpr double EPSILON0_SI = 8.8541878128e-12; /* F/m,  NIST 2018 CODATA */
/* Epsilon in our MD units: (e^2 / Na (kJ nm)) == (e^2 mol/(kJ nm)) */
constexpr double EPSILON0 = ((EPSILON0_SI * NANO * KILO) / (E_CHARGE * E_CHARGE * AVOGADRO));

constexpr double SPEED_OF_LIGHT =
        2.99792458e05; /* units of nm/ps, Exact definition, NIST 2018 CODATA */

constexpr double RYDBERG = 1.0973731568160e-02; /* nm^-1, NIST 2018 CODATA */

constexpr double ONE_4PI_EPS0 = (1.0 / (4.0 * M_PI * EPSILON0));

/* Pressure in MD units is:
 * 1 bar = 1e5 Pa = 1e5 kg m^-1 s^-2 = 1e-28 kg nm^-1 ps^-2 = 1e-28 / AMU amu nm^1 ps ^2
 */
constexpr double BAR_MDUNITS = (1e5 * NANO * PICO * PICO / AMU);
constexpr double PRESFAC     = 1.0 / BAR_MDUNITS;

/* DEBYE2ENM should be (1e-21*PICO)/(SPEED_OF_LIGHT*E_CHARGE*NANO*NANO),
 * but we need to factor out some of the exponents to avoid single-precision overflows.
 */
constexpr double DEBYE2ENM = (1e-15 / (SPEED_OF_LIGHT * E_CHARGE));
constexpr double ENM2DEBYE = 1.0 / DEBYE2ENM;

/* to convert from a acceleration in (e V)/(amu nm) */
/* FIELDFAC is also Faraday's constant and E_CHARGE/(1e6 AMU) */
constexpr double FIELDFAC = FARADAY / KILO;

/* to convert AU to MD units: */
constexpr double HARTREE2KJ      = ((2.0 * RYDBERG * PLANCK * SPEED_OF_LIGHT) / AVOGADRO);
constexpr double BOHR2NM         = 0.0529177210903; /* nm^-1, NIST 2018 CODATA */
constexpr double HARTREE_BOHR2MD = (HARTREE2KJ * AVOGADRO / BOHR2NM);

} // namespace gmx

/* The four basic units */
#define unit_length "nm"
#define unit_time "ps"
#define unit_mass "u"
#define unit_energy "kJ/mol"

/* Temperature unit, T in this unit times BOLTZ give energy in unit_energy */
#define unit_temp_K "K"

/* Charge unit, electron charge, involves ONE_4PI_EPS0 */
#define unit_charge_e "e"

/* Pressure unit, pressure in basic units times PRESFAC gives this unit */
#define unit_pres_bar "bar"

/* Dipole unit, debye, conversion from the unit_charge_e involves ENM2DEBYE */
#define unit_dipole_D "D"

/* Derived units from basic units only */
#define unit_vel unit_length "/" unit_time
#define unit_volume unit_length "^3"
#define unit_invtime "1/" unit_time

/* Other derived units */
#define unit_surft_bar unit_pres_bar " " unit_length

/* SI units, conversion from basic units involves NANO, PICO and AMU */
#define unit_length_SI "m"
#define unit_time_SI "s"
#define unit_mass_SI "kg"

#define unit_density_SI unit_mass_SI "/" unit_length_SI "^3"
#define unit_invvisc_SI unit_length_SI " " unit_time_SI "/" unit_mass_SI

#endif
