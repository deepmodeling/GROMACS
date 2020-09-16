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
 * Implements some definitions that are identical to those of gromacs
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#ifndef GMX_NBLIB_BASICDEFINITIONS_H
#define GMX_NBLIB_BASICDEFINITIONS_H

#include <cmath>

// from utility/real.h
#if GMX_DOUBLE
#    ifndef HAVE_REAL
typedef double real;
#        define HAVE_REAL
#    endif
#else /* GMX_DOUBLE */
#    ifndef HAVE_REAL
typedef float real;
#        define HAVE_REAL
#    endif
#endif /* GMX_DOUBLE */

// from math/units.h
#define KILO (1e3)                     /* Thousand	*/
#define NANO (1e-9)                    /* A Number	*/
#define E_CHARGE (1.602176634e-19)     /* Exact definition, Coulomb NIST 2018 CODATA */
#define BOLTZMANN (1.380649e-23)       /* (J/K, Exact definition, NIST 2018 CODATA */
#define AVOGADRO (6.02214076e23)       /* 1/mol, Exact definition, NIST 2018 CODATA */
#define RGAS (BOLTZMANN * AVOGADRO)    /* (J/(mol K))  */
#define BOLTZ (RGAS / KILO)            /* (kJ/(mol K)) */
#define EPSILON0_SI (8.8541878128e-12) /* F/m,  NIST 2018 CODATA */
#define EPSILON0 ((EPSILON0_SI * NANO * KILO) / (E_CHARGE * E_CHARGE * AVOGADRO))
#define ONE_4PI_EPS0 (1.0 / (4.0 * M_PI * EPSILON0))
#define DEG2RAD (M_PI / 180.0)

// from pbc/ishift.h
#define D_BOX_Z 1
#define D_BOX_Y 1
#define D_BOX_X 2
#define N_BOX_Z (2 * D_BOX_Z + 1)
#define N_BOX_Y (2 * D_BOX_Y + 1)
#define N_BOX_X (2 * D_BOX_X + 1)
#define N_IVEC (N_BOX_Z * N_BOX_Y * N_BOX_X)
#define CENTRAL (N_IVEC / 2)
#define SHIFTS N_IVEC

// from math/vectypes.h
#define XX 0 /* Defines for indexing in vectors */
#define YY 1
#define ZZ 2
#define DIM 3 /* Dimension of vectors    */
typedef real   rvec[DIM];
typedef double dvec[DIM];
typedef real   matrix[DIM][DIM];

#endif // GMX_NBLIB_BASICDEFINITIONS_H
