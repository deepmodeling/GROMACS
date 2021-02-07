/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2017 by the GROMACS development team.
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

/* The gmx_bools indicate whether a field was read from the trajectory.
 * Do not try to use a pointer when its gmx_bool is FALSE, as memory might
 * not be allocated.
 */
#ifndef GMX_TRAJECTORY_TRX_H
#define GMX_TRAJECTORY_TRX_H

#include <cstdio>

#include <array>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct t_atoms;
enum class PbcType : int;

struct t_trxframe // NOLINT (clang-analyzer-optin.performance.Padding)
{
    int      not_ok = 0;  /* integrity flags                  */
    gmx_bool bDouble = false; /* Double precision?                */
    int      natoms = 0;  /* number of atoms (atoms, x, v, f, index) */
    gmx_bool bStep = false;
    int64_t  step = 0; /* MD step number                   */
    gmx_bool bTime = false;
    real     time = 0; /* time of the frame                */
    gmx_bool bLambda = false;
    gmx_bool bFepState = false; /* does it contain fep_state?       */
    real     lambda = 0;    /* free energy perturbation lambda  */
    int      fep_state =0; /* which fep state are we in? */
    gmx_bool bAtoms = false;
    t_atoms* atoms = nullptr; /* atoms struct (natoms)            */
    gmx_bool bPrec = false;
    real     prec = 0; /* precision of x, fraction of 1 nm */
    gmx_bool bX =false;
    rvec*    x = nullptr; /* coordinates (natoms)             */
    gmx_bool bV = false;
    rvec*    v = nullptr; /* velocities (natoms)              */
    gmx_bool bF = false;
    rvec*    f = nullptr; /* forces (natoms)                  */
    gmx_bool bBox = false;
    matrix   box = {{0}}; /* the 3 box vectors                */
    gmx_bool bPBC = false;
    PbcType  pbcType; /* the type of pbc                  */
    gmx_bool bIndex = false;
    int*     index = nullptr; /* atom indices of contained coordinates */
};

t_trxframe copyFrame(const t_trxframe& src)

void comp_frame(FILE* fp, t_trxframe* fr1, t_trxframe* fr2, gmx_bool bRMSD, real ftol, real abstol);



void done_frame(t_trxframe* frame);

namespace gmx
{

/*!\brief A 3x3 matrix data type useful for simulation boxes
 *
 * \todo Implement a full replacement for C-style real[DIM][DIM] */
using BoxMatrix = std::array<std::array<real, DIM>, DIM>;

/*! \internal
 * \brief Contains a valid trajectory frame.
 *
 * Valid frames have a step and time, but need not have any particular
 * other fields.
 *
 * \todo Eventually t_trxframe should be replaced by a class such as
 * this. Currently we need to introduce BoxMatrix so that we can have
 * a normal C++ getter that returns the contents of a box matrix,
 * since you cannot use a real[DIM][DIM] as a function return type.
 *
 * \todo Consider a std::optional work-alike type for expressing that
 * a field may or may not have content. */
class TrajectoryFrame
{
public:
    /*! \brief Constructor
     *
     * \throws APIError If \c frame lacks either step or time.
     */
    explicit TrajectoryFrame(const t_trxframe& frame);
    /*! \brief Return a string that helps users identify this frame, containing time and step number.
     *
     * \throws std::bad_alloc  when out of memory */
    std::string frameName() const;
    //! Step number read from the trajectory file frame.
    std::int64_t step() const;
    //! Time read from the trajectory file frame.
    double time() const;
    //! The PBC characteristics of the box.
    PbcType pbc() const;
    //! Get a view of position coordinates of the frame (which could be empty).
    ArrayRef<const RVec> x() const;
    //! Get a view of velocity coordinates of the frame (which could be empty).
    ArrayRef<const RVec> v() const;
    //! Get a view of force coordinates of the frame (which could be empty).
    ArrayRef<const RVec> f() const;
    //! Return whether the frame has a box.
    bool hasBox() const;
    //! Return a handle to the frame's box, which is all zero if the frame has no box.
    const BoxMatrix& box() const;

private:
    //! Handle to trajectory data
    const t_trxframe& frame_;
    //! Box matrix data from the frame_.
    BoxMatrix box_;
};

} // namespace gmx

#endif
