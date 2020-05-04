/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020, by the GROMACS development team, led by
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
/*! \inlibraryapi \file
 * \brief
 * Interface class for frame handling, provides handles for all calls.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */
#ifndef GMX_COORDINATEIO_IFRAMECONVERTER_H
#define GMX_COORDINATEIO_IFRAMECONVERTER_H

#include <memory>

#include "frameconverterenums.h"

struct t_trxframe;

namespace gmx
{

/*!\inlibraryapi
 * \brief
 * IFrameConverter interface for manipulating coordinate information.
 *
 * This interface is aimed at providing the base methods to manipulate the
 * coordinate (usually position) data in a t_trxframe object according
 * to the requirements of a analysis module. It is similar to the
 * ICoordinateOutput interface, but instead of
 * simply passing through frames returns new t_trxframe objects with
 * changes applied to them.
 *
 * \ingroup module_coordinateio
 *
 */
class IFrameConverter
{
public:
    IFrameConverter() {}

    virtual ~IFrameConverter() {}

    //! Move constructor for old object.
    explicit IFrameConverter(IFrameConverter&& old) noexcept = default;

    /*! \brief
     * Change coordinate frame information for output.
     *
     * Takes the previously internally stored coordinates and saves them again.
     * Applies correct number of atoms, as well as changing things such as
     * frame time or affect output of velocities or forces.
     * Methods derived from this should not affect total number of atoms,
     * and should check for internal consistency of the input and output data.
     *
     * \param[in,out]  input  Coordinate frame to be modified.
     */
    virtual void convertFrame(t_trxframe* input) = 0;

    /*! \brief
     * Allows other methods to probe if a specific requirement is fulfilled by running a converter.
     *
     * When modifying coordinate frames with different frameconverters,
     * it can be important to know what kind of modifications are done by a
     * specific converter, to e.g. check if it makes a system whole or moves
     * the simulation box in a specific way.
     *
     * \returns What kind of modification is guaranteed by this converter.
     */
    virtual unsigned long guarantee() const = 0;
};
/*! \brief
 * Typedef to have direct access to the individual FrameConverter modules.
 */
using FrameConverterPointer = std::unique_ptr<IFrameConverter>;

} // namespace gmx

#endif