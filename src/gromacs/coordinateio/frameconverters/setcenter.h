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
/*! \file
 * \brief
 * Method to center molecules in box.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \inlibraryapi
 * \ingroup module_coordinateio
 */
#ifndef GMX_COORDINATEIO_FRAMECONVERTERS_SETCENTER_H
#define GMX_COORDINATEIO_FRAMECONVERTERS_SETCENTER_H

#include <algorithm>

#include "gromacs/coordinateio/iframeconverter.h"
#include "gromacs/math/vec.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/trajectory/trajectoryframe.h"

namespace gmx
{

/*! \brief
 * Helper enum class to define centering types.
 */
enum class CenteringType
{
    Triclinic,
    Rectangular,
    Zero,
    Count
};
//! Mapping for centering enum values to names.
const char* const cCenterTypeEnum[] = { "tric", "rect", "zero", "none" };

/*!\brief
 * SetCenter class performs single operation to center molecule based on a user
 * selection in the box.
 *
 * \inlibraryapi
 * \ingroup module_coordinateio
 */
class SetCenter : public IFrameConverter
{
public:
    /*! \brief
     * Construct SetCenter object with initial selection.
     *
     * Can be used to initialize SetCenter from outside of trajectoryanalysis
     * framework.
     * TODO Add initializers for the remaining fields.
     */
    explicit SetCenter(const Selection& center, const CenteringType& centerFlag);
    /*! \brief
     *  Move constructor for SetCenter.
     */
    SetCenter(SetCenter&& old) noexcept : center_(old.center_), centerFlag_(old.centerFlag_) {}

    ~SetCenter() override {}

    /*! \brief
     * Change coordinate frame information for output.
     *
     * Takes the previously internally stored coordinates and saves them again.
     * Applies correct number of atoms in this case.
     *
     * \param[in] input Coordinate frame to be modified later.
     */
    void convertFrame(t_trxframe* input) override;

    //! What kind of guarantee the object gives depends on the user input.
    unsigned long guarantee() const override { return guarantee_; }

private:
    //! Pointer to selection of atoms for centering group.
    const Selection& center_;
    //! Stored flag on how to center system.
    CenteringType centerFlag_;
    //! What kind of guarantee does the object give.
    unsigned long guarantee_;
};

//! Smart pointer to manage the outputselector object.
using SetCenterPointer = std::unique_ptr<SetCenter>;

} // namespace gmx

#endif
