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
 *
 * \brief Declares the GPU Force Reduction
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_mdlib
 */
#ifndef GMX_MDLIB_GPUFORCEREDUCTION
#define GMX_MDLIB_GPUFORCEREDUCTION

#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/utility/fixedcapacityvector.h"

class GpuEventSynchronizer;

namespace gmx
{

/*! \brief Whether force data is in Rvec or Nbat format */
enum class ForceFormat
{
    Rvec, //!< Force data is in Rvec format
    Nbat, //!< Force data is in Nbat format
};

/*! \brief structure containing pointer to force data, and which format it is in */
struct GpuForceForReduction
{
    void*       forcePtr = nullptr;
    ForceFormat forceFormat;
};

typedef struct GpuForceForReduction GpuForceForReduction_t;

class GpuForceReduction
{

public:
    /*! \brief Creates GPU force reduction object
     *
     * \param [in] deviceContext GPU device context
     * \param [in] deviceStream  Stream to use for reduction
     */
    GpuForceReduction(const DeviceContext& deviceContext, const DeviceStream& deviceStream);
    ~GpuForceReduction();

    /*! \brief Set force to be used as a "base", i.e. to be reduced into
     *
     * \param [in] forceReductionBase   Force to be used as a base
     */
    void setBase(GpuForceForReduction_t forceReductionBase);

    /*! \brief Set the number of atoms for this reduction
     *
     * \param [in] numAtoms  The number of atoms
     */
    void setNumAtoms(const int numAtoms);

    /*! \brief Set the atom at which this reduction should start (i.e. the atom offset)
     *
     * \param [in] atomStart  The start atom for the reduction
     */
    void setAtomStart(const int atomStart);

    /*! \brief Register a force to be reduced
     *
     * \param [in] forceForReduction  Force to be reduced
     */
    void registerForce(const GpuForceForReduction_t forceForReduction);

    /*! \brief Add a dependency for this force reduction
     *
     * \param [in] dependency   Dependency for this reduction
     */
    void addDependency(GpuEventSynchronizer* const dependency);


    /*! \brief Whether this reduction should be accumulated (or set) into the base force buffer
     *
     * \param [in] accumulate   Whether reduction should be accumulated
     */
    void setAccumulate(const bool accumulate);

    /*! \brief Set the cell index mapping array for any nbat-format forces
     *
     * \param [in] cell   Pointer to the cell array
     */
    void setCell(const int* cell);

    /*! \brief Set an event to be marked when the launch of the reduction is completed
     *
     * \param [in] completionMarker   Event to be marked when launch of reduction is complete
     */
    void setCompletionMarker(GpuEventSynchronizer* completionMarker);

    /*! \brief Apply the force reduction */
    void apply();

    /*! \brief Clear all dependencies for the reduction */
    void clearDependencies();

    /*! \brief Remove the last force added to this reduction */
    void popForce();

private:
    class Impl;
    gmx::PrivateImplPointer<Impl> impl_;
};

} // namespace gmx

#endif
