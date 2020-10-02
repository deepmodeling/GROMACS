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
 *  \brief
 *  Stubs of functions that must be defined by nbnxm sycl implementation.
 *
 *  \ingroup module_nbnxm
 */

#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/utility/exceptions.h"

#include "nbnxm_sycl_types.h"

namespace Nbnxm
{

//! This function is documented in the header file
void gpu_clear_outputs(NbnxmGpu* /*nb*/, bool /*computeVirial*/)
{
    GMX_THROW(gmx::NotImplementedError("Not implemented on SYCL yet"));
}

//! This function is documented in the header file
NbnxmGpu* gpu_init(const gmx::DeviceStreamManager& /*deviceStreamManager*/,
                   const interaction_const_t* /*ic*/,
                   const PairlistParams& /*listParams*/,
                   const nbnxn_atomdata_t* /*nbat*/,
                   const bool /*bLocalAndNonlocal*/)
{
    GMX_THROW(gmx::NotImplementedError("Not implemented on SYCL yet"));
}

//! This function is documented in the header file
void gpu_upload_shiftvec(NbnxmGpu* /*nb*/, const nbnxn_atomdata_t* /*nbatom*/)
{
    GMX_THROW(gmx::NotImplementedError("Not implemented on SYCL yet"));
}

//! This function is documented in the header file
void gpu_init_atomdata(NbnxmGpu* /*nb*/, const nbnxn_atomdata_t* /*nbat*/)
{
    GMX_THROW(gmx::NotImplementedError("Not implemented on SYCL yet"));
}

//! This function is documented in the header file
void gpu_free(NbnxmGpu* nb)
{
    if (nb == nullptr)
    {
        return;
    }
    GMX_THROW(gmx::NotImplementedError("Not implemented on SYCL yet"));
}

//! This function is documented in the header file
int gpu_min_ci_balanced(NbnxmGpu* /*nb*/)
{
    GMX_THROW(gmx::NotImplementedError("Not implemented on SYCL yet"));
}

} // namespace Nbnxm
