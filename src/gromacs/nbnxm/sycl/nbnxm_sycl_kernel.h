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
 * Declares nbnxm SYCL kernel functor, launcher and factory of launchers.
 *
 * \ingroup module_nbnxm
 */
#ifndef GMX_NBNXM_SYCL_KERNEL_H
#define GMX_NBNXM_SYCL_KERNEL_H

#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/nbnxm/gpu_types_common.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"

#include "nbnxm_sycl_types.h"

//! \internal Kernel params
struct NbnxmSyclKernelParams
{
    NbnxmSyclKernelParams(sycl_atomdata* atomdata_, NBParamGpu* nbparam_, Nbnxm::gpu_plist* plist_, bool bCalcFshift_) :
        atomdata(atomdata_),
        nbparam(nbparam_),
        plist(plist_),
        bCalcFshift(bCalcFshift_)
    {
    }
    sycl_atomdata*    atomdata;
    NBParamGpu*       nbparam;
    Nbnxm::gpu_plist* plist;
    bool              bCalcFshift;
};

class INbnxmSyclKernelLauncher
{
public:
    virtual ~INbnxmSyclKernelLauncher()                                      = default;
    virtual cl::sycl::event launch(const struct KernelLaunchConfig& config,
                                   const DeviceStream&              deviceStream,
                                   CommandEvent gmx_unused*            timingEvent,
                                   const struct NbnxmSyclKernelParams& args) = 0;
};

template<bool doPruneNBL, bool doCalcEnergies, enum eelType flavorEL, enum evdwType flavorLJ>
class NbnxmSyclKernelLauncher : public INbnxmSyclKernelLauncher
{
    cl::sycl::event launch(const struct KernelLaunchConfig& config,
                           const DeviceStream&              deviceStream,
                           CommandEvent gmx_unused*            timingEvent,
                           const struct NbnxmSyclKernelParams& args) final;
};

INbnxmSyclKernelLauncher* getNbnxmSyclKernelLauncher(bool          doPruneNBL,
                                                     bool          doCalcEnergies,
                                                     enum eelType  flavorEL,
                                                     enum evdwType flavorLJ);

#endif // GROMACS_NBNXM_SYCL_KERNEL_H
