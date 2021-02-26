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
/*! \internal \file
 * \brief Common functions for the different SITS GPU implementations.
 *
 * \author Junhan Chang <changjh@pku.edu.cn>
 *
 * \ingroup module_sits
 */

#include "gmxpre.h"

#include "gromacs/hardware/hw_info.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/sits/sits.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/logger.h"

namespace Sits
{

/*! \brief Resources that can be used to execute non-bonded kernels on */
enum class SitsResource : int
{
    Cpu,
    Gpu
};

std::unique_ptr<sits_t> init_sits(const gmx::MDLogger&     mdlog,
                                  gmx_bool                 bFEP_SITS,
                                  const t_inputrec*        ir,
                                  const t_forcerec*        fr,
                                  const t_commrec*         cr,
                                  const gmx_hw_info_t&     hardwareInfo,
                                  const gmx_device_info_t* deviceInfo,
                                  const gmx_mtop_t*        mtop,
                                  matrix                   box,
                                  gmx_wallcycle*           wcycle)
{
    const bool useGpu     = deviceInfo != nullptr;

    SitsResource sitsResource;
    if (useGpu)
    {
        sitsResource = SitsResource::Gpu;
    }
    else
    {
        sitsResource = SitsResource::Cpu;
    }

    // Nbnxm::KernelSetup kernelSetup = pick_nbnxn_kernel(mdlog, fr->use_simd_kernels, hardwareInfo,
    //                                                    nonbondedResource, ir, fr->bNonbonded);

    const bool haveMultipleDomains = (DOMAINDECOMP(cr) && cr->dd->nnodes > 1);

    auto pinPolicy = (useGpu ? gmx::PinningPolicy::PinnedIfSupported : gmx::PinningPolicy::CannotBePinned);

    auto nbat = std::make_unique<nbnxn_atomdata_t>(pinPolicy);

    int mimimumNumEnergyGroupNonbonded = ir->opts.ngener;
    if (ir->opts.ngener - ir->nwall == 1)
    {
        /* We have only one non-wall energy group, we do not need energy group
         * support in the non-bondeds kernels, since all non-bonded energy
         * contributions go to the first element of the energy group matrix.
         */
        mimimumNumEnergyGroupNonbonded = 1;
    }
    nbnxn_atomdata_init(mdlog, nbat.get(), kernelSetup.kernelType, enbnxninitcombrule, fr->ntype,
                        fr->nbfp, mimimumNumEnergyGroupNonbonded,
                        (useGpu || emulateGpu) ? 1 : gmx_omp_nthreads_get(emntNonbonded));

    gmx_sits_cuda_t* gpu_sits = nullptr;
    if (useGpu)
    {
        /* init the NxN GPU data; the last argument tells whether we'll have
         * both local and non-local NB calculation on GPU */
        gpu_sits = gpu_init(deviceInfo, fr->ic, pairlistParams, nbat.get(), cr->nodeid, haveMultipleDomains);
    }

    return std::make_unique<sits_t>(std::move(pairlistSets), std::move(pairSearch),
                                                std::move(nbat), kernelSetup, gpu_nbv, wcycle);
}

} // namespace Sits

sits_t::sits_t(std::unique_ptr<PairlistSets>     pairlistSets,
                                       std::unique_ptr<PairSearch>       pairSearch,
                                       std::unique_ptr<nbnxn_atomdata_t> nbat_in,
                                       const Nbnxm::KernelSetup&         kernelSetup,
                                       gmx_nbnxn_gpu_t*                  gpu_nbv_ptr,
                                       gmx_wallcycle*                    wcycle) :
    pairlistSets_(std::move(pairlistSets)),
    pairSearch_(std::move(pairSearch)),
    nbat(std::move(nbat_in)),
    kernelSetup_(kernelSetup),
    wcycle_(wcycle),
    gpu_nbv(gpu_nbv_ptr)
{
    GMX_RELEASE_ASSERT(pairlistSets_, "Need valid pairlistSets");
    GMX_RELEASE_ASSERT(pairSearch_, "Need valid search object");
    GMX_RELEASE_ASSERT(nbat, "Need valid atomdata object");
}

sits_t::~sits_t()
{
    Sits::gpu_free(gpu_sits);
}
