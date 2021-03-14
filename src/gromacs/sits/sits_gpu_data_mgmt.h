/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2014,2015,2017,2018,2019, by the GROMACS development team, led by
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
/*! \libinternal \file
 *  \brief Declare interface for GPU data transfer for SITS module
 *
 *  \author Junhan Chang <changjh@pku.edu.cn>
 *  \ingroup module_sits
 *  \inlibraryapi
 */

#ifndef GMX_SITS_GPU_DATA_MGMT_H
#define GMX_SITS_GPU_DATA_MGMT_H

#include <memory>

#include "gromacs/gpu_utils/gpu_macros.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_types.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/sits/sits.h"
// #include "gromacs/sits/cuda/sits_cuda_types.h"

struct sits_atomdata_t;
struct nbnxn_atomdata_t;
struct gmx_sits_cuda_t;
// struct gmx_nbnxn_gpu_t;
struct gmx_gpu_info_t;
struct gmx_device_info_t;

namespace Sits
{

/** Initializes the data structures related to GPU sits calculations. */
GPU_FUNC_QUALIFIER
gmx_sits_cuda_t* gpu_init_sits(const gmx_device_info_t*   deviceInfo,
                                const sits_atomdata_t*     sits_at,
                                int gmx_unused rank) GPU_FUNC_TERM_WITH_RETURN(nullptr);

/** Initializes atom-data on the GPU, called at every pair search step. */
CUDA_FUNC_QUALIFIER
void gpu_init_sits_atomdata(gmx_sits_cuda_t gmx_unused* gpu_sits, const nbnxn_atomdata_t gmx_unused* nbat); CUDA_FUNC_TERM_WITH_RETURN(nullptr);

/** Clears GPU outputs: nonbonded force, shift force and energy. */
CUDA_FUNC_QUALIFIER
void sits_gpu_clear_outputs(gmx_sits_cuda_t gmx_unused* gpu_sits, bool gmx_unused computeVirial); CUDA_FUNC_TERM_WITH_RETURN(nullptr);

CUDA_FUNC_QUALIFIER
void gpu_print_sitsvals(gmx_sits_cuda_t gmx_unused* gpu_sits); CUDA_FUNC_TERM_WITH_RETURN(nullptr);

/** Frees all GPU resources used for the nonbonded calculations. */
CUDA_FUNC_QUALIFIER
void gpu_free(gmx_sits_cuda_t gmx_unused* gpu_sits); CUDA_FUNC_TERM_WITH_RETURN(nullptr);

/** Returns an opaque pointer to the GPU command stream
 *  Note: CUDA only.
 */
// CUDA_FUNC_QUALIFIER
// void* gpu_get_command_stream(gmx_nbnxn_gpu_t gmx_unused* nb, gmx::InteractionLocality gmx_unused iloc)
//         CUDA_FUNC_TERM_WITH_RETURN(nullptr);

/** Returns an opaque pointer to the GPU coordinate+charge array
 *  Note: CUDA only.
 */
// CUDA_FUNC_QUALIFIER
// void* gpu_get_xq(gmx_nbnxn_gpu_t gmx_unused* nb) CUDA_FUNC_TERM_WITH_RETURN(nullptr);

/** Returns an opaque pointer to the GPU force array
 *  Note: CUDA only.
 */
// CUDA_FUNC_QUALIFIER
// void* gpu_get_f(gmx_nbnxn_gpu_t gmx_unused* nb) CUDA_FUNC_TERM_WITH_RETURN(nullptr);

/** Returns an opaque pointer to the GPU shift force array
 *  Note: CUDA only.
 */
// CUDA_FUNC_QUALIFIER
// rvec* gpu_get_fshift(gmx_nbnxn_gpu_t gmx_unused* nb) CUDA_FUNC_TERM_WITH_RETURN(nullptr);

/*! \brief Force buffer operations on GPU.
 *
 * Transforms non-bonded forces into plain rvec format and add all the force components to the total
 * force buffer
 *
 * \param[in]   totalForcesDevice    Device buffer to accumulate resulting force.
 * \param[in]   gpu_nbv              The NBNXM GPU data structure.
 * \param[in]   gpu_sits             The SITS GPU data structure.
 * \param[in]   dependencyList       List of synchronizers that represent the dependencies the reduction task needs to sync on.
 * \param[in]   accumulateForce      Whether there are usefull data already in the total force buffer.
 *
 */
CUDA_FUNC_QUALIFIER
void nbnxn_gpu_add_sits_f_to_f(DeviceBuffer<float> gmx_unused totalForcesDevice,
                               gmx_nbnxn_gpu_t gmx_unused* gpu_nbv,
                               gmx_sits_cuda_t gmx_unused* gpu_sits,
                               gmx::ArrayRef<GpuEventSynchronizer* const> gmx_unused dependencyList,
                               bool gmx_unused accumulateForce) CUDA_FUNC_TERM;
} // namespace Sits

#endif
