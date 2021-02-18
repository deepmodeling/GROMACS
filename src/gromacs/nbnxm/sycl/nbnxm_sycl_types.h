/*
 * This file is part of the GROMACS molecular simulation package.
 *
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

/*! \internal \file
 *  \brief
 *  Data types used internally in the nbnxm_sycl module.
 *
 *  \ingroup module_nbnxm
 */

#ifndef NBNXM_SYCL_TYPES_H
#define NBNXM_SYCL_TYPES_H

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/devicebuffer_sycl.h"
#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/gpu_utils/gpueventsynchronizer_sycl.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/gpu_utils/syclutils.h"
#include "gromacs/nbnxm/gpu_types_common.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/pairlist.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/utility/enumerationhelpers.h"

/*! \internal
 * \brief Staging area for temporary data downloaded from the GPU.
 *
 * Since SYCL buffers already have host-side storage, this is a bit redundant.
 * But it allows prefetching of the data from GPU, and brings GPU backends closer together.
 */
struct nb_staging_t
{
    //! LJ energy
    float* e_lj = nullptr;
    //! electrostatic energy
    float* e_el = nullptr;
    //! shift forces
    Float3* fshift = nullptr;
};

/** \internal
 * \brief Nonbonded atom data - both inputs and outputs.
 */
struct sycl_atomdata_t
{
    //! number of atoms
    int natoms;
    //! number of local atoms
    int natoms_local; //
    //! allocation size for the atom data (xq, f)
    int numAlloc;

    //! atom coordinates + charges, size \ref natoms
    DeviceBuffer<Float4> xq;
    //! force output array, size \ref natoms
    DeviceBuffer<Float3> f;

    //! LJ energy output, size 1
    DeviceBuffer<float> eLJ;
    //! Electrostatics energy input, size 1
    DeviceBuffer<float> eElec;

    //! shift forces
    DeviceBuffer<Float3> fShift;

    //! number of atom types
    int numTypes;
    //! atom type indices, size \ref natoms
    DeviceBuffer<int> atomTypes;
    //! sqrt(c6),sqrt(c12) size \ref natoms
    DeviceBuffer<Float2> ljComb;

    //! shifts
    DeviceBuffer<Float3> shiftVec;
    //! true if the shift vector has been uploaded
    bool shiftVecUploaded;
};

class GpuEventSynchronizer;

/*! \internal
 * \brief Main data structure for SYCL nonbonded force calculations.
 */
struct NbnxmGpu
{
    /*! \brief GPU device context.
     *
     * \todo Make it constant reference, once NbnxmGpu is a proper class.
     */
    const DeviceContext* deviceContext_;
    /*! \brief true if doing both local/non-local NB work on GPU */
    bool bUseTwoStreams = false;
    /*! \brief true indicates that the nonlocal_done event was enqueued */
    bool bNonLocalStreamActive = false;
    /*! \brief atom data */
    sycl_atomdata_t* atdat = nullptr;
    /*! \brief f buf ops cell index mapping */

    NBParamGpu* nbparam = nullptr;
    /*! \brief pair-list data structures (local and non-local) */
    gmx::EnumerationArray<Nbnxm::InteractionLocality, Nbnxm::gpu_plist*> plist = { { nullptr } };
    /*! \brief staging area where fshift/energies get downloaded. Will be removed in SYCL. */
    nb_staging_t nbst;
    /*! \brief local and non-local GPU streams */
    gmx::EnumerationArray<Nbnxm::InteractionLocality, const DeviceStream*> deviceStreams;

    /*! \brief True if event-based timing is enabled. Always false for SYCL. */
    bool bDoTime = false;
    /*! \brief Dummy timers. */
    Nbnxm::gpu_timers_t* timers = nullptr;
    /*! \brief Dummy timing data. */
    gmx_wallclock_gpu_nbnxn_t* timings = nullptr;

    //! true when a pair-list transfer has been done at this step
    gmx::EnumerationArray<Nbnxm::InteractionLocality, bool> didPairlistH2D = { { false } };
    //! true when we we did pruning on this step
    gmx::EnumerationArray<Nbnxm::InteractionLocality, bool> didPrune = { { false } };
    //! true when we did rolling pruning (at the previous step)
    gmx::EnumerationArray<Nbnxm::InteractionLocality, bool> didRollingPrune = { { false } };

    /*! \brief Events used for synchronization. Would be deprecated in SYCL. */
    /*! \{ */
    /*! \brief Event triggered when the non-local non-bonded
     * kernel is done (and the local transfer can proceed) */
    GpuEventSynchronizer nonlocal_done;
    /*! \brief Event triggered when the tasks issued in the local
     * stream that need to precede the non-local force or buffer
     * operation calculations are done (e.g. f buffer 0-ing, local
     * x/q H2D, buffer op initialization in local stream that is
     * required also by nonlocal stream ) */
    GpuEventSynchronizer misc_ops_and_local_H2D_done;
    /*! \} */

    /*! \brief True if there is work for the current domain in the
     * respective locality.
     *
     * This includes local/nonlocal GPU work, either bonded or
     * nonbonded, scheduled to be executed in the current
     * domain. As long as bonded work is not split up into
     * local/nonlocal, if there is bonded GPU work, both flags
     * will be true. */
    gmx::EnumerationArray<Nbnxm::InteractionLocality, bool> haveWork = { { false } };

    /*! \brief Pointer to event synchronizer triggered when the local
     * GPU buffer ops / reduction is complete. Would be deprecated in SYCL.
     *
     * \note That the synchronizer is managed outside of this module
     * in StatePropagatorDataGpu.
     */
    GpuEventSynchronizer* localFReductionDone = nullptr;

    /*! \brief Event triggered when non-local coordinate buffer
     * has been copied from device to host. Would be deprecated in SYCL. */
    GpuEventSynchronizer* xNonLocalCopyD2HDone = nullptr;
};

#endif /* NBNXM_SYCL_TYPES_H */
