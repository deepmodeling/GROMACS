/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2016 by the GROMACS development team.
 * Copyright (c) 2017,2018,2019,2020,2021, by the GROMACS development team, led by
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
 *  \brief Define OpenCL implementation of nbnxm_gpu_data_mgmt.h
 *
 *  \author Anca Hamuraru <anca@streamcomputing.eu>
 *  \author Dimitrios Karkoulis <dimitris.karkoulis@gmail.com>
 *  \author Teemu Virolainen <teemu@streamcomputing.eu>
 *  \author Szilárd Páll <pall.szilard@gmail.com>
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>

#include "gromacs/gpu_utils/device_stream_manager.h"
#include "gromacs/gpu_utils/oclutils.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/hardware/device_management.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/gpu_jit_support.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/nbnxm_gpu_data_mgmt.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"

#include "nbnxm_ocl_types.h"

namespace Nbnxm
{

/*! \brief Copies of values from cl_driver_diagnostics_intel.h,
 * which isn't guaranteed to be available. */
/**@{*/
#define CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL 0x4106
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL 0x1
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL 0x2
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL 0x4
/**@}*/

/*! \brief This parameter should be determined heuristically from the
 * kernel execution times
 *
 * This value is best for small systems on a single AMD Radeon R9 290X
 * (and about 5% faster than 40, which is the default for CUDA
 * devices). Larger simulation systems were quite insensitive to the
 * value of this parameter.
 */
static unsigned int gpu_min_ci_balanced_factor = 50;


/*! \brief Initializes the atomdata structure first time, it only gets filled at
    pair-search.
 */
static void init_atomdata_first(cl_atomdata_t* ad, int ntypes, const DeviceContext& deviceContext)
{
    ad->ntypes = ntypes;

    allocateDeviceBuffer(&ad->shift_vec, SHIFTS * DIM, deviceContext);
    ad->bShiftVecUploaded = CL_FALSE;

    allocateDeviceBuffer(&ad->fshift, SHIFTS * DIM, deviceContext);
    allocateDeviceBuffer(&ad->e_lj, 1, deviceContext);
    allocateDeviceBuffer(&ad->e_el, 1, deviceContext);

    /* initialize to nullptr pointers to data that is not allocated here and will
       need reallocation in nbnxn_gpu_init_atomdata */
    ad->xq = nullptr;
    ad->f  = nullptr;

    /* size -1 indicates that the respective array hasn't been initialized yet */
    ad->natoms = -1;
    ad->nalloc = -1;
}


/*! \brief Initializes the nonbonded parameter data structure.
 */
static void init_nbparam(NBParamGpu*                     nbp,
                         const interaction_const_t*      ic,
                         const PairlistParams&           listParams,
                         const nbnxn_atomdata_t::Params& nbatParams,
                         const DeviceContext&            deviceContext)
{
    set_cutoff_parameters(nbp, ic, listParams);

    nbp->vdwType  = nbnxmGpuPickVdwKernelType(ic, nbatParams.ljCombinationRule);
    nbp->elecType = nbnxmGpuPickElectrostaticsKernelType(ic, deviceContext.deviceInfo());

    if (ic->vdwtype == evdwPME)
    {
        if (ic->ljpme_comb_rule == eljpmeGEOM)
        {
            GMX_ASSERT(nbatParams.ljCombinationRule == LJCombinationRule::Geometric,
                       "Combination rule mismatch!");
        }
        else
        {
            GMX_ASSERT(nbatParams.ljCombinationRule == LJCombinationRule::LorentzBerthelot,
                       "Combination rule mismatch!");
        }
    }
    /* generate table for PME */
    nbp->coulomb_tab = nullptr;
    if (nbp->elecType == ElecType::EwaldTab || nbp->elecType == ElecType::EwaldTabTwin)
    {
        GMX_RELEASE_ASSERT(ic->coulombEwaldTables, "Need valid Coulomb Ewald correction tables");
        init_ewald_coulomb_force_table(*ic->coulombEwaldTables, nbp, deviceContext);
    }
    else
    {
        allocateDeviceBuffer(&nbp->coulomb_tab, 1, deviceContext);
    }

    const int nnbfp      = 2 * nbatParams.numTypes * nbatParams.numTypes;
    const int nnbfp_comb = 2 * nbatParams.numTypes;

    {
        /* set up LJ parameter lookup table */
        DeviceBuffer<real> nbfp;
        initParamLookupTable(&nbfp, nullptr, nbatParams.nbfp.data(), nnbfp, deviceContext);
        nbp->nbfp = nbfp;

        if (ic->vdwtype == evdwPME)
        {
            DeviceBuffer<float> nbfp_comb;
            initParamLookupTable(&nbfp_comb, nullptr, nbatParams.nbfp_comb.data(), nnbfp_comb, deviceContext);
            nbp->nbfp_comb = nbfp_comb;
        }
    }
}

/*! \brief Initializes the OpenCL kernel pointers of the nbnxn_ocl_ptr_t input data structure. */
static cl_kernel nbnxn_gpu_create_kernel(NbnxmGpu* nb, const char* kernel_name)
{
    cl_kernel kernel;
    cl_int    cl_error;

    kernel = clCreateKernel(nb->dev_rundata->program, kernel_name, &cl_error);
    if (CL_SUCCESS != cl_error)
    {
        gmx_fatal(FARGS,
                  "Failed to create kernel '%s' for GPU #%s: OpenCL error %d",
                  kernel_name,
                  nb->deviceContext_->deviceInfo().device_name,
                  cl_error);
    }

    return kernel;
}

/*! \brief Clears nonbonded shift force output array and energy outputs on the GPU.
 */
static void nbnxn_ocl_clear_e_fshift(NbnxmGpu* nb)
{

    cl_int           cl_error;
    cl_atomdata_t*   adat = nb->atdat;
    cl_command_queue ls   = nb->deviceStreams[InteractionLocality::Local]->stream();

    size_t local_work_size[3]  = { 1, 1, 1 };
    size_t global_work_size[3] = { 1, 1, 1 };

    cl_int shifts = SHIFTS * 3;

    cl_int arg_no;

    cl_kernel zero_e_fshift = nb->kernel_zero_e_fshift;

    local_work_size[0] = 64;
    // Round the total number of threads up from the array size
    global_work_size[0] = ((shifts + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];

    arg_no   = 0;
    cl_error = clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_mem), &(adat->fshift));
    cl_error |= clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_mem), &(adat->e_lj));
    cl_error |= clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_mem), &(adat->e_el));
    cl_error |= clSetKernelArg(zero_e_fshift, arg_no++, sizeof(cl_uint), &shifts);
    GMX_ASSERT(cl_error == CL_SUCCESS, ocl_get_error_string(cl_error).c_str());

    cl_error = clEnqueueNDRangeKernel(
            ls, zero_e_fshift, 3, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
    GMX_ASSERT(cl_error == CL_SUCCESS, ocl_get_error_string(cl_error).c_str());
}

/*! \brief Initializes the OpenCL kernel pointers of the nbnxn_ocl_ptr_t input data structure. */
static void nbnxn_gpu_init_kernels(NbnxmGpu* nb)
{
    /* Init to 0 main kernel arrays */
    /* They will be later on initialized in select_nbnxn_kernel */
    // TODO: consider always creating all variants of the kernels here so that there is no
    // need for late call to clCreateKernel -- if that gives any advantage?
    memset(nb->kernel_ener_noprune_ptr, 0, sizeof(nb->kernel_ener_noprune_ptr));
    memset(nb->kernel_ener_prune_ptr, 0, sizeof(nb->kernel_ener_prune_ptr));
    memset(nb->kernel_noener_noprune_ptr, 0, sizeof(nb->kernel_noener_noprune_ptr));
    memset(nb->kernel_noener_prune_ptr, 0, sizeof(nb->kernel_noener_prune_ptr));

    /* Init pruning kernels
     *
     * TODO: we could avoid creating kernels if dynamic pruning is turned off,
     * but ATM that depends on force flags not passed into the initialization.
     */
    nb->kernel_pruneonly[epruneFirst] = nbnxn_gpu_create_kernel(nb, "nbnxn_kernel_prune_opencl");
    nb->kernel_pruneonly[epruneRolling] =
            nbnxn_gpu_create_kernel(nb, "nbnxn_kernel_prune_rolling_opencl");

    /* Init auxiliary kernels */
    nb->kernel_zero_e_fshift = nbnxn_gpu_create_kernel(nb, "zero_e_fshift");
}

/*! \brief Initializes simulation constant data.
 *
 *  Initializes members of the atomdata and nbparam structs and
 *  clears e/fshift output buffers.
 */
static void nbnxn_ocl_init_const(cl_atomdata_t*                  atomData,
                                 NBParamGpu*                     nbParams,
                                 const interaction_const_t*      ic,
                                 const PairlistParams&           listParams,
                                 const nbnxn_atomdata_t::Params& nbatParams,
                                 const DeviceContext&            deviceContext)
{
    init_atomdata_first(atomData, nbatParams.numTypes, deviceContext);
    init_nbparam(nbParams, ic, listParams, nbatParams, deviceContext);
}


//! This function is documented in the header file
NbnxmGpu* gpu_init(const gmx::DeviceStreamManager& deviceStreamManager,
                   const interaction_const_t*      ic,
                   const PairlistParams&           listParams,
                   const nbnxn_atomdata_t*         nbat,
                   const bool                      bLocalAndNonlocal)
{
    GMX_ASSERT(ic, "Need a valid interaction constants object");

    auto nb            = new NbnxmGpu();
    nb->deviceContext_ = &deviceStreamManager.context();
    snew(nb->atdat, 1);
    snew(nb->nbparam, 1);
    snew(nb->plist[InteractionLocality::Local], 1);
    if (bLocalAndNonlocal)
    {
        snew(nb->plist[InteractionLocality::NonLocal], 1);
    }

    nb->bUseTwoStreams = bLocalAndNonlocal;

    nb->timers = new cl_timers_t();
    snew(nb->timings, 1);

    /* set device info, just point it to the right GPU among the detected ones */
    nb->dev_rundata = new gmx_device_runtime_data_t();

    /* init nbst */
    pmalloc(reinterpret_cast<void**>(&nb->nbst.e_lj), sizeof(*nb->nbst.e_lj));
    pmalloc(reinterpret_cast<void**>(&nb->nbst.e_el), sizeof(*nb->nbst.e_el));
    pmalloc(reinterpret_cast<void**>(&nb->nbst.fshift), SHIFTS * sizeof(*nb->nbst.fshift));

    init_plist(nb->plist[InteractionLocality::Local]);

    /* OpenCL timing disabled if GMX_DISABLE_GPU_TIMING is defined. */
    nb->bDoTime = (getenv("GMX_DISABLE_GPU_TIMING") == nullptr);

    /* local/non-local GPU streams */
    GMX_RELEASE_ASSERT(deviceStreamManager.streamIsValid(gmx::DeviceStreamType::NonBondedLocal),
                       "Local non-bonded stream should be initialized to use GPU for non-bonded.");
    nb->deviceStreams[InteractionLocality::Local] =
            &deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal);

    if (nb->bUseTwoStreams)
    {
        init_plist(nb->plist[InteractionLocality::NonLocal]);

        GMX_RELEASE_ASSERT(deviceStreamManager.streamIsValid(gmx::DeviceStreamType::NonBondedNonLocal),
                           "Non-local non-bonded stream should be initialized to use GPU for "
                           "non-bonded with domain decomposition.");
        nb->deviceStreams[InteractionLocality::NonLocal] =
                &deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedNonLocal);
    }

    if (nb->bDoTime)
    {
        init_timings(nb->timings);
    }

    nbnxn_ocl_init_const(nb->atdat, nb->nbparam, ic, listParams, nbat->params(), *nb->deviceContext_);

    /* Enable LJ param manual prefetch for AMD or Intel or if we request through env. var.
     * TODO: decide about NVIDIA
     */
    nb->bPrefetchLjParam = (getenv("GMX_OCL_DISABLE_I_PREFETCH") == nullptr)
                           && ((nb->deviceContext_->deviceInfo().deviceVendor == DeviceVendor::Amd)
                               || (nb->deviceContext_->deviceInfo().deviceVendor == DeviceVendor::Intel)
                               || (getenv("GMX_OCL_ENABLE_I_PREFETCH") != nullptr));

    /* NOTE: in CUDA we pick L1 cache configuration for the nbnxn kernels here,
     * but sadly this is not supported in OpenCL (yet?). Consider adding it if
     * it becomes supported.
     */
    nbnxn_gpu_compile_kernels(nb);
    nbnxn_gpu_init_kernels(nb);

    /* clear energy and shift force outputs */
    nbnxn_ocl_clear_e_fshift(nb);

    if (debug)
    {
        fprintf(debug, "Initialized OpenCL data structures.\n");
    }

    return nb;
}

/*! \brief Clears the first natoms_clear elements of the GPU nonbonded force output array.
 */
static void nbnxn_ocl_clear_f(NbnxmGpu* nb, int natoms_clear)
{
    if (natoms_clear == 0)
    {
        return;
    }

    cl_atomdata_t*      atomData    = nb->atdat;
    const DeviceStream& localStream = *nb->deviceStreams[InteractionLocality::Local];

    clearDeviceBufferAsync(&atomData->f, 0, natoms_clear, localStream);
}

//! This function is documented in the header file
void gpu_clear_outputs(NbnxmGpu* nb, bool computeVirial)
{
    nbnxn_ocl_clear_f(nb, nb->atdat->natoms);
    /* clear shift force array and energies if the outputs were
       used in the current step */
    if (computeVirial)
    {
        nbnxn_ocl_clear_e_fshift(nb);
    }

    /* kick off buffer clearing kernel to ensure concurrency with constraints/update */
    cl_int gmx_unused cl_error;
    cl_error = clFlush(nb->deviceStreams[InteractionLocality::Local]->stream());
    GMX_ASSERT(cl_error == CL_SUCCESS, ("clFlush failed: " + ocl_get_error_string(cl_error)).c_str());
}

//! This function is documented in the header file
void gpu_upload_shiftvec(NbnxmGpu* nb, const nbnxn_atomdata_t* nbatom)
{
    cl_atomdata_t*      adat        = nb->atdat;
    const DeviceStream& localStream = *nb->deviceStreams[InteractionLocality::Local];

    /* only if we have a dynamic box */
    if (nbatom->bDynamicBox || !adat->bShiftVecUploaded)
    {
        static_assert(sizeof(Float3) == sizeof(nbatom->shift_vec[0]),
                      "Sizes of host- and device-side shift vectors should be the same.");
        copyToDeviceBuffer(&adat->shift_vec,
                           reinterpret_cast<const Float3*>(nbatom->shift_vec.data()),
                           0,
                           SHIFTS,
                           localStream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        adat->bShiftVecUploaded = CL_TRUE;
    }
}

//! This function is documented in the header file
void gpu_init_atomdata(NbnxmGpu* nb, const nbnxn_atomdata_t* nbat)
{
    cl_int               cl_error;
    int                  nalloc, natoms;
    bool                 realloced;
    bool                 bDoTime       = nb->bDoTime;
    cl_timers_t*         timers        = nb->timers;
    cl_atomdata_t*       d_atdat       = nb->atdat;
    const DeviceContext& deviceContext = *nb->deviceContext_;
    const DeviceStream&  localStream   = *nb->deviceStreams[InteractionLocality::Local];

    natoms    = nbat->numAtoms();
    realloced = false;

    if (bDoTime)
    {
        /* time async copy */
        timers->atdat.openTimingRegion(localStream);
    }

    /* need to reallocate if we have to copy more atoms than the amount of space
       available and only allocate if we haven't initialized yet, i.e d_atdat->natoms == -1 */
    if (natoms > d_atdat->nalloc)
    {
        nalloc = over_alloc_small(natoms);

        /* free up first if the arrays have already been initialized */
        if (d_atdat->nalloc != -1)
        {
            freeDeviceBuffer(&d_atdat->f);
            freeDeviceBuffer(&d_atdat->xq);
            freeDeviceBuffer(&d_atdat->lj_comb);
            freeDeviceBuffer(&d_atdat->atom_types);
        }


        allocateDeviceBuffer(&d_atdat->f, nalloc, deviceContext);
        allocateDeviceBuffer(&d_atdat->xq, nalloc, deviceContext);

        if (useLjCombRule(nb->nbparam->vdwType))
        {
            // Two Lennard-Jones parameters per atom
            allocateDeviceBuffer(&d_atdat->lj_comb, nalloc, deviceContext);
        }
        else
        {
            allocateDeviceBuffer(&d_atdat->atom_types, nalloc, deviceContext);
        }

        d_atdat->nalloc = nalloc;
        realloced       = true;
    }

    d_atdat->natoms       = natoms;
    d_atdat->natoms_local = nbat->natoms_local;

    /* need to clear GPU f output if realloc happened */
    if (realloced)
    {
        nbnxn_ocl_clear_f(nb, nalloc);
    }

    if (useLjCombRule(nb->nbparam->vdwType))
    {
        static_assert(sizeof(float) == sizeof(*nbat->params().lj_comb.data()),
                      "Size of the LJ parameters element should be equal to the size of float2.");
        copyToDeviceBuffer(&d_atdat->lj_comb,
                           reinterpret_cast<const Float2*>(nbat->params().lj_comb.data()),
                           0,
                           natoms,
                           localStream,
                           GpuApiCallBehavior::Async,
                           bDoTime ? timers->atdat.fetchNextEvent() : nullptr);
    }
    else
    {
        static_assert(sizeof(int) == sizeof(*nbat->params().type.data()),
                      "Sizes of host- and device-side atom types should be the same.");
        copyToDeviceBuffer(&d_atdat->atom_types,
                           nbat->params().type.data(),
                           0,
                           natoms,
                           localStream,
                           GpuApiCallBehavior::Async,
                           bDoTime ? timers->atdat.fetchNextEvent() : nullptr);
    }

    if (bDoTime)
    {
        timers->atdat.closeTimingRegion(localStream);
    }

    /* kick off the tasks enqueued above to ensure concurrency with the search */
    cl_error = clFlush(localStream.stream());
    GMX_RELEASE_ASSERT(cl_error == CL_SUCCESS,
                       ("clFlush failed: " + ocl_get_error_string(cl_error)).c_str());
}

/*! \brief Releases an OpenCL kernel pointer */
static void free_kernel(cl_kernel* kernel_ptr)
{
    cl_int gmx_unused cl_error;

    GMX_ASSERT(kernel_ptr, "Need a valid kernel pointer");

    if (*kernel_ptr)
    {
        cl_error = clReleaseKernel(*kernel_ptr);
        GMX_RELEASE_ASSERT(cl_error == CL_SUCCESS,
                           ("clReleaseKernel failed: " + ocl_get_error_string(cl_error)).c_str());

        *kernel_ptr = nullptr;
    }
}

/*! \brief Releases a list of OpenCL kernel pointers */
static void free_kernels(cl_kernel* kernels, int count)
{
    int i;

    for (i = 0; i < count; i++)
    {
        free_kernel(kernels + i);
    }
}

/*! \brief Free the OpenCL program.
 *
 *  The function releases the OpenCL program assuciated with the
 *  device that the calling PP rank is running on.
 *
 *  \param program [in]  OpenCL program to release.
 */
static void freeGpuProgram(cl_program program)
{
    if (program)
    {
        cl_int cl_error = clReleaseProgram(program);
        GMX_RELEASE_ASSERT(cl_error == CL_SUCCESS,
                           ("clReleaseProgram failed: " + ocl_get_error_string(cl_error)).c_str());
        program = nullptr;
    }
}

//! This function is documented in the header file
void gpu_free(NbnxmGpu* nb)
{
    if (nb == nullptr)
    {
        return;
    }

    /* Free kernels */
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    int kernel_count = sizeof(nb->kernel_ener_noprune_ptr) / sizeof(nb->kernel_ener_noprune_ptr[0][0]);
    free_kernels(nb->kernel_ener_noprune_ptr[0], kernel_count);

    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    kernel_count = sizeof(nb->kernel_ener_prune_ptr) / sizeof(nb->kernel_ener_prune_ptr[0][0]);
    free_kernels(nb->kernel_ener_prune_ptr[0], kernel_count);

    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    kernel_count = sizeof(nb->kernel_noener_noprune_ptr) / sizeof(nb->kernel_noener_noprune_ptr[0][0]);
    free_kernels(nb->kernel_noener_noprune_ptr[0], kernel_count);

    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    kernel_count = sizeof(nb->kernel_noener_prune_ptr) / sizeof(nb->kernel_noener_prune_ptr[0][0]);
    free_kernels(nb->kernel_noener_prune_ptr[0], kernel_count);

    free_kernel(&(nb->kernel_zero_e_fshift));

    /* Free atdat */
    freeDeviceBuffer(&(nb->atdat->xq));
    freeDeviceBuffer(&(nb->atdat->f));
    freeDeviceBuffer(&(nb->atdat->e_lj));
    freeDeviceBuffer(&(nb->atdat->e_el));
    freeDeviceBuffer(&(nb->atdat->fshift));
    freeDeviceBuffer(&(nb->atdat->lj_comb));
    freeDeviceBuffer(&(nb->atdat->atom_types));
    freeDeviceBuffer(&(nb->atdat->shift_vec));
    sfree(nb->atdat);

    /* Free nbparam */
    freeDeviceBuffer(&(nb->nbparam->nbfp));
    freeDeviceBuffer(&(nb->nbparam->nbfp_comb));
    freeDeviceBuffer(&(nb->nbparam->coulomb_tab));
    sfree(nb->nbparam);

    /* Free plist */
    auto* plist = nb->plist[InteractionLocality::Local];
    freeDeviceBuffer(&plist->sci);
    freeDeviceBuffer(&plist->cj4);
    freeDeviceBuffer(&plist->imask);
    freeDeviceBuffer(&plist->excl);
    sfree(plist);
    if (nb->bUseTwoStreams)
    {
        auto* plist_nl = nb->plist[InteractionLocality::NonLocal];
        freeDeviceBuffer(&plist_nl->sci);
        freeDeviceBuffer(&plist_nl->cj4);
        freeDeviceBuffer(&plist_nl->imask);
        freeDeviceBuffer(&plist_nl->excl);
        sfree(plist_nl);
    }

    /* Free nbst */
    pfree(nb->nbst.e_lj);
    nb->nbst.e_lj = nullptr;

    pfree(nb->nbst.e_el);
    nb->nbst.e_el = nullptr;

    pfree(nb->nbst.fshift);
    nb->nbst.fshift = nullptr;

    freeGpuProgram(nb->dev_rundata->program);
    delete nb->dev_rundata;

    /* Free timers and timings */
    delete nb->timers;
    sfree(nb->timings);
    delete nb;

    if (debug)
    {
        fprintf(debug, "Cleaned up OpenCL data structures.\n");
    }
}

//! This function is documented in the header file
int gpu_min_ci_balanced(NbnxmGpu* nb)
{
    return nb != nullptr ? gpu_min_ci_balanced_factor * nb->deviceContext_->deviceInfo().compute_units : 0;
}

} // namespace Nbnxm
