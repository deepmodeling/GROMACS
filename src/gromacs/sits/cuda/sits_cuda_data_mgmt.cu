/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2016,2017,2018,2019, by the GROMACS development team, led by
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
 *  \brief Define CUDA implementation of nbnxn_gpu_data_mgmt.h
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 */
#include "gmxpre.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// TODO We would like to move this down, but the way gmx_nbnxn_gpu_t
//      is currently declared means this has to be before gpu_types.h
#include "nbnxm_cuda_types.h"

// TODO Remove this comment when the above order issue is resolved
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.cuh"
#include "gromacs/gpu_utils/pmalloc_cuda.h"
#include "gromacs/hardware/gpu_hw_info.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/gridset.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"

#include "nbnxm_cuda.h"

namespace Sits
{

/* Fw. decl. */
static void sits_cuda_clear_e_fshift(gmx_sits_cuda_t* gpu_sits);

/*! Initializes the atomdata structure first time, it only gets filled at
    pair-search. */
static void sits_init_atomdata_first(cu_sits_atdat_t* atdat)
{
    cudaError_t stat;

    stat = cudaMalloc((void**)&ad->fshift, SHIFTS * sizeof(*ad->fshift));
    CU_RET_ERR(stat, "cudaMalloc failed on ad->fshift");

    stat = cudaMalloc((void**)&ad->e_lj, sizeof(*ad->e_lj));
    CU_RET_ERR(stat, "cudaMalloc failed on ad->e_lj");
    stat = cudaMalloc((void**)&ad->e_el, sizeof(*ad->e_el));
    CU_RET_ERR(stat, "cudaMalloc failed on ad->e_el");

    /* initialize to nullptr pointers to data that is not allocated here and will
       need reallocation in sits_cuda_init_atomdata */
    atdat->d_force_tot = nullptr;
    atdat->d_force_pw  = nullptr;
    atdat->d_force_tot_nbat = nullptr;
    atdat->d_force_pw_nbat  = nullptr;

    /* size -1 indicates that the respective array hasn't been initialized yet */
    atdat->natoms = -1;
    atdat->nalloc = -1;
}

/*! Initializes simulation constant data. */
static void cuda_init_sits_params(gmx_sits_cuda_t*           gpu_sits,
                                  const sits_atomdata_t*     sits_at)
{
    cu_sits_param_t* param = gpu_sits->sits_param;
    cudaStream_t stream    = gpu_sits->stream;

    sits_init_atomdata_first(gpu_sits->sits_atdat);

    // SITS ensemble definition
    param->record_interval = sits_at->record_interval;   // interval of energy record
    param->update_interval = sits_at->update_interval; // interval of $n_k$ update
    param->constant_nk     = sits_at->constant_nk;   // whether iteratively update n_k
    param->k_numbers       = sits_at->k_numbers;
    param->beta0           = sits_at->beta0;

    //计算时，可以对fc_ball直接修正，+ fb_shift进行调节，
    param->fb_shift        = sits_at->fb_shift;
    // energy record modifications: energy_record = energy_multiple * U + energy_shift;
    param->energy_multiple = sits_at->energy_multiple;
    param->energy_shift    = sits_at->energy_shift;

    // Derivations and physical quantities see:
    // \ref A selective integrated tempering method
    // \ref Self-adaptive enhanced sampling in the energy and trajectory spaces : Accelerated thermodynamics and kinetic calculations

    DeviceContext context = nullptr;

    param->k_nalloc = 0;
    reallocateDeviceBuffer(&param->beta_k, sits_at->k_numbers, &param->k_numbers, &param->k_nalloc, context);
    copyToDeviceBuffer(&param->beta_k, sits_at->beta_k.data(), 0, sits_at->k_numbers, stream,
                       GpuApiCallBehavior::Async, nullptr);
    
    param->k_nalloc = 0;
    reallocateDeviceBuffer(&param->nkExpBetakU, sits_at->k_numbers, &param->k_numbers, &param->k_nalloc, context);
    copyToDeviceBuffer(&param->nkExpBetakU, sits_at->nkExpBetakU.data(), 0, sits_at->k_numbers, stream,
                       GpuApiCallBehavior::Async, nullptr);
    
    param->k_nalloc = 0;
    reallocateDeviceBuffer(&param->nk, sits_at->k_numbers, &param->k_numbers, &param->k_nalloc, context);
    copyToDeviceBuffer(&param->nk, sits_at->nk.data(), 0, sits_at->k_numbers, stream,
                       GpuApiCallBehavior::Async, nullptr);
    
    stat = cudaMalloc((void**)&param->sum_a, sizeof(*param->sum_a));
    CU_RET_ERR(stat, "cudaMalloc failed on param->sum_a");
    stat = cudaMalloc((void**)&param->sum_b, sizeof(*param->sum_b));
    CU_RET_ERR(stat, "cudaMalloc failed on param->sum_b");
    stat = cudaMalloc((void**)&param->factor, 2 * sizeof(*param->sum_b));
    CU_RET_ERR(stat, "cudaMalloc failed on param->factor");

    // Details of $n_k$ iteration see:
    // \ref An integrate-over-temperature approach for enhanced sampling

    // |   .cpp var    |  ylj .F90 var  |  Ref var
    // | ene_recorded  | vshift         | U  
    // | gf            | gf             | log( n_k * exp(-beta_k * U) )
    // | gfsum         | gfsum          | log( Sum_(k=1)^N ( log( n_k * exp(-beta_k * U) ) ) )
    // | log_weight    | rb             | log of the weighting function
    // | log_mk_inv    | ratio          | log(m_k^-1)
    // | log_norm_old  | normlold       | W(j-1)
    // | log_norm      | norml          | W(j)
    // | log_pk        | rbfb           | log(p_k)
    // | log_nk_inv    | pratio         | log(n_k^-1)
    // | log_nk        | fb             | log(n_k)

    stat = cudaMalloc((void**)&param->ene_recorded, sizeof(*param->ene_recorded));
    CU_RET_ERR(stat, "cudaMalloc failed on param->ene_recorded");
    stat = cudaMalloc((void**)&param->gfsum, sizeof(*param->gfsum));
    CU_RET_ERR(stat, "cudaMalloc failed on param->gfsum");

    param->k_nalloc = 0;
    reallocateDeviceBuffer(&param->gf, sits_at->k_numbers, &param->k_numbers, &param->k_nalloc, context);
    copyToDeviceBuffer(&param->gf, sits_at->gf.data(), 0, sits_at->k_numbers, stream,
                       GpuApiCallBehavior::Async, nullptr);
    
    param->k_nalloc = 0;
    reallocateDeviceBuffer(&param->log_weight, sits_at->k_numbers, &param->k_numbers, &param->k_nalloc, context);
    copyToDeviceBuffer(&param->log_weight, sits_at->log_weight.data(), 0, sits_at->k_numbers, stream,
                       GpuApiCallBehavior::Async, nullptr);
    
    param->k_nalloc = 0;
    reallocateDeviceBuffer(&param->log_mk_inv, sits_at->k_numbers, &param->k_numbers, &param->k_nalloc, context);
    copyToDeviceBuffer(&param->log_mk_inv, sits_at->log_mk_inv.data(), 0, sits_at->k_numbers, stream,
                       GpuApiCallBehavior::Async, nullptr);
    
    param->k_nalloc = 0;
    reallocateDeviceBuffer(&param->log_norm_old, sits_at->k_numbers, &param->k_numbers, &param->k_nalloc, context);
    copyToDeviceBuffer(&param->log_norm_old, sits_at->log_norm_old.data(), 0, sits_at->k_numbers, stream,
                       GpuApiCallBehavior::Async, nullptr);
    
    param->k_nalloc = 0;
    reallocateDeviceBuffer(&param->log_norm, sits_at->k_numbers, &param->k_numbers, &param->k_nalloc, context);
    copyToDeviceBuffer(&param->log_norm, sits_at->log_norm.data(), 0, sits_at->k_numbers, stream,
                       GpuApiCallBehavior::Async, nullptr);
    
    param->k_nalloc = 0;
    reallocateDeviceBuffer(&param->log_pk, sits_at->k_numbers, &param->k_numbers, &param->k_nalloc, context);
    copyToDeviceBuffer(&param->log_pk, sits_at->log_pk.data(), 0, sits_at->k_numbers, stream,
                       GpuApiCallBehavior::Async, nullptr);
    
    param->k_nalloc = 0;
    reallocateDeviceBuffer(&param->log_nk_inv, sits_at->k_numbers, &param->k_numbers, &param->k_nalloc, context);
    copyToDeviceBuffer(&param->log_nk_inv, sits_at->log_nk_inv.data(), 0, sits_at->k_numbers, stream,
                       GpuApiCallBehavior::Async, nullptr);
    
    param->k_nalloc = 0;
    reallocateDeviceBuffer(&param->log_nk, sits_at->k_numbers, &param->k_numbers, &param->k_nalloc, context);
    copyToDeviceBuffer(&param->log_nk, sits_at->log_nk.data(), 0, sits_at->k_numbers, stream,
                       GpuApiCallBehavior::Async, nullptr);

    /* clear energy and shift force outputs */
    sits_cuda_clear_e_fshift(gpu_sits);
}

gmx_sits_cuda_t* gpu_init_sits(const gmx_device_info_t*   deviceInfo,
                                const sits_atomdata_t*     sits_at,
                                int /*rank*/)
{
    cudaError_t stat;

    gmx_sits_cuda_t* gpu_sits;
    snew(gpu_sits, 1);
    snew(gpu_sits->sits_atdat, 1);
    snew(gpu_sits->sits_param, 1);

    /* init nbst */
    // pmalloc((void**)&nb->nbst.e_lj, sizeof(*nb->nbst.e_lj));
    // pmalloc((void**)&nb->nbst.e_el, sizeof(*nb->nbst.e_el));
    // pmalloc((void**)&nb->nbst.fshift, SHIFTS * sizeof(*nb->nbst.fshift));

    /* set device info, just point it to the right GPU among the detected ones */
    gpu_sits->dev_info = deviceInfo;

    /* local/non-local GPU streams */
    stat = cudaStreamCreate(&gpu_sits->stream);

    /* set the kernel type for the current GPU */
    /* pick L1 cache configuration */
    cuda_set_cacheconfig();

    cuda_init_sits_params(gpu_sits, sits_at);

    nb->atomIndicesSize       = 0;
    nb->atomIndicesSize_alloc = 0;

    if (debug)
    {
        fprintf(debug, "Initialized SITS CUDA data structures.\n");
    }

    return gpu_sits;
}

/*! Clears the first natoms_clear elements of the GPU nonbonded force output array. */
static void sits_cuda_clear_f(gmx_sits_cuda_t* gpu_sits, int natoms_clear)
{
    cudaError_t    stat;
    cu_sits_atdat_t* adat = gpu_sits->sits_atdat;
    cudaStream_t   ls   = nb->stream[InteractionLocality::Local];

    stat = cudaMemsetAsync(adat->d_force_tot, 0, natoms_clear * sizeof(*adat->d_force_tot), ls);
    CU_RET_ERR(stat, "cudaMemsetAsync on f failed");
    stat = cudaMemsetAsync(adat->d_force_pw, 0, natoms_clear * sizeof(*adat->d_force_pw), ls);
    CU_RET_ERR(stat, "cudaMemsetAsync on f failed");
    stat = cudaMemsetAsync(adat->d_force_tot_nbat, 0, natoms_clear * sizeof(*adat->d_force_tot_nbat), ls);
    CU_RET_ERR(stat, "cudaMemsetAsync on f failed");
    stat = cudaMemsetAsync(adat->d_force_pw_nbat, 0, natoms_clear * sizeof(*adat->d_force_pw_nbat), ls);
    CU_RET_ERR(stat, "cudaMemsetAsync on f failed");
}

/*! Clears nonbonded shift force output array and energy outputs on the GPU. */
static void sits_cuda_clear_e_fshift(gmx_sits_cuda_t* gpu_sits)
{
    cudaError_t    stat;
    cu_atomdata_t* adat = gpu_sits->sits_atdat;
    cudaStream_t   ls   = gpu_sits->stream;

    stat = cudaMemsetAsync(adat->d_enerd, 0, sizeof(*adat->d_enerd), ls);
    CU_RET_ERR(stat, "cudaMemsetAsync on enerd failed");
    // stat = cudaMemsetAsync(adat->e_lj, 0, sizeof(*adat->e_lj), ls);
    // CU_RET_ERR(stat, "cudaMemsetAsync on e_lj failed");
    // stat = cudaMemsetAsync(adat->e_el, 0, sizeof(*adat->e_el), ls);
    // CU_RET_ERR(stat, "cudaMemsetAsync on e_el failed");
}

void sits_gpu_clear_outputs(gmx_sits_cuda_t* gpu_sits, bool computeVirial)
{
    sits_cuda_clear_f(gpu_sits, gpu_sits->sits_atdat->natoms);
    /* clear shift force array and energies if the outputs were
       used in the current step */
    if (computeVirial)
    {
        sits_cuda_clear_e_fshift(nb);
    }
}

void gpu_init_sits_atomdata(gmx_sits_cuda_t* gpu_sits, const nbnxm_atomdata_t* nbat)
{
    cudaError_t      stat;
    int              nalloc, natoms;
    bool             realloced;
    cu_sits_atdat_t* d_atdat = gpu_sits->sits_atdat;

    natoms    = nbat->numAtoms();
    realloced = false;

    if (nbat->params().nenergrp > 1)
    {
        d_atdat->nenergrp = nbat->params().nenergrp;
        d_atdat->neg_2log = nbat->params().neg_2log;
    }

    /* need to reallocate if we have to copy more atoms than the amount of space
       available and only allocate if we haven't initialized yet, i.e d_atdat->natoms == -1 */
    if (natoms > d_atdat->nalloc)
    {
        nalloc = over_alloc_small(natoms);

        /* free up first if the arrays have already been initialized */
        if (d_atdat->nalloc != -1)
        {
            freeDeviceBuffer(&d_atdat->d_force_tot);
            freeDeviceBuffer(&d_atdat->d_force_pw);
            freeDeviceBuffer(&d_atdat->d_force_tot_nbat);
            freeDeviceBuffer(&d_atdat->d_force_pw_nbat);
            freeDeviceBuffer(&d_atdat->energrp);
        }

        stat = cudaMalloc((void**)&d_atdat->d_force_tot, nalloc * sizeof(*d_atdat->d_force_tot));
        CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->d_force_tot");
        stat = cudaMalloc((void**)&d_atdat->d_force_pw, nalloc * sizeof(*d_atdat->d_force_pw));
        CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->d_force_pw");
        stat = cudaMalloc((void**)&d_atdat->d_force_tot_nbat, nalloc * sizeof(*d_atdat->d_force_tot_nbat));
        CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->d_force_tot");
        stat = cudaMalloc((void**)&d_atdat->d_force_pw_nbat, nalloc * sizeof(*d_atdat->d_force_pw_nbat));
        CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->d_force_pw");
        if (nbat->params().nenergrp > 1)
        {
            stat = cudaMalloc((void**)&d_atdat->energrp, nalloc * sizeof(*d_atdat->energrp));
            CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->energrp");
        }

        d_atdat->nalloc = nalloc;
        realloced       = true;
    }

    d_atdat->natoms       = natoms;

    /* need to clear GPU f output if realloc happened */
    if (realloced)
    {
        sits_cuda_clear_f(gpu_sits, nalloc);
    }

    if (nbat->params().nenergrp > 1)
    {
        cu_copy_H2D_async(d_atdat->energrp, nbat->params().energrp_1x1.data(),
                          natoms * sizeof(*d_atdat->energrp), ls);
    }
}

void gpu_free(gmx_sits_cuda_t* gpu_sits)
{
    cudaError_t      stat;
    cu_sits_atdat_t* atdat;
    cu_sits_param_t* sits_param;

    if (gpu_sits == nullptr)
    {
        return;
    }

    atdat      = gpu_sits->sits_atdat;
    sits_param = gpu_sits->sits_param;

    // if ((info.sits_mode & 0x0000000F) == SIMPLE_SITS_MODE)
    // {
    //     if (simple_info.fc_pdf != NULL)
    //     {
    //         free(simple_info.fc_pdf);
    //     }
    // }

    if (atdat->d_enerd != NULL)
    {
        stat = cudaFree(atdat->d_enerd);
        CU_RET_ERR(stat, "cudaFree failed on atdat->d_enerd");
    }

    freeDeviceBuffer(&atdat->d_force_tot);
    freeDeviceBuffer(&atdat->d_force_pw);
    freeDeviceBuffer(&atdat->d_force_tot_nbat);
    freeDeviceBuffer(&atdat->d_force_pw_nbat);
    freeDeviceBuffer(&atdat->atomIndices);
    freeDeviceBuffer(&atdat->energrp);

    /* Free nbst */
    // pfree(nb->nbst.e_lj);
    // nb->nbst.e_lj = nullptr;

    // pfree(nb->nbst.e_el);
    // nb->nbst.e_el = nullptr;

    // pfree(nb->nbst.fshift);
    // nb->nbst.fshift = nullptr;

    sfree(atdat);
    sfree(sits_param);
    sfree(nb);

    if (debug)
    {
        fprintf(debug, "Cleaned up CUDA data structures.\n");
    }
}

// void* gpu_get_xq(gmx_nbnxn_gpu_t* nb)
// {
//     assert(nb);

//     return static_cast<void*>(nb->atdat->xq);
// }

// void* gpu_get_f(gmx_nbnxn_gpu_t* nb)
// {
//     assert(nb);

//     return static_cast<void*>(nb->atdat->f);
// }

} // namespace Sits
