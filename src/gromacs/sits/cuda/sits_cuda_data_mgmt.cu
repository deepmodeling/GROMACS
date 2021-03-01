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

/* This is a heuristically determined parameter for the Kepler
 * and Maxwell architectures for the minimum size of ci lists by multiplying
 * this constant with the # of multiprocessors on the current device.
 * Since the maximum number of blocks per multiprocessor is 16, the ideal
 * count for small systems is 32 or 48 blocks per multiprocessor. Because
 * there is a bit of fluctuations in the generated block counts, we use
 * a target of 44 instead of the ideal value of 48.
 */
static unsigned int gpu_min_ci_balanced_factor = 44;

/* Fw. decl. */
static void nbnxn_cuda_clear_e_fshift(gmx_nbnxn_cuda_t* nb);

/* Fw. decl, */
static void nbnxn_cuda_free_nbparam_table(cu_nbparam_t* nbparam);

/*! Initializes the atomdata structure first time, it only gets filled at
    pair-search. */
static void init_atomdata_first(cu_atomdata_t* ad, int ntypes)
{
    cudaError_t stat;

    ad->ntypes = ntypes;
    stat       = cudaMalloc((void**)&ad->shift_vec, SHIFTS * sizeof(*ad->shift_vec));
    CU_RET_ERR(stat, "cudaMalloc failed on ad->shift_vec");
    ad->bShiftVecUploaded = false;

    stat = cudaMalloc((void**)&ad->fshift, SHIFTS * sizeof(*ad->fshift));
    CU_RET_ERR(stat, "cudaMalloc failed on ad->fshift");

    stat = cudaMalloc((void**)&ad->e_lj, sizeof(*ad->e_lj));
    CU_RET_ERR(stat, "cudaMalloc failed on ad->e_lj");
    stat = cudaMalloc((void**)&ad->e_el, sizeof(*ad->e_el));
    CU_RET_ERR(stat, "cudaMalloc failed on ad->e_el");

    /* initialize to nullptr poiters to data that is not allocated here and will
       need reallocation in nbnxn_cuda_init_atomdata */
    ad->xq = nullptr;
    ad->f  = nullptr;

    /* size -1 indicates that the respective array hasn't been initialized yet */
    ad->natoms = -1;
    ad->nalloc = -1;
}

/*! Copies all parameters related to the cut-off from ic to nbp */
static void set_cutoff_parameters(cu_nbparam_t* nbp, const interaction_const_t* ic, const PairlistParams& listParams)
{
    nbp->ewald_beta        = ic->ewaldcoeff_q;
    nbp->sh_ewald          = ic->sh_ewald;
    nbp->epsfac            = ic->epsfac;
    nbp->two_k_rf          = 2.0 * ic->k_rf;
    nbp->c_rf              = ic->c_rf;
    nbp->rvdw_sq           = ic->rvdw * ic->rvdw;
    nbp->rcoulomb_sq       = ic->rcoulomb * ic->rcoulomb;
    nbp->rlistOuter_sq     = listParams.rlistOuter * listParams.rlistOuter;
    nbp->rlistInner_sq     = listParams.rlistInner * listParams.rlistInner;
    nbp->useDynamicPruning = listParams.useDynamicPruning;

    nbp->sh_lj_ewald   = ic->sh_lj_ewald;
    nbp->ewaldcoeff_lj = ic->ewaldcoeff_lj;

    nbp->rvdw_switch      = ic->rvdw_switch;
    nbp->dispersion_shift = ic->dispersion_shift;
    nbp->repulsion_shift  = ic->repulsion_shift;
    nbp->vdw_switch       = ic->vdw_switch;
}

/*! Initializes the nonbonded parameter data structure. */
static void init_nbparam(cu_nbparam_t*                   nbp,
                         const interaction_const_t*      ic,
                         const PairlistParams&           listParams,
                         const nbnxn_atomdata_t::Params& nbatParams)
{
    int ntypes;

    ntypes = nbatParams.numTypes;

    set_cutoff_parameters(nbp, ic, listParams);

    /* The kernel code supports LJ combination rules (geometric and LB) for
     * all kernel types, but we only generate useful combination rule kernels.
     * We currently only use LJ combination rule (geometric and LB) kernels
     * for plain cut-off LJ. On Maxwell the force only kernels speed up 15%
     * with PME and 20% with RF, the other kernels speed up about half as much.
     * For LJ force-switch the geometric rule would give 7% speed-up, but this
     * combination is rarely used. LJ force-switch with LB rule is more common,
     * but gives only 1% speed-up.
     */
    if (ic->vdwtype == evdwCUT)
    {
        switch (ic->vdw_modifier)
        {
            case eintmodNONE:
            case eintmodPOTSHIFT:
                switch (nbatParams.comb_rule)
                {
                    case ljcrNONE: nbp->vdwtype = evdwCuCUT; break;
                    case ljcrGEOM: nbp->vdwtype = evdwCuCUTCOMBGEOM; break;
                    case ljcrLB: nbp->vdwtype = evdwCuCUTCOMBLB; break;
                    default:
                        gmx_incons(
                                "The requested LJ combination rule is not implemented in the CUDA "
                                "GPU accelerated kernels!");
                }
                break;
            case eintmodFORCESWITCH: nbp->vdwtype = evdwCuFSWITCH; break;
            case eintmodPOTSWITCH: nbp->vdwtype = evdwCuPSWITCH; break;
            default:
                gmx_incons(
                        "The requested VdW interaction modifier is not implemented in the CUDA GPU "
                        "accelerated kernels!");
        }
    }
    else if (ic->vdwtype == evdwPME)
    {
        if (ic->ljpme_comb_rule == ljcrGEOM)
        {
            assert(nbatParams.comb_rule == ljcrGEOM);
            nbp->vdwtype = evdwCuEWALDGEOM;
        }
        else
        {
            assert(nbatParams.comb_rule == ljcrLB);
            nbp->vdwtype = evdwCuEWALDLB;
        }
    }
    else
    {
        gmx_incons(
                "The requested VdW type is not implemented in the CUDA GPU accelerated kernels!");
    }

    if (ic->eeltype == eelCUT)
    {
        nbp->eeltype = eelCuCUT;
    }
    else if (EEL_RF(ic->eeltype))
    {
        nbp->eeltype = eelCuRF;
    }
    else if ((EEL_PME(ic->eeltype) || ic->eeltype == eelEWALD))
    {
        nbp->eeltype = pick_ewald_kernel_type(*ic);
    }
    else
    {
        /* Shouldn't happen, as this is checked when choosing Verlet-scheme */
        gmx_incons(
                "The requested electrostatics type is not implemented in the CUDA GPU accelerated "
                "kernels!");
    }

    /* generate table for PME */
    nbp->coulomb_tab = nullptr;
    if (nbp->eeltype == eelCuEWALD_TAB || nbp->eeltype == eelCuEWALD_TAB_TWIN)
    {
        GMX_RELEASE_ASSERT(ic->coulombEwaldTables, "Need valid Coulomb Ewald correction tables");
        init_ewald_coulomb_force_table(*ic->coulombEwaldTables, nbp);
    }

    /* set up LJ parameter lookup table */
    if (!useLjCombRule(nbp))
    {
        initParamLookupTable(nbp->nbfp, nbp->nbfp_texobj, nbatParams.nbfp.data(), 2 * ntypes * ntypes);
    }

    /* set up LJ-PME parameter lookup table */
    if (ic->vdwtype == evdwPME)
    {
        initParamLookupTable(nbp->nbfp_comb, nbp->nbfp_comb_texobj, nbatParams.nbfp_comb.data(), 2 * ntypes);
    }
}

/*! Initializes the pair list data structure. */
static void init_plist(cu_plist_t* pl)
{
    /* initialize to nullptr pointers to data that is not allocated here and will
       need reallocation in nbnxn_gpu_init_pairlist */
    pl->sci   = nullptr;
    pl->cj4   = nullptr;
    pl->imask = nullptr;
    pl->excl  = nullptr;

    /* size -1 indicates that the respective array hasn't been initialized yet */
    pl->na_c          = -1;
    pl->nsci          = -1;
    pl->sci_nalloc    = -1;
    pl->ncj4          = -1;
    pl->cj4_nalloc    = -1;
    pl->nimask        = -1;
    pl->imask_nalloc  = -1;
    pl->nexcl         = -1;
    pl->excl_nalloc   = -1;
    pl->haveFreshList = false;
}

/*! Initializes simulation constant data. */
static void cuda_init_const(gmx_nbnxn_cuda_t*               nb,
                            const interaction_const_t*      ic,
                            const PairlistParams&           listParams,
                            const nbnxn_atomdata_t::Params& nbatParams)
{
    init_atomdata_first(nb->atdat, nbatParams.numTypes);
    init_nbparam(nb->nbparam, ic, listParams, nbatParams);

    /* clear energy and shift force outputs */
    nbnxn_cuda_clear_e_fshift(nb);
}

gmx_nbnxn_cuda_t* gpu_init_sits(const gmx_device_info_t*   deviceInfo,
                                const interaction_const_t* ic,
                                const sits_atomdata_t*     sits_at,
                                int /*rank*/)
{
    cudaError_t stat;

    gmx_sits_cuda_t* gpu_sits;
    snew(gpu_sits, 1);
    snew(gpu_sits->sits_atdat, 1);
    snew(gpu_sits->sits_param, 1);

    /* init nbst */
    pmalloc((void**)&nb->nbst.e_lj, sizeof(*nb->nbst.e_lj));
    pmalloc((void**)&nb->nbst.e_el, sizeof(*nb->nbst.e_el));
    pmalloc((void**)&nb->nbst.fshift, SHIFTS * sizeof(*nb->nbst.fshift));

    init_plist(nb->plist[InteractionLocality::Local]);

    /* set device info, just point it to the right GPU among the detected ones */
    gpu_sits->dev_info = deviceInfo;

    /* local/non-local GPU streams */
    stat = cudaStreamCreate(&nb->stream[InteractionLocality::Local]);

    /* set the kernel type for the current GPU */
    /* pick L1 cache configuration */
    cuda_set_cacheconfig();

    cuda_init_const(nb, ic, listParams, nbat->params());

    nb->atomIndicesSize       = 0;
    nb->atomIndicesSize_alloc = 0;

    if (debug)
    {
        fprintf(debug, "Initialized CUDA data structures.\n");
    }

    return nb;
}

void gpu_init_pairlist(gmx_nbnxn_cuda_t* nb, const NbnxnPairlistGpu* h_plist, const InteractionLocality iloc)
{
    char         sbuf[STRLEN];
    bool         bDoTime = (nb->bDoTime && !h_plist->sci.empty());
    cudaStream_t stream  = nb->stream[iloc];
    cu_plist_t*  d_plist = nb->plist[iloc];

    if (d_plist->na_c < 0)
    {
        d_plist->na_c = h_plist->na_ci;
    }
    else
    {
        if (d_plist->na_c != h_plist->na_ci)
        {
            sprintf(sbuf, "In cu_init_plist: the #atoms per cell has changed (from %d to %d)",
                    d_plist->na_c, h_plist->na_ci);
            gmx_incons(sbuf);
        }
    }

    gpu_timers_t::Interaction& iTimers = nb->timers->interaction[iloc];

    if (bDoTime)
    {
        iTimers.pl_h2d.openTimingRegion(stream);
        iTimers.didPairlistH2D = true;
    }

    DeviceContext context = nullptr;

    reallocateDeviceBuffer(&d_plist->sci, h_plist->sci.size(), &d_plist->nsci, &d_plist->sci_nalloc, context);
    copyToDeviceBuffer(&d_plist->sci, h_plist->sci.data(), 0, h_plist->sci.size(), stream,
                       GpuApiCallBehavior::Async, bDoTime ? iTimers.pl_h2d.fetchNextEvent() : nullptr);

    reallocateDeviceBuffer(&d_plist->cj4, h_plist->cj4.size(), &d_plist->ncj4, &d_plist->cj4_nalloc, context);
    copyToDeviceBuffer(&d_plist->cj4, h_plist->cj4.data(), 0, h_plist->cj4.size(), stream,
                       GpuApiCallBehavior::Async, bDoTime ? iTimers.pl_h2d.fetchNextEvent() : nullptr);

    reallocateDeviceBuffer(&d_plist->imask, h_plist->cj4.size() * c_nbnxnGpuClusterpairSplit,
                           &d_plist->nimask, &d_plist->imask_nalloc, context);

    reallocateDeviceBuffer(&d_plist->excl, h_plist->excl.size(), &d_plist->nexcl,
                           &d_plist->excl_nalloc, context);
    copyToDeviceBuffer(&d_plist->excl, h_plist->excl.data(), 0, h_plist->excl.size(), stream,
                       GpuApiCallBehavior::Async, bDoTime ? iTimers.pl_h2d.fetchNextEvent() : nullptr);

    if (bDoTime)
    {
        iTimers.pl_h2d.closeTimingRegion(stream);
    }

    /* the next use of thist list we be the first one, so we need to prune */
    d_plist->haveFreshList = true;
}

/*! Clears the first natoms_clear elements of the GPU nonbonded force output array. */
static void sits_cuda_clear_f(gmx_sits_cuda_t* gpu_sits, int natoms_clear)
{
    cudaError_t    stat;
    cu_atomdata_t* adat = nb->atdat;
    cudaStream_t   ls   = nb->stream[InteractionLocality::Local];

    stat = cudaMemsetAsync(adat->f, 0, natoms_clear * sizeof(*adat->f), ls);
    CU_RET_ERR(stat, "cudaMemsetAsync on f falied");
}

/*! Clears nonbonded shift force output array and energy outputs on the GPU. */
static void nbnxn_cuda_clear_e_fshift(gmx_nbnxn_cuda_t* nb)
{
    cudaError_t    stat;
    cu_atomdata_t* adat = nb->atdat;
    cudaStream_t   ls   = nb->stream[InteractionLocality::Local];

    stat = cudaMemsetAsync(adat->fshift, 0, SHIFTS * sizeof(*adat->fshift), ls);
    CU_RET_ERR(stat, "cudaMemsetAsync on fshift falied");
    stat = cudaMemsetAsync(adat->e_lj, 0, sizeof(*adat->e_lj), ls);
    CU_RET_ERR(stat, "cudaMemsetAsync on e_lj falied");
    stat = cudaMemsetAsync(adat->e_el, 0, sizeof(*adat->e_el), ls);
    CU_RET_ERR(stat, "cudaMemsetAsync on e_el falied");
}

void gpu_clear_outputs(gmx_nbnxn_cuda_t* nb, bool computeVirial)
{
    nbnxn_cuda_clear_f(nb, nb->atdat->natoms);
    /* clear shift force array and energies if the outputs were
       used in the current step */
    if (computeVirial)
    {
        nbnxn_cuda_clear_e_fshift(nb);
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
            freeDeviceBuffer(&d_atdat->d_force_nbat_tot);
            freeDeviceBuffer(&d_atdat->d_force_nbat_pw);
            freeDeviceBuffer(&d_atdat->energrp);
        }

        stat = cudaMalloc((void**)&d_atdat->d_force_tot, nalloc * sizeof(*d_atdat->d_force_tot));
        CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->d_force_tot");
        stat = cudaMalloc((void**)&d_atdat->d_force_pw, nalloc * sizeof(*d_atdat->d_force_pw));
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
        nbnxn_cuda_clear_f(nb, nalloc);
    }

    if (nbat->params().nenergrp > 1)
    {
        cu_copy_H2D_async(d_atdat->energrp, nbat->params().energrp_gpu.data(),
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
    freeDeviceBuffer(&atdat->d_force_nbat_tot);
    freeDeviceBuffer(&atdat->d_force_nbat_pw);
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

void* gpu_get_xq(gmx_nbnxn_gpu_t* nb)
{
    assert(nb);

    return static_cast<void*>(nb->atdat->xq);
}

void* gpu_get_f(gmx_nbnxn_gpu_t* nb)
{
    assert(nb);

    return static_cast<void*>(nb->atdat->f);
}

} // namespace Sits
