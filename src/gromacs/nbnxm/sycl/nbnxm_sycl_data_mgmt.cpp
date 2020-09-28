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
#include "gmxpre.h"

#include "gromacs/gpu_utils/device_stream_manager.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/nbnxm_gpu_data_mgmt.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"

#include "nbnxm_sycl.h"
#include "nbnxm_sycl_types.h"

namespace Nbnxm
{

/*! Initializes the atomdata structure first time, it only gets filled at pair-search. */
static void initAtomdataFirst(sycl_atomdata_t* ad, int ntypes, const DeviceContext& deviceContext)
{
    ad->ntypes = ntypes;
    allocateDeviceBuffer(&ad->shift_vec, SHIFTS, deviceContext);
    ad->bShiftVecUploaded = false;

    allocateDeviceBuffer(&ad->fshift, SHIFTS, deviceContext);
    allocateDeviceBuffer(&ad->e_lj, 1, deviceContext);
    allocateDeviceBuffer(&ad->e_el, 1, deviceContext);

    /* size -1 indicates that the respective array hasn't been initialized yet */
    ad->natoms = -1;
    ad->nalloc = -1;
}

/*! Initializes the nonbonded parameter data structure. */
static void initNbparam(NBParamGpu*                     nbp,
                        const interaction_const_t*      ic,
                        const PairlistParams&           listParams,
                        const nbnxn_atomdata_t::Params& nbatParams,
                        const DeviceContext&            deviceContext)
{
    // SYCL-TODO: Copypasted from CUDA version
    const int ntypes = nbatParams.numTypes;

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
                    case ljcrNONE: nbp->vdwtype = evdwTypeCUT; break;
                    case ljcrGEOM: nbp->vdwtype = evdwTypeCUTCOMBGEOM; break;
                    case ljcrLB: nbp->vdwtype = evdwTypeCUTCOMBLB; break;
                    default:
                        gmx_incons(
                                "The requested LJ combination rule is not implemented in the SYCL "
                                "GPU accelerated kernels!");
                }
                break;
            case eintmodFORCESWITCH: nbp->vdwtype = evdwTypeFSWITCH; break;
            case eintmodPOTSWITCH: nbp->vdwtype = evdwTypePSWITCH; break;
            default:
                gmx_incons(
                        "The requested VdW interaction modifier is not implemented in the SYCL GPU "
                        "accelerated kernels!");
        }
    }
    else if (ic->vdwtype == evdwPME)
    {
        if (ic->ljpme_comb_rule == ljcrGEOM)
        {
            assert(nbatParams.comb_rule == ljcrGEOM);
            nbp->vdwtype = evdwTypeEWALDGEOM;
        }
        else
        {
            assert(nbatParams.comb_rule == ljcrLB);
            nbp->vdwtype = evdwTypeEWALDLB;
        }
    }
    else
    {
        gmx_incons(
                "The requested VdW type is not implemented in the SYCL GPU accelerated kernels!");
    }

    if (ic->eeltype == eelCUT)
    {
        nbp->eeltype = eelTypeCUT;
    }
    else if (EEL_RF(ic->eeltype))
    {
        nbp->eeltype = eelTypeRF;
    }
    else if ((EEL_PME(ic->eeltype) || ic->eeltype == eelEWALD))
    {
        nbp->eeltype = nbnxn_gpu_pick_ewald_kernel_type(*ic);
    }
    else
    {
        /* Shouldn't happen, as this is checked when choosing Verlet-scheme */
        gmx_incons(
                "The requested electrostatics type is not implemented in the CUDA GPU accelerated "
                "kernels!");
    }

    /* generate table for PME */
    if (nbp->eeltype == eelTypeEWALD_TAB || nbp->eeltype == eelTypeEWALD_TAB_TWIN)
    {
        GMX_RELEASE_ASSERT(ic->coulombEwaldTables, "Need valid Coulomb Ewald correction tables");
        init_ewald_coulomb_force_table(*ic->coulombEwaldTables, nbp, deviceContext);
    }

    /* set up LJ parameter lookup table */
    if (!useLjCombRule(nbp->vdwtype))
    {
        initParamLookupTable(&nbp->nbfp, &nbp->nbfp_texobj, nbatParams.nbfp.data(),
                             2 * ntypes * ntypes, deviceContext);
    }

    /* set up LJ-PME parameter lookup table */
    if (ic->vdwtype == evdwPME)
    {
        initParamLookupTable(&nbp->nbfp_comb, &nbp->nbfp_comb_texobj, nbatParams.nbfp_comb.data(),
                             2 * ntypes, deviceContext);
    }
}

/*! Clears the first natoms_clear elements of the GPU nonbonded force output array. */
static void clearF(NbnxmGpu* nb, int natoms_clear)
{
    sycl_atomdata_t*    adat        = nb->atdat;
    const DeviceStream& localStream = *nb->deviceStreams[InteractionLocality::Local];
    clearDeviceBufferAsync(&adat->f, 0, natoms_clear, localStream);
}

/*! Clears nonbonded shift force output array and energy outputs on the GPU. */
static void clearEFShift(NbnxmGpu* nb)
{
    sycl_atomdata_t*    adat        = nb->atdat;
    const DeviceStream& localStream = *nb->deviceStreams[InteractionLocality::Local];

    clearDeviceBufferAsync(&adat->fshift, 0, SHIFTS, localStream);
    clearDeviceBufferAsync(&adat->e_lj, 0, 1, localStream);
    clearDeviceBufferAsync(&adat->e_el, 0, 1, localStream);
}

/*! Initializes simulation constant data. */
static void syclInitConst(NbnxmGpu*                       nb,
                          const interaction_const_t*      ic,
                          const PairlistParams&           listParams,
                          const nbnxn_atomdata_t::Params& nbatParams)
{
    initAtomdataFirst(nb->atdat, nbatParams.numTypes, *nb->deviceContext_);
    initNbparam(nb->nbparam, ic, listParams, nbatParams, *nb->deviceContext_);

    /* clear energy and shift force outputs */
    clearEFShift(nb);
}

//! This function is documented in the header file
void gpu_clear_outputs(NbnxmGpu* nb, bool computeVirial)
{
    clearF(nb, nb->atdat->natoms);
    /* clear shift force array and energies if the outputs were
       used in the current step */
    if (computeVirial)
    {
        clearEFShift(nb);
    }
}

//! This function is documented in the header file
NbnxmGpu* gpu_init(const gmx::DeviceStreamManager& deviceStreamManager,
                   const interaction_const_t*      ic,
                   const PairlistParams&           listParams,
                   const nbnxn_atomdata_t*         nbat,
                   const bool                      bLocalAndNonlocal)
{
    auto* nb                              = new NbnxmGpu();
    nb->deviceContext_                    = &deviceStreamManager.context();
    nb->atdat                             = new sycl_atomdata_t();
    nb->nbparam                           = new NBParamGpu();
    nb->plist[InteractionLocality::Local] = new Nbnxm::gpu_plist();
    if (bLocalAndNonlocal)
    {
        nb->plist[InteractionLocality::NonLocal] = new Nbnxm::gpu_plist();
    }

    nb->bUseTwoStreams = bLocalAndNonlocal;

    nb->timers = new gpu_timers_t();
    snew(nb->timings, 1);

    smalloc(nb->nbst.e_lj, sizeof(*nb->nbst.e_lj));
    smalloc(nb->nbst.e_el, sizeof(*nb->nbst.e_el));
    smalloc(nb->nbst.fshift, SHIFTS * sizeof(*nb->nbst.fshift));

    init_plist(nb->plist[InteractionLocality::Local]);

    /* local/non-local GPU streams */
    GMX_RELEASE_ASSERT(deviceStreamManager.streamIsValid(gmx::DeviceStreamType::NonBondedLocal),
                       "Local non-bonded stream should be initialized to use GPU for non-bonded.");
    nb->deviceStreams[InteractionLocality::Local] =
            &deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal);
    if (nb->bUseTwoStreams)
    {
        init_plist(nb->plist[InteractionLocality::NonLocal]);

        /* Note that the device we're running on does not have to support
         * priorities, because we are querying the priority range which in this
         * case will be a single value.
         */
        GMX_RELEASE_ASSERT(deviceStreamManager.streamIsValid(gmx::DeviceStreamType::NonBondedNonLocal),
                           "Non-local non-bonded stream should be initialized to use GPU for "
                           "non-bonded with domain decomposition.");
        nb->deviceStreams[InteractionLocality::NonLocal] =
                &deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedNonLocal);
        ;
    }

    nb->xNonLocalCopyD2HDone = new GpuEventSynchronizer();

    if (getenv("GMX_ENABLE_GPU_TIMING") != nullptr)
    {
        // SYCL-TODO
        gmx_warning(
                "Trying to enable GPU timings via GMX_ENABLE_GPU_TIMING, but they are not "
                "supported in SYCL yet.");
    }

    nb->bDoTime = false;

    /* set the kernel type for the current GPU */
    /* pick L1 cache configuration */
    syclInitConst(nb, ic, listParams, nbat->params());

    nb->atomIndicesSize       = 0;
    nb->atomIndicesSize_alloc = 0;
    nb->ncxy_na               = 0;
    nb->ncxy_na_alloc         = 0;
    nb->ncxy_ind              = 0;
    nb->ncxy_ind_alloc        = 0;
    nb->ncell                 = 0;
    nb->ncell_alloc           = 0;

    return nb;
}

//! This function is documented in the header file
void gpu_upload_shiftvec(NbnxmGpu* nb, const nbnxn_atomdata_t* nbatom)
{
    sycl_atomdata_t*    adat        = nb->atdat;
    const DeviceStream& localStream = *nb->deviceStreams[InteractionLocality::Local];

    /* only if we have a dynamic box */
    if (nbatom->bDynamicBox || !adat->bShiftVecUploaded)
    {
        GMX_ASSERT(adat->shift_vec.elementSize() == sizeof(nbatom->shift_vec[0]),
                   "Sizes of host- and device-side shift vectors should be the same.");
        copyToDeviceBuffer(&adat->shift_vec, reinterpret_cast<const float3*>(nbatom->shift_vec.data()),
                           0, SHIFTS, localStream, GpuApiCallBehavior::Async, nullptr);
        adat->bShiftVecUploaded = true;
    }
}

//! This function is documented in the header file
void gpu_init_atomdata(NbnxmGpu* nb, const nbnxn_atomdata_t* nbat)
{
    sycl_atomdata_t*     atdat         = nb->atdat;
    const DeviceContext& deviceContext = *nb->deviceContext_;
    const DeviceStream&  localStream   = *nb->deviceStreams[InteractionLocality::Local];

    const int natoms = nbat->numAtoms();
    int       nalloc;
    bool      realloced = false;

    /* need to reallocate if we have to copy more atoms than the amount of space
       available and only allocate if we haven't initialized yet, i.e d_atdat->natoms == -1 */
    if (natoms > atdat->nalloc)
    {
        nalloc = over_alloc_small(natoms);

        /* free up first if the arrays have already been initialized */
        if (atdat->nalloc != -1)
        {
            freeDeviceBuffer(&atdat->f);
            freeDeviceBuffer(&atdat->xq);
            freeDeviceBuffer(&atdat->atom_types);
            freeDeviceBuffer(&atdat->lj_comb);
        }

        allocateDeviceBuffer(&atdat->f, nalloc, deviceContext);
        allocateDeviceBuffer(&atdat->xq, nalloc, deviceContext);
        if (useLjCombRule(nb->nbparam->vdwtype))
        {
            allocateDeviceBuffer(&atdat->lj_comb, nalloc, deviceContext);
        }
        else
        {
            allocateDeviceBuffer(&atdat->atom_types, nalloc, deviceContext);
        }

        atdat->nalloc = nalloc;
        realloced     = true;
    }

    atdat->natoms       = natoms;
    atdat->natoms_local = nbat->natoms_local;

    /* need to clear GPU f output if realloc happened */
    if (realloced)
    {
        clearF(nb, nalloc);
    }

    if (useLjCombRule(nb->nbparam->vdwtype))
    {
        GMX_ASSERT(atdat->lj_comb.elementSize() == sizeof(float2),
                   "Size of the LJ parameters element should be equal to the size of float2.");
        copyToDeviceBuffer(&atdat->lj_comb,
                           reinterpret_cast<const float2*>(nbat->params().lj_comb.data()), 0,
                           natoms, localStream, GpuApiCallBehavior::Async, nullptr);
    }
    else
    {
        GMX_ASSERT(atdat->atom_types.elementSize() == sizeof(nbat->params().type[0]),
                   "Sizes of host- and device-side atom types should be the same.");
        copyToDeviceBuffer(&atdat->atom_types, nbat->params().type.data(), 0, natoms, localStream,
                           GpuApiCallBehavior::Async, nullptr);
    }
}

//! This function is documented in the header file
void gpu_free(NbnxmGpu* nb)
{
    sycl_atomdata_t* atdat;
    NBParamGpu*      nbparam;

    if (nb == nullptr)
    {
        return;
    }

    atdat   = nb->atdat;
    nbparam = nb->nbparam;

    if ((nbparam->coulomb_tab.buffer_ == nullptr)
        && (nbparam->eeltype == eelTypeEWALD_TAB || nbparam->eeltype == eelTypeEWALD_TAB_TWIN))
    {
        destroyParamLookupTable(&nbparam->coulomb_tab, nbparam->coulomb_tab_texobj);
    }

    delete nb->timers;

    delete nb->nbst.e_lj;
    delete nb->nbst.e_el;
    delete[] nb->nbst.fshift;

    if (!useLjCombRule(nb->nbparam->vdwtype))
    {
        destroyParamLookupTable(&nbparam->nbfp, nbparam->nbfp_texobj);
    }

    if (nbparam->vdwtype == evdwTypeEWALDGEOM || nbparam->vdwtype == evdwTypeEWALDLB)
    {
        destroyParamLookupTable(&nbparam->nbfp_comb, nbparam->nbfp_comb_texobj);
    }

    freeDeviceBuffer(&atdat->shift_vec);
    freeDeviceBuffer(&atdat->fshift);

    freeDeviceBuffer(&atdat->e_lj);
    freeDeviceBuffer(&atdat->e_el);

    freeDeviceBuffer(&atdat->f);
    freeDeviceBuffer(&atdat->xq);
    freeDeviceBuffer(&atdat->atom_types);
    freeDeviceBuffer(&atdat->lj_comb);

    /* Free plist */
    auto* plist = nb->plist[InteractionLocality::Local];
    freeDeviceBuffer(&plist->sci);
    freeDeviceBuffer(&plist->cj4);
    freeDeviceBuffer(&plist->imask);
    freeDeviceBuffer(&plist->excl);
    delete plist;
    if (nb->bUseTwoStreams)
    {
        auto* plist_nl = nb->plist[InteractionLocality::NonLocal];
        freeDeviceBuffer(&plist_nl->sci);
        freeDeviceBuffer(&plist_nl->cj4);
        freeDeviceBuffer(&plist_nl->imask);
        freeDeviceBuffer(&plist_nl->excl);
        delete plist_nl;
    }

    delete atdat;
    delete nbparam;
    sfree(nb->timings);
    delete nb;
}

//! This function is documented in the header file
int gpu_min_ci_balanced(NbnxmGpu* /*nb*/)
{
    GMX_THROW(gmx::NotImplementedError("Not implemented on SYCL yet"));
}

} // namespace Nbnxm
