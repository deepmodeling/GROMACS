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
 * \brief
 * Implements functionality for PairlistSet.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#include "gmxpre.h"

#include "pairlistset.h"

#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/mdtypes/nblist.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gridset.h"
#include "gromacs/nbnxm/pairlistsethelpers.h"
#include "gromacs/nbnxm/pairsearch.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/listoflists.h"

#include "pairlistwork.h"

PairlistSet::PairlistSet(const gmx::InteractionLocality locality, const PairlistParams& pairlistParams) :
    locality_(locality),
    params_(pairlistParams),
    combineLists_(sc_isGpuPairListType[pairlistParams.pairlistType]), // Currently GPU lists are always combined
    isCpuType_(!sc_isGpuPairListType[pairlistParams.pairlistType])
{

    const int numLists = gmx_omp_nthreads_get(emntNonbonded);

    if (!combineLists_ && numLists > NBNXN_BUFFERFLAG_MAX_THREADS)
    {
        gmx_fatal(FARGS,
                  "%d OpenMP threads were requested. Since the non-bonded force buffer reduction "
                  "is prohibitively slow with more than %d threads, we do not allow this. Use %d "
                  "or less OpenMP threads.",
                  numLists,
                  NBNXN_BUFFERFLAG_MAX_THREADS,
                  NBNXN_BUFFERFLAG_MAX_THREADS);
    }

    if (isCpuType_)
    {
        cpuLists_.resize(numLists);
        if (numLists > 1)
        {
            cpuListsWork_.resize(numLists);
        }
    }
    else
    {
        /* Only list 0 is used on the GPU, use normal allocation for i>0 */
        gpuLists_.emplace_back(gmx::PinningPolicy::PinnedIfSupported);
        /* Lists 0 to numLists are use for constructing lists in parallel
         * on the CPU using numLists threads (and then merged into list 0).
         */
        for (int i = 1; i < numLists; i++)
        {
            gpuLists_.emplace_back(gmx::PinningPolicy::CannotBePinned);
        }
    }
    if (params_.haveFep)
    {
        fepLists_.resize(numLists);

        /* Execute in order to avoid memory interleaving between threads */
#pragma omp parallel for num_threads(numLists) schedule(static)
        for (int i = 0; i < numLists; i++)
        {
            try
            {
                /* We used to allocate all normal lists locally on each thread
                 * as well. The question is if allocating the object on the
                 * master thread (but all contained list memory thread local)
                 * impacts performance.
                 */
                fepLists_[i] = std::make_unique<t_nblist>();
            }
            GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
        }
    }
}

PairlistSet::~PairlistSet() = default;

void PairlistSet::constructPairlists(const Nbnxm::GridSet&         gridSet,
                                     gmx::ArrayRef<PairsearchWork> searchWork,
                                     nbnxn_atomdata_t*             nbat,
                                     const gmx::ListOfLists<int>&  exclusions,
                                     const int                     minimumIlistCountForGpuBalancing,
                                     t_nrnb*                       nrnb,
                                     SearchCycleCounting*          searchCycleCounting)
{
    const real rlist = params_.rlistOuter;

    const int numLists = (isCpuType_ ? cpuLists_.size() : gpuLists_.size());

    if (debug)
    {
        fprintf(debug, "ns making %d nblists\n", numLists);
    }

    nbat->bUseBufferFlags = (nbat->out.size() > 1);
    /* We should re-init the flags before making the first list */
    if (nbat->bUseBufferFlags && locality_ == gmx::InteractionLocality::Local)
    {
        resizeAndZeroBufferFlags(&nbat->buffer_flags, nbat->numAtoms());
    }

    int   nsubpair_target  = 0;
    float nsubpair_tot_est = 0.0F;
    if (!isCpuType_ && minimumIlistCountForGpuBalancing > 0)
    {
        get_nsubpair_target(
                gridSet, locality_, rlist, minimumIlistCountForGpuBalancing, &nsubpair_target, &nsubpair_tot_est);
    }

    /* Clear all pair-lists */
    for (int th = 0; th < numLists; th++)
    {
        if (isCpuType_)
        {
            clear_pairlist(&cpuLists_[th]);
        }
        else
        {
            clear_pairlist(&gpuLists_[th]);
        }

        if (params_.haveFep)
        {
            clear_pairlist_fep(fepLists_[th].get());
        }
    }

    const gmx_domdec_zones_t* ddZones = gridSet.domainSetup().zones;
    GMX_ASSERT(locality_ == gmx::InteractionLocality::Local || ddZones != nullptr,
               "Nonlocal interaction locality with null ddZones.");

    const auto iZoneRange = getIZoneRange(gridSet.domainSetup(), locality_);

    for (const int iZone : iZoneRange)
    {
        const Nbnxm::Grid& iGrid = gridSet.grids()[iZone];

        const auto jZoneRange = getJZoneRange(ddZones, locality_, iZone);

        for (int jZone : jZoneRange)
        {
            const Nbnxm::Grid& jGrid = gridSet.grids()[jZone];

            if (debug)
            {
                fprintf(debug, "ns search grid %d vs %d\n", iZone, jZone);
            }

            searchCycleCounting->start(enbsCCsearch);

            const int ci_block =
                    get_ci_block_size(iGrid, gridSet.domainSetup().haveMultipleDomains, numLists);

            /* With GPU: generate progressively smaller lists for
             * load balancing for local only or non-local with 2 zones.
             */
            const bool progBal = (locality_ == gmx::InteractionLocality::Local || ddZones->n <= 2);

#pragma omp parallel for num_threads(numLists) schedule(static)
            for (int th = 0; th < numLists; th++)
            {
                try
                {
                    /* Re-init the thread-local work flag data before making
                     * the first list (not an elegant conditional).
                     */
                    if (nbat->bUseBufferFlags && (iZone == 0 && jZone == 0))
                    {
                        resizeAndZeroBufferFlags(&searchWork[th].buffer_flags, nbat->numAtoms());
                    }

                    if (combineLists_ && th > 0)
                    {
                        GMX_ASSERT(!isCpuType_, "Can only combine GPU lists");

                        clear_pairlist(&gpuLists_[th]);
                    }

                    PairsearchWork& work = searchWork[th];

                    work.cycleCounter.start();

                    t_nblist* fepListPtr = (fepLists_.empty() ? nullptr : fepLists_[th].get());

                    /* Divide the i cells equally over the pairlists */
                    if (isCpuType_)
                    {
                        nbnxn_make_pairlist_part<NbnxnPairlistCpu>(gridSet,
                                                                   iGrid,
                                                                   jGrid,
                                                                   &work,
                                                                   nbat,
                                                                   exclusions,
                                                                   rlist,
                                                                   params_.pairlistType,
                                                                   ci_block,
                                                                   nbat->bUseBufferFlags,
                                                                   nsubpair_target,
                                                                   progBal,
                                                                   nsubpair_tot_est,
                                                                   th,
                                                                   numLists,
                                                                   &cpuLists_[th],
                                                                   fepListPtr);
                    }
                    else
                    {
                        nbnxn_make_pairlist_part<NbnxnPairlistGpu>(gridSet,
                                                                   iGrid,
                                                                   jGrid,
                                                                   &work,
                                                                   nbat,
                                                                   exclusions,
                                                                   rlist,
                                                                   params_.pairlistType,
                                                                   ci_block,
                                                                   nbat->bUseBufferFlags,
                                                                   nsubpair_target,
                                                                   progBal,
                                                                   nsubpair_tot_est,
                                                                   th,
                                                                   numLists,
                                                                   &gpuLists_[th],
                                                                   fepListPtr);
                    }

                    work.cycleCounter.stop();
                }
                GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
            }
            searchCycleCounting->stop(enbsCCsearch);

            int np_tot = 0;
            int np_noq = 0;
            int np_hlj = 0;
            for (int th = 0; th < numLists; th++)
            {
                inc_nrnb(nrnb, eNR_NBNXN_DIST2, searchWork[th].ndistc);

                if (isCpuType_)
                {
                    const NbnxnPairlistCpu& nbl = cpuLists_[th];
                    np_tot += nbl.cj.size();
                    np_noq += nbl.work->ncj_noq;
                    np_hlj += nbl.work->ncj_hlj;
                }
                else
                {
                    const NbnxnPairlistGpu& nbl = gpuLists_[th];
                    /* This count ignores potential subsequent pair pruning */
                    np_tot += nbl.nci_tot;
                }
            }
            const int nap = isCpuType_ ? cpuLists_[0].na_ci * cpuLists_[0].na_cj
                                       : gmx::square(gpuLists_[0].na_ci);

            natpair_ljq_ = (np_tot - np_noq) * nap - np_hlj * nap / 2;
            natpair_lj_  = np_noq * nap;
            natpair_q_   = np_hlj * nap / 2;

            if (combineLists_ && numLists > 1)
            {
                GMX_ASSERT(!isCpuType_, "Can only combine GPU lists");

                searchCycleCounting->start(enbsCCcombine);

                combine_nblists(gmx::constArrayRefFromArray(&gpuLists_[1], numLists - 1), &gpuLists_[0]);

                searchCycleCounting->stop(enbsCCcombine);
            }
        }
    }

    if (isCpuType_)
    {
        if (numLists > 1 && checkRebalanceSimpleLists(cpuLists_))
        {
            rebalanceSimpleLists(cpuLists_, cpuListsWork_, searchWork);

            /* Swap the sets of pair lists */
            cpuLists_.swap(cpuListsWork_);
        }
    }
    else
    {
        /* Sort the entries on size, large ones first */
        if (combineLists_ || gpuLists_.size() == 1)
        {
            sort_sci(&gpuLists_[0]);
        }
        else
        {
#pragma omp parallel for num_threads(numLists) schedule(static)
            for (int th = 0; th < numLists; th++)
            {
                try
                {
                    sort_sci(&gpuLists_[th]);
                }
                GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
            }
        }
    }

    if (nbat->bUseBufferFlags)
    {
        reduce_buffer_flags(searchWork, numLists, nbat->buffer_flags);
    }

    if (gridSet.haveFep())
    {
        /* Balance the free-energy lists over all the threads */
        balance_fep_lists(fepLists_, searchWork);
    }

    if (isCpuType_)
    {
        /* This is a fresh list, so not pruned, stored using ci.
         * ciOuter is invalid at this point.
         */
        GMX_ASSERT(cpuLists_[0].ciOuter.empty(), "ciOuter is invalid so it should be empty");
    }

    /* If we have more than one list, they either got rebalancing (CPU)
     * or combined (GPU), so we should dump the final result to debug.
     */
    if (debug)
    {
        if (isCpuType_ && cpuLists_.size() > 1)
        {
            for (auto& cpuList : cpuLists_)
            {
                print_nblist_statistics(debug, cpuList, gridSet, rlist);
            }
        }
        else if (!isCpuType_ && gpuLists_.size() > 1)
        {
            print_nblist_statistics(debug, gpuLists_[0], gridSet, rlist);
        }
    }

    if (debug)
    {
        if (gmx_debug_at)
        {
            if (isCpuType_)
            {
                for (auto& cpuList : cpuLists_)
                {
                    print_nblist_ci_cj(debug, cpuList);
                }
            }
            else
            {
                print_nblist_sci_cj(debug, gpuLists_[0]);
            }
        }

        if (nbat->bUseBufferFlags)
        {
            print_reduction_cost(nbat->buffer_flags, numLists);
        }
    }

    if (params_.useDynamicPruning && isCpuType_)
    {
        prepareListsForDynamicPruning(cpuLists_);
    }
}
