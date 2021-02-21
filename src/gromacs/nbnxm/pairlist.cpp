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

#include "gmxpre.h"

#include "pairlist.h"

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/mdtypes/nblist.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"

#include "pairlistset.h"
#include "pairlistsets.h"
#include "pairlistwork.h"
#include "pairsearch.h"

NbnxnPairlistCpu::NbnxnPairlistCpu() :
    na_ci(c_nbnxnCpuIClusterSize),
    na_cj(0),
    rlist(0),
    ncjInUse(0),
    nci_tot(0),
    work(std::make_unique<NbnxnPairlistCpuWork>())
{
}

NbnxnPairlistGpu::NbnxnPairlistGpu(gmx::PinningPolicy pinningPolicy) :
    na_ci(c_nbnxnGpuClusterSize),
    na_cj(c_nbnxnGpuClusterSize),
    na_sc(c_gpuNumClusterPerCell * c_nbnxnGpuClusterSize),
    rlist(0),
    sci({}, { pinningPolicy }),
    cj4({}, { pinningPolicy }),
    excl({}, { pinningPolicy }),
    nci_tot(0),
    work(std::make_unique<NbnxnPairlistGpuWork>())
{
    static_assert(c_nbnxnGpuNumClusterPerSupercluster == c_gpuNumClusterPerCell,
                  "The search code assumes that the a super-cluster matches a search grid cell");

    static_assert(sizeof(cj4[0].imei[0].imask) * 8 >= c_nbnxnGpuJgroupSize * c_gpuNumClusterPerCell,
                  "The i super-cluster cluster interaction mask does not contain a sufficient "
                  "number of bits");

    static_assert(sizeof(excl[0]) * 8 >= c_nbnxnGpuJgroupSize * c_gpuNumClusterPerCell,
                  "The GPU exclusion mask does not contain a sufficient number of bits");

    // We always want a first entry without any exclusions
    excl.resize(1);
}

void clear_pairlist(NbnxnPairlistCpu* nbl)
{
    nbl->ci.clear();
    nbl->cj.clear();
    nbl->ncjInUse = 0;
    nbl->nci_tot  = 0;
    nbl->ciOuter.clear();
    nbl->cjOuter.clear();

    nbl->work->ncj_noq = 0;
    nbl->work->ncj_hlj = 0;
}

void clear_pairlist(NbnxnPairlistGpu* nbl)
{
    nbl->sci.clear();
    nbl->cj4.clear();
    nbl->excl.resize(1);
    nbl->nci_tot = 0;
}

void clear_pairlist_fep(t_nblist* nl)
{
    nl->nri = 0;
    nl->nrj = 0;
    if (nl->jindex.empty())
    {
        nl->jindex.resize(1);
    }
    nl->jindex[0] = 0;
}

void nonbonded_verlet_t::constructPairlist(const gmx::InteractionLocality iLocality,
                                           const gmx::ListOfLists<int>&   exclusions,
                                           int64_t                        step,
                                           t_nrnb*                        nrnb) const
{
    pairlistSets_->construct(iLocality, pairSearch_.get(), nbat.get(), exclusions, step, nrnb);

    if (useGpu())
    {
        /* Launch the transfer of the pairlist to the GPU.
         *
         * NOTE: The launch overhead is currently not timed separately
         */
        Nbnxm::gpu_init_pairlist(gpu_nbv, pairlistSets().pairlistSet(iLocality).gpuList(), iLocality);
    }
}
