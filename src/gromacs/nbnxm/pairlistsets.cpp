/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021, by the GROMACS development team, led by
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

#include "pairlistsets.h"

#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/utility/listoflists.h"

#include "pairlistset.h"
#include "pairsearch.h"

void PairlistSets::construct(const gmx::InteractionLocality iLocality,
                             PairSearch*                    pairSearch,
                             nbnxn_atomdata_t*              nbat,
                             const gmx::ListOfLists<int>&   exclusions,
                             const int64_t                  step,
                             t_nrnb*                        nrnb)
{
    const auto& gridSet = pairSearch->gridSet();
    const auto* ddZones = gridSet.domainSetup().zones;

    /* The Nbnxm code can also work with more exclusions than those in i-zones only
     * when using DD, but the equality check can catch more issues.
     */
    GMX_RELEASE_ASSERT(
            exclusions.empty() || (!ddZones && exclusions.ssize() == gridSet.numRealAtomsTotal())
                    || (ddZones && exclusions.ssize() == ddZones->cg_range[ddZones->iZones.size()]),
            "exclusions should either be empty or the number of lists should match the number of "
            "local i-atoms");

    pairlistSet(iLocality).constructPairlists(gridSet,
                                              pairSearch->work(),
                                              nbat,
                                              exclusions,
                                              minimumIlistCountForGpuBalancing_,
                                              nrnb,
                                              &pairSearch->cycleCounting_);

    if (iLocality == gmx::InteractionLocality::Local)
    {
        outerListCreationStep_ = step;
    }
    else
    {
        GMX_RELEASE_ASSERT(outerListCreationStep_ == step,
                           "Outer list should be created at the same step as the inner list");
    }

    /* Special performance logging stuff (env.var. GMX_NBNXN_CYCLE) */
    if (iLocality == gmx::InteractionLocality::Local)
    {
        pairSearch->cycleCounting_.searchCount_++;
    }
    if (pairSearch->cycleCounting_.recordCycles_
        && (!pairSearch->gridSet().domainSetup().haveMultipleDomains
            || iLocality == gmx::InteractionLocality::NonLocal)
        && pairSearch->cycleCounting_.searchCount_ % 100 == 0)
    {
        pairSearch->cycleCounting_.printCycles(stderr, pairSearch->work());
    }
}
