/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2008, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2017,2018 by the GROMACS development team.
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
#include "gmxpre.h"

#include "topsort.h"

#include <cstdio>

#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/smalloc.h"

static gmx_bool ip_pert(int ftype, const t_iparams* ip)
{
    gmx_bool bPert;
    int      i;

    if (NRFPB(ftype) == 0)
    {
        return FALSE;
    }

    switch (ftype)
    {
        case F_BONDS:
        case F_G96BONDS:
        case F_HARMONIC:
        case F_ANGLES:
        case F_G96ANGLES:
        case F_IDIHS:
            bPert = (ip->harmonic.rA != ip->harmonic.rB || ip->harmonic.krA != ip->harmonic.krB);
            break;
        case F_MORSE:
            bPert = (ip->morse.b0A != ip->morse.b0B || ip->morse.cbA != ip->morse.cbB
                     || ip->morse.betaA != ip->morse.betaB);
            break;
        case F_RESTRBONDS:
            bPert = (ip->restraint.lowA != ip->restraint.lowB || ip->restraint.up1A != ip->restraint.up1B
                     || ip->restraint.up2A != ip->restraint.up2B
                     || ip->restraint.kA != ip->restraint.kB);
            break;
        case F_UREY_BRADLEY:
            bPert = (ip->u_b.thetaA != ip->u_b.thetaB || ip->u_b.kthetaA != ip->u_b.kthetaB
                     || ip->u_b.r13A != ip->u_b.r13B || ip->u_b.kUBA != ip->u_b.kUBB);
            break;
        case F_PDIHS:
        case F_PIDIHS:
        case F_ANGRES:
        case F_ANGRESZ:
            bPert = (ip->pdihs.phiA != ip->pdihs.phiB || ip->pdihs.cpA != ip->pdihs.cpB);
            break;
        case F_RBDIHS:
            bPert = FALSE;
            for (i = 0; i < NR_RBDIHS; i++)
            {
                if (ip->rbdihs.rbcA[i] != ip->rbdihs.rbcB[i])
                {
                    bPert = TRUE;
                }
            }
            break;
        case F_TABBONDS:
        case F_TABBONDSNC:
        case F_TABANGLES:
        case F_TABDIHS: bPert = (ip->tab.kA != ip->tab.kB); break;
        case F_POSRES:
            bPert = FALSE;
            for (i = 0; i < DIM; i++)
            {
                if (ip->posres.pos0A[i] != ip->posres.pos0B[i] || ip->posres.fcA[i] != ip->posres.fcB[i])
                {
                    bPert = TRUE;
                }
            }
            break;
        case F_DIHRES:
            bPert = ((ip->dihres.phiA != ip->dihres.phiB) || (ip->dihres.dphiA != ip->dihres.dphiB)
                     || (ip->dihres.kfacA != ip->dihres.kfacB));
            break;
        case F_LJ14:
            bPert = (ip->lj14.c6A != ip->lj14.c6B || ip->lj14.c12A != ip->lj14.c12B);
            break;
        case F_CMAP: bPert = FALSE; break;
        case F_RESTRANGLES:
        case F_RESTRDIHS:
        case F_CBTDIHS:
            gmx_fatal(FARGS, "Function type %s does not support currentely free energy calculations",
                      interaction_function[ftype].longname);
        default:
            gmx_fatal(FARGS, "Function type %s not implemented in ip_pert",
                      interaction_function[ftype].longname);
    }

    return bPert;
}

static gmx_bool ip_q_pert(int                         ftype,
                          const InteractionListEntry& ilistEntry,
                          const t_iparams*            ip,
                          const real*                 qA,
                          const real*                 qB)
{
    /* 1-4 interactions do not have the charges stored in the iparams list,
     * so we need a separate check for those.
     */
    return (ip_pert(ftype, ip + ilistEntry.parameterType)
            || (ftype == F_LJ14
                && (qA[ilistEntry.atoms[0]] != qB[ilistEntry.atoms[0]]
                    || qA[ilistEntry.atoms[1]] != qB[ilistEntry.atoms[1]])));
}

gmx_bool gmx_mtop_bondeds_free_energy(const gmx_mtop_t* mtop)
{
    const gmx_ffparams_t* ffparams = &mtop->ffparams;

    /* Loop over all the function types and compare the A/B parameters */
    gmx_bool bPert = FALSE;
    for (int i = 0; i < ffparams->numTypes(); i++)
    {
        int ftype = ffparams->functype[i];
        if (interaction_function[ftype].flags & IF_BOND)
        {
            if (ip_pert(ftype, &ffparams->iparams[i]))
            {
                bPert = TRUE;
            }
        }
    }

    /* Check perturbed charges for 1-4 interactions */
    for (const gmx_molblock_t& molb : mtop->molblock)
    {
        const t_atom* atom = mtop->moltype[molb.type].atoms.atom;
        for (const auto entry : mtop->moltype[molb.type].ilist[F_LJ14])
        {
            if (atom[entry.atoms[0]].q != atom[entry.atoms[0]].qB
                || atom[entry.atoms[1]].q != atom[entry.atoms[1]].qB)
            {
                bPert = TRUE;
            }
        }
    }

    return bPert;
}

void gmx_sort_ilist_fe(InteractionDefinitions* idef, const real* qA, const real* qB)
{
    if (qB == nullptr)
    {
        qB = qA;
    }

    const t_iparams* iparams                   = idef->iparams.data();
    bool             havePerturbedInteractions = false;


    for (int ftype = 0; ftype < F_NRE; ftype++)
    {
        if (interaction_function[ftype].flags & IF_BOND)
        {
            InteractionList& ilist       = idef->il[ftype];
            const int        nral        = NRAL(ftype);
            const auto       itBegin     = ilist.begin();
            const auto       itEnd       = ilist.end();
            auto             newIterator = itBegin;
            InteractionList  perturbed(ftype);
            int              numPerturbed = 0;
            for (auto it = itBegin; it != itEnd; it++)
            {
                const auto entry = *it;
                /* Check if this interaction is perturbed */
                if (ip_q_pert(ftype, entry, iparams, qA, qB))
                {
                    /* Copy to the perturbed buffer */
                    perturbed.push_back(entry.parameterType, entry.atoms);
                    numPerturbed++;

                    havePerturbedInteractions = true;
                }
                else
                {
                    if (numPerturbed)
                    {
                        /* Move entry */
                        (*newIterator).parameterType = entry.parameterType;
                        for (int a = 0; a < nral; a++)
                        {
                            (*newIterator).atoms[a] = entry.atoms[a];
                        }
                    }
                    newIterator++;
                }
            }
            /* Now we know the number of non-perturbed interactions */
            idef->numNonperturbedInteractions[ftype] =
                    ilist.rawIndices().size() - numPerturbed * (1 + nral);

            /* Copy back the perturbed interactions */
            for (const auto entry : perturbed)
            {
                (*newIterator).parameterType = entry.parameterType;
                for (int a = 0; a < nral; a++)
                {
                    (*newIterator).atoms[a] = entry.atoms[a];
                }
                newIterator++;
            }
            GMX_RELEASE_ASSERT(newIterator == itEnd,
                               "The total number of entries should be unchanged");

            if (debug)
            {
                const int numNonperturbed = idef->numNonperturbedInteractions[ftype] / (1 + nral);
                fprintf(debug, "%s non-pert %d pert %zu\n", interaction_function[ftype].longname,
                        numNonperturbed, idef->il[ftype].numInteractions() - numNonperturbed);
            }
        }
    }

    idef->ilsort = (havePerturbedInteractions ? ilsortFE_SORTED : ilsortNO_FE);
}
