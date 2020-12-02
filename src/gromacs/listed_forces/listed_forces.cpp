/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2014,2015,2016,2017,2018 by the GROMACS development team.
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
/*! \internal \file
 *
 * \brief This file defines high-level functions for mdrun to compute
 * energies and forces for listed interactions.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_listed_forces
 */
#include "gmxpre.h"

#include "listed_forces.h"

#include <cassert>

#include <algorithm>
#include <array>
#include <numeric>

#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/listed_forces/bonded.h"
#include "gromacs/listed_forces/disre.h"
#include "gromacs/listed_forces/orires.h"
#include "gromacs/listed_forces/pairs.h"
#include "gromacs/listed_forces/position_restraints.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/enerdata_utils.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"

#include "listed_internal.h"
#include "manage_threading.h"
#include "utilities.h"

ListedForces::ListedForces(const gmx_ffparams_t&      ffparams,
                           const int                  numEnergyGroups,
                           const int                  numThreads,
                           const InteractionSelection interactionSelection,
                           FILE*                      fplog) :
    idefSelection_(ffparams),
    threading_(std::make_unique<bonded_threading_t>(numThreads, numEnergyGroups, fplog)),
    interactionSelection_(interactionSelection)
{
}

ListedForces::ListedForces(ListedForces&& o) noexcept = default;

ListedForces::~ListedForces() = default;

//! Copies the selection interactions from \p idefSrc to \p idef
static void selectInteractions(InteractionDefinitions*                  idef,
                               const InteractionDefinitions&            idefSrc,
                               const ListedForces::InteractionSelection interactionSelection)
{
    const bool selectPairs =
            interactionSelection.test(static_cast<int>(ListedForces::InteractionGroup::Pairs));
    const bool selectDihedrals =
            interactionSelection.test(static_cast<int>(ListedForces::InteractionGroup::Dihedrals));
    const bool selectAngles =
            interactionSelection.test(static_cast<int>(ListedForces::InteractionGroup::Angles));
    const bool selectRest =
            interactionSelection.test(static_cast<int>(ListedForces::InteractionGroup::Rest));

    for (int ftype = 0; ftype < F_NRE; ftype++)
    {
        const t_interaction_function& ifunc = interaction_function[ftype];
        if (ifunc.flags & IF_BOND)
        {
            bool assign = false;
            if (ifunc.flags & IF_PAIR)
            {
                assign = selectPairs;
            }
            else if (ifunc.flags & IF_DIHEDRAL)
            {
                assign = selectDihedrals;
            }
            else if (ifunc.flags & IF_ATYPE)
            {
                assign = selectAngles;
            }
            else
            {
                assign = selectRest;
            }
            if (assign)
            {
                idef->il[ftype] = idefSrc.il[ftype];
            }
            else
            {
                idef->il[ftype].clear();
            }
        }
    }
}

void ListedForces::setup(const InteractionDefinitions& domainIdef, const int numAtomsForce, const bool useGpu)
{
    if (interactionSelection_.all())
    {
        // Avoid the overhead of copying all interaction lists by simply setting the reference to the domain idef
        idef_ = &domainIdef;
    }
    else
    {
        idef_ = &idefSelection_;

        selectInteractions(&idefSelection_, domainIdef, interactionSelection_);

        idefSelection_.ilsort = domainIdef.ilsort;

        if (interactionSelection_.test(static_cast<int>(ListedForces::InteractionGroup::Rest)))
        {
            idefSelection_.iparams_posres   = domainIdef.iparams_posres;
            idefSelection_.iparams_fbposres = domainIdef.iparams_fbposres;
        }
        else
        {
            idefSelection_.iparams_posres.clear();
            idefSelection_.iparams_fbposres.clear();
        }
    }

    setup_bonded_threading(threading_.get(), numAtomsForce, useGpu, *idef_);

    if (idef_->ilsort == ilsortFE_SORTED)
    {
        forceBufferLambda_.resize(numAtomsForce * sizeof(rvec4) / sizeof(real));
        shiftForceBufferLambda_.resize(SHIFTS);
    }
}

namespace
{

using gmx::ArrayRef;

/*! \brief Return true if ftype is an explicit pair-listed LJ or
 * COULOMB interaction type: bonded LJ (usually 1-4), or special
 * listed non-bonded for FEP. */
bool isPairInteraction(int ftype)
{
    return ((ftype) >= F_LJ14 && (ftype) <= F_LJC_PAIRS_NB);
}

/*! \brief Zero thread-local output buffers */
void zero_thread_output(f_thread_t* f_t)
{
    constexpr int nelem_fa = sizeof(f_t->f[0]) / sizeof(real);

    for (int i = 0; i < f_t->nblock_used; i++)
    {
        int a0 = f_t->block_index[i] * reduction_block_size;
        int a1 = a0 + reduction_block_size;
        for (int a = a0; a < a1; a++)
        {
            for (int d = 0; d < nelem_fa; d++)
            {
                f_t->f[a][d] = 0;
            }
        }
    }

    for (int i = 0; i < SHIFTS; i++)
    {
        clear_rvec(f_t->fshift[i]);
    }
    for (int i = 0; i < F_NRE; i++)
    {
        f_t->ener[i] = 0;
    }
    for (int i = 0; i < egNR; i++)
    {
        for (int j = 0; j < f_t->grpp.nener; j++)
        {
            f_t->grpp.ener[i][j] = 0;
        }
    }
    for (int i = 0; i < efptNR; i++)
    {
        f_t->dvdl[i] = 0;
    }
}

/*! \brief The max thread number is arbitrary, we used a fixed number
 * to avoid memory management.  Using more than 16 threads is probably
 * never useful performance wise. */
#define MAX_BONDED_THREADS 256

/*! \brief Reduce thread-local force buffers */
void reduce_thread_forces(gmx::ArrayRef<gmx::RVec> force, const bonded_threading_t* bt, int nthreads)
{
    if (nthreads > MAX_BONDED_THREADS)
    {
        gmx_fatal(FARGS, "Can not reduce bonded forces on more than %d threads", MAX_BONDED_THREADS);
    }

    rvec* gmx_restrict f = as_rvec_array(force.data());

    const int numAtomsForce = bt->numAtomsForce;

    /* This reduction can run on any number of threads,
     * independently of bt->nthreads.
     * But if nthreads matches bt->nthreads (which it currently does)
     * the uniform distribution of the touched blocks over nthreads will
     * match the distribution of bonded over threads well in most cases,
     * which means that threads mostly reduce their own data which increases
     * the number of cache hits.
     */
#pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int b = 0; b < bt->nblock_used; b++)
    {
        try
        {
            int    ind = bt->block_index[b];
            rvec4* fp[MAX_BONDED_THREADS];

            /* Determine which threads contribute to this block */
            int nfb = 0;
            for (int ft = 0; ft < bt->nthreads; ft++)
            {
                if (bitmask_is_set(bt->mask[ind], ft))
                {
                    fp[nfb++] = bt->f_t[ft]->f;
                }
            }
            if (nfb > 0)
            {
                /* Reduce force buffers for threads that contribute */
                int a0 = ind * reduction_block_size;
                int a1 = (ind + 1) * reduction_block_size;
                /* It would be nice if we could pad f to avoid this min */
                a1 = std::min(a1, numAtomsForce);
                for (int a = a0; a < a1; a++)
                {
                    for (int fb = 0; fb < nfb; fb++)
                    {
                        rvec_inc(f[a], fp[fb][a]);
                    }
                }
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
}

/*! \brief Reduce thread-local forces, shift forces and energies */
void reduce_thread_output(gmx::ForceWithShiftForces* forceWithShiftForces,
                          real*                      ener,
                          gmx_grppairener_t*         grpp,
                          real*                      dvdl,
                          const bonded_threading_t*  bt,
                          const gmx::StepWorkload&   stepWork)
{
    assert(bt->haveBondeds);

    if (bt->nblock_used > 0)
    {
        /* Reduce the bonded force buffer */
        reduce_thread_forces(forceWithShiftForces->force(), bt, bt->nthreads);
    }

    rvec* gmx_restrict fshift = as_rvec_array(forceWithShiftForces->shiftForces().data());

    /* When necessary, reduce energy and virial using one thread only */
    if ((stepWork.computeEnergy || stepWork.computeVirial || stepWork.computeDhdl) && bt->nthreads > 1)
    {
        gmx::ArrayRef<const std::unique_ptr<f_thread_t>> f_t = bt->f_t;

        if (stepWork.computeVirial)
        {
            for (int i = 0; i < SHIFTS; i++)
            {
                for (int t = 1; t < bt->nthreads; t++)
                {
                    rvec_inc(fshift[i], f_t[t]->fshift[i]);
                }
            }
        }
        if (stepWork.computeEnergy)
        {
            for (int i = 0; i < F_NRE; i++)
            {
                for (int t = 1; t < bt->nthreads; t++)
                {
                    ener[i] += f_t[t]->ener[i];
                }
            }
            for (int i = 0; i < egNR; i++)
            {
                for (int j = 0; j < f_t[1]->grpp.nener; j++)
                {
                    for (int t = 1; t < bt->nthreads; t++)
                    {
                        grpp->ener[i][j] += f_t[t]->grpp.ener[i][j];
                    }
                }
            }
        }
        if (stepWork.computeDhdl)
        {
            for (int i = 0; i < efptNR; i++)
            {

                for (int t = 1; t < bt->nthreads; t++)
                {
                    dvdl[i] += f_t[t]->dvdl[i];
                }
            }
        }
    }
}

/*! \brief Returns the bonded kernel flavor
 *
 * Note that energies are always requested when the virial
 * is requested (performance gain would be small).
 * Note that currently we do not have bonded kernels that
 * do not compute forces.
 */
BondedKernelFlavor selectBondedKernelFlavor(const gmx::StepWorkload& stepWork,
                                            const bool               useSimdKernels,
                                            const bool               havePerturbedInteractions)
{
    BondedKernelFlavor flavor;
    if (stepWork.computeEnergy || stepWork.computeVirial)
    {
        if (stepWork.computeVirial)
        {
            flavor = BondedKernelFlavor::ForcesAndVirialAndEnergy;
        }
        else
        {
            flavor = BondedKernelFlavor::ForcesAndEnergy;
        }
    }
    else
    {
        if (useSimdKernels && !havePerturbedInteractions)
        {
            flavor = BondedKernelFlavor::ForcesSimdWhenAvailable;
        }
        else
        {
            flavor = BondedKernelFlavor::ForcesNoSimd;
        }
    }

    return flavor;
}

/*! \brief Calculate one element of the list of bonded interactions
    for this thread */
real calc_one_bond(int                           thread,
                   int                           ftype,
                   const InteractionDefinitions& idef,
                   ArrayRef<const int>           iatoms,
                   const int                     numNonperturbedInteractions,
                   const WorkDivision&           workDivision,
                   const rvec                    x[],
                   rvec4                         f[],
                   rvec                          fshift[],
                   const t_forcerec*             fr,
                   const t_pbc*                  pbc,
                   gmx_grppairener_t*            grpp,
                   t_nrnb*                       nrnb,
                   const real*                   lambda,
                   real*                         dvdl,
                   const t_mdatoms*              md,
                   t_fcdata*                     fcd,
                   const gmx::StepWorkload&      stepWork,
                   int*                          global_atom_index)
{
    GMX_ASSERT(idef.ilsort == ilsortNO_FE || idef.ilsort == ilsortFE_SORTED,
               "The topology should be marked either as no FE or sorted on FE");

    const bool havePerturbedInteractions =
            (idef.ilsort == ilsortFE_SORTED && numNonperturbedInteractions < iatoms.ssize());
    BondedKernelFlavor flavor =
            selectBondedKernelFlavor(stepWork, fr->use_simd_kernels, havePerturbedInteractions);
    int efptFTYPE;
    if (IS_RESTRAINT_TYPE(ftype))
    {
        efptFTYPE = efptRESTRAINT;
    }
    else
    {
        efptFTYPE = efptBONDED;
    }

    const int nat1   = interaction_function[ftype].nratoms + 1;
    const int nbonds = iatoms.ssize() / nat1;

    GMX_ASSERT(fr->gpuBonded != nullptr || workDivision.end(ftype) == iatoms.ssize(),
               "The thread division should match the topology");

    const int nb0 = workDivision.bound(ftype, thread);
    const int nbn = workDivision.bound(ftype, thread + 1) - nb0;

    ArrayRef<const t_iparams> iparams = idef.iparams;

    real v = 0;
    if (!isPairInteraction(ftype))
    {
        if (ftype == F_CMAP)
        {
            /* TODO The execution time for CMAP dihedrals might be
               nice to account to its own subtimer, but first
               wallcycle needs to be extended to support calling from
               multiple threads. */
            v = cmap_dihs(nbn, iatoms.data() + nb0, iparams.data(), &idef.cmap_grid, x, f, fshift,
                          pbc, lambda[efptFTYPE], &(dvdl[efptFTYPE]), md, fcd, global_atom_index);
        }
        else
        {
            v = calculateSimpleBond(ftype, nbn, iatoms.data() + nb0, iparams.data(), x, f, fshift,
                                    pbc, lambda[efptFTYPE], &(dvdl[efptFTYPE]), md, fcd,
                                    global_atom_index, flavor);
        }
    }
    else
    {
        /* TODO The execution time for pairs might be nice to account
           to its own subtimer, but first wallcycle needs to be
           extended to support calling from multiple threads. */
        do_pairs(ftype, nbn, iatoms.data() + nb0, iparams.data(), x, f, fshift, pbc, lambda, dvdl,
                 md, fr, havePerturbedInteractions, stepWork, grpp, global_atom_index);
    }

    if (thread == 0)
    {
        inc_nrnb(nrnb, nrnbIndex(ftype), nbonds);
    }

    return v;
}

} // namespace

/*! \brief Compute the bonded part of the listed forces, parallelized over threads
 */
static void calcBondedForces(const InteractionDefinitions& idef,
                             bonded_threading_t*           bt,
                             const rvec                    x[],
                             const t_forcerec*             fr,
                             const t_pbc*                  pbc_null,
                             rvec*                         fshiftMasterBuffer,
                             gmx_enerdata_t*               enerd,
                             t_nrnb*                       nrnb,
                             const real*                   lambda,
                             real*                         dvdl,
                             const t_mdatoms*              md,
                             t_fcdata*                     fcd,
                             const gmx::StepWorkload&      stepWork,
                             int*                          global_atom_index)
{
#pragma omp parallel for num_threads(bt->nthreads) schedule(static)
    for (int thread = 0; thread < bt->nthreads; thread++)
    {
        try
        {
            f_thread_t& threadBuffers = *bt->f_t[thread];
            int         ftype;
            real *      epot, v;
            /* thread stuff */
            rvec*              fshift;
            real*              dvdlt;
            gmx_grppairener_t* grpp;

            zero_thread_output(&threadBuffers);

            rvec4* ft = threadBuffers.f;

            /* Thread 0 writes directly to the main output buffers.
             * We might want to reconsider this.
             */
            if (thread == 0)
            {
                fshift = fshiftMasterBuffer;
                epot   = enerd->term;
                grpp   = &enerd->grpp;
                dvdlt  = dvdl;
            }
            else
            {
                fshift = as_rvec_array(threadBuffers.fshift.data());
                epot   = threadBuffers.ener;
                grpp   = &threadBuffers.grpp;
                dvdlt  = threadBuffers.dvdl;
            }
            /* Loop over all bonded force types to calculate the bonded forces */
            for (ftype = 0; (ftype < F_NRE); ftype++)
            {
                const InteractionList& ilist = idef.il[ftype];
                if (!ilist.empty() && ftype_is_bonded_potential(ftype))
                {
                    ArrayRef<const int> iatoms = gmx::makeConstArrayRef(ilist.iatoms);
                    v = calc_one_bond(thread, ftype, idef, iatoms, idef.numNonperturbedInteractions[ftype],
                                      bt->workDivision, x, ft, fshift, fr, pbc_null, grpp, nrnb,
                                      lambda, dvdlt, md, fcd, stepWork, global_atom_index);
                    epot[ftype] += v;
                }
            }
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
}

bool ListedForces::haveRestraints(const t_fcdata& fcdata) const
{
    GMX_ASSERT(fcdata.orires && fcdata.disres, "NMR restraints objects should be set up");

    return (!idef_->il[F_POSRES].empty() || !idef_->il[F_FBPOSRES].empty() || fcdata.orires->nr > 0
            || fcdata.disres->nres > 0);
}

bool ListedForces::haveCpuBondeds() const
{
    return threading_->haveBondeds;
}

bool ListedForces::haveCpuListedForces(const t_fcdata& fcdata) const
{
    return haveCpuBondeds() || haveRestraints(fcdata);
}

namespace
{

/*! \brief Calculates all listed force interactions. */
void calc_listed(struct gmx_wallcycle*         wcycle,
                 const InteractionDefinitions& idef,
                 bonded_threading_t*           bt,
                 const rvec                    x[],
                 gmx::ForceOutputs*            forceOutputs,
                 const t_forcerec*             fr,
                 const t_pbc*                  pbc,
                 gmx_enerdata_t*               enerd,
                 t_nrnb*                       nrnb,
                 const real*                   lambda,
                 const t_mdatoms*              md,
                 t_fcdata*                     fcd,
                 int*                          global_atom_index,
                 const gmx::StepWorkload&      stepWork)
{
    if (bt->haveBondeds)
    {
        gmx::ForceWithShiftForces& forceWithShiftForces = forceOutputs->forceWithShiftForces();

        wallcycle_sub_start(wcycle, ewcsLISTED);
        /* The dummy array is to have a place to store the dhdl at other values
           of lambda, which will be thrown away in the end */
        real dvdl[efptNR] = { 0 };
        calcBondedForces(idef, bt, x, fr, fr->bMolPBC ? pbc : nullptr,
                         as_rvec_array(forceWithShiftForces.shiftForces().data()), enerd, nrnb,
                         lambda, dvdl, md, fcd, stepWork, global_atom_index);
        wallcycle_sub_stop(wcycle, ewcsLISTED);

        wallcycle_sub_start(wcycle, ewcsLISTED_BUF_OPS);
        reduce_thread_output(&forceWithShiftForces, enerd->term, &enerd->grpp, dvdl, bt, stepWork);

        if (stepWork.computeDhdl)
        {
            for (int i = 0; i < efptNR; i++)
            {
                enerd->dvdl_nonlin[i] += dvdl[i];
            }
        }
        wallcycle_sub_stop(wcycle, ewcsLISTED_BUF_OPS);
    }

    /* Copy the sum of violations for the distance restraints from fcd */
    if (fcd)
    {
        enerd->term[F_DISRESVIOL] = fcd->disres->sumviol;
    }
}

/*! \brief As calc_listed(), but only determines the potential energy
 * for the perturbed interactions.
 *
 * The shift forces in fr are not affected.
 */
void calc_listed_lambda(const InteractionDefinitions& idef,
                        bonded_threading_t*           bt,
                        const rvec                    x[],
                        const t_forcerec*             fr,
                        const struct t_pbc*           pbc,
                        gmx::ArrayRef<real>           forceBufferLambda,
                        gmx::ArrayRef<gmx::RVec>      shiftForceBufferLambda,
                        gmx_grppairener_t*            grpp,
                        real*                         epot,
                        gmx::ArrayRef<real>           dvdl,
                        t_nrnb*                       nrnb,
                        const real*                   lambda,
                        const t_mdatoms*              md,
                        t_fcdata*                     fcd,
                        int*                          global_atom_index)
{
    WorkDivision& workDivision = bt->foreignLambdaWorkDivision;

    const t_pbc* pbc_null;
    if (fr->bMolPBC)
    {
        pbc_null = pbc;
    }
    else
    {
        pbc_null = nullptr;
    }

    /* We already have the forces, so we use temp buffers here */
    std::fill(forceBufferLambda.begin(), forceBufferLambda.end(), 0.0_real);
    std::fill(shiftForceBufferLambda.begin(), shiftForceBufferLambda.end(),
              gmx::RVec{ 0.0_real, 0.0_real, 0.0_real });
    rvec4* f      = reinterpret_cast<rvec4*>(forceBufferLambda.data());
    rvec*  fshift = as_rvec_array(shiftForceBufferLambda.data());

    /* Loop over all bonded force types to calculate the bonded energies */
    for (int ftype = 0; (ftype < F_NRE); ftype++)
    {
        if (ftype_is_bonded_potential(ftype))
        {
            const InteractionList& ilist = idef.il[ftype];
            /* Create a temporary iatom list with only perturbed interactions */
            const int           numNonperturbed = idef.numNonperturbedInteractions[ftype];
            ArrayRef<const int> iatomsPerturbed = gmx::constArrayRefFromArray(
                    ilist.iatoms.data() + numNonperturbed, ilist.size() - numNonperturbed);
            if (!iatomsPerturbed.empty())
            {
                /* Set the work range of thread 0 to the perturbed bondeds */
                workDivision.setBound(ftype, 0, 0);
                workDivision.setBound(ftype, 1, iatomsPerturbed.ssize());

                gmx::StepWorkload tempFlags;
                tempFlags.computeEnergy = true;
                real v = calc_one_bond(0, ftype, idef, iatomsPerturbed, iatomsPerturbed.ssize(),
                                       workDivision, x, f, fshift, fr, pbc_null, grpp, nrnb, lambda,
                                       dvdl.data(), md, fcd, tempFlags, global_atom_index);
                epot[ftype] += v;
            }
        }
    }
}

} // namespace

void ListedForces::calculate(struct gmx_wallcycle*                     wcycle,
                             const matrix                              box,
                             const t_lambda*                           fepvals,
                             const t_commrec*                          cr,
                             const gmx_multisim_t*                     ms,
                             gmx::ArrayRefWithPadding<const gmx::RVec> coordinates,
                             gmx::ArrayRef<const gmx::RVec>            xWholeMolecules,
                             t_fcdata*                                 fcdata,
                             history_t*                                hist,
                             gmx::ForceOutputs*                        forceOutputs,
                             const t_forcerec*                         fr,
                             const struct t_pbc*                       pbc,
                             gmx_enerdata_t*                           enerd,
                             t_nrnb*                                   nrnb,
                             const real*                               lambda,
                             const t_mdatoms*                          md,
                             int*                                      global_atom_index,
                             const gmx::StepWorkload&                  stepWork)
{
    if (interactionSelection_.none() || !stepWork.computeListedForces)
    {
        return;
    }

    const InteractionDefinitions& idef = *idef_;

    // Todo: replace all rvec use here with ArrayRefWithPadding
    const rvec* x = as_rvec_array(coordinates.paddedArrayRef().data());

    t_pbc pbc_full; /* Full PBC is needed for position restraints */
    if (haveRestraints(*fcdata))
    {
        if (!idef.il[F_POSRES].empty() || !idef.il[F_FBPOSRES].empty())
        {
            /* Not enough flops to bother counting */
            set_pbc(&pbc_full, fr->pbcType, box);
        }

        /* TODO Use of restraints triggers further function calls
           inside the loop over calc_one_bond(), but those are too
           awkward to account to this subtimer properly in the present
           code. We don't test / care much about performance with
           restraints, anyway. */
        wallcycle_sub_start(wcycle, ewcsRESTRAINTS);

        if (!idef.il[F_POSRES].empty())
        {
            posres_wrapper(nrnb, idef, &pbc_full, x, enerd, lambda, fr, &forceOutputs->forceWithVirial());
        }

        if (!idef.il[F_FBPOSRES].empty())
        {
            fbposres_wrapper(nrnb, idef, &pbc_full, x, enerd, fr, &forceOutputs->forceWithVirial());
        }

        /* Do pre force calculation stuff which might require communication */
        if (fcdata->orires->nr > 0)
        {
            GMX_ASSERT(!xWholeMolecules.empty(), "Need whole molecules for orienation restraints");
            enerd->term[F_ORIRESDEV] = calc_orires_dev(
                    ms, idef.il[F_ORIRES].size(), idef.il[F_ORIRES].iatoms.data(), idef.iparams.data(),
                    md, xWholeMolecules, x, fr->bMolPBC ? pbc : nullptr, fcdata->orires, hist);
        }
        if (fcdata->disres->nres > 0)
        {
            calc_disres_R_6(cr, ms, idef.il[F_DISRES].size(), idef.il[F_DISRES].iatoms.data(), x,
                            fr->bMolPBC ? pbc : nullptr, fcdata->disres, hist);
        }

        wallcycle_sub_stop(wcycle, ewcsRESTRAINTS);
    }

    calc_listed(wcycle, idef, threading_.get(), x, forceOutputs, fr, pbc, enerd, nrnb, lambda, md,
                fcdata, global_atom_index, stepWork);

    /* Check if we have to determine energy differences
     * at foreign lambda's.
     */
    if (fepvals->n_lambda > 0 && stepWork.computeDhdl)
    {
        real dvdl[efptNR] = { 0 };
        if (!idef.il[F_POSRES].empty())
        {
            posres_wrapper_lambda(wcycle, fepvals, idef, &pbc_full, x, enerd, lambda, fr);
        }
        if (idef.ilsort != ilsortNO_FE)
        {
            wallcycle_sub_start(wcycle, ewcsLISTED_FEP);
            if (idef.ilsort != ilsortFE_SORTED)
            {
                gmx_incons("The bonded interactions are not sorted for free energy");
            }
            for (int i = 0; i < 1 + enerd->foreignLambdaTerms.numLambdas(); i++)
            {
                real lam_i[efptNR];

                reset_foreign_enerdata(enerd);
                for (int j = 0; j < efptNR; j++)
                {
                    lam_i[j] = (i == 0 ? lambda[j] : fepvals->all_lambda[j][i - 1]);
                }
                calc_listed_lambda(idef, threading_.get(), x, fr, pbc, forceBufferLambda_,
                                   shiftForceBufferLambda_, &(enerd->foreign_grpp), enerd->foreign_term,
                                   dvdl, nrnb, lam_i, md, fcdata, global_atom_index);
                sum_epot(enerd->foreign_grpp, enerd->foreign_term);
                const double dvdlSum = std::accumulate(std::begin(dvdl), std::end(dvdl), 0.);
                std::fill(std::begin(dvdl), std::end(dvdl), 0.0);
                enerd->foreignLambdaTerms.accumulate(i, enerd->foreign_term[F_EPOT], dvdlSum);
            }
            wallcycle_sub_stop(wcycle, ewcsLISTED_FEP);
        }
    }
}
