/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2017 The GROMACS development team.
 * Copyright (c) 2018,2019,2020, by the GROMACS development team, led by
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

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <unordered_map>

#include "gromacs/commandline/pargs.h"
#include "gromacs/commandline/viewit.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/matio.h"
#include "gromacs/fileio/pdbio.h"
#include "gromacs/fileio/tpxio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/gmxana/gstat.h"
#include "gromacs/listed_forces/disre.h"
#include "gromacs/math/do_fit.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pbcutil/rmpbc.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/smalloc.h"

typedef struct
{
    int  n;
    real v;
} t_toppop;

static t_toppop* top  = nullptr;
static int       ntop = 0;

typedef struct
{
    int   nv, nframes;
    real  sumv, averv, maxv;
    real *aver1, *aver2, *aver_3, *aver_6;
} t_dr_result;

static void init5(int n)
{
    ntop = n;
    snew(top, ntop);
}

static void reset5()
{
    int i;

    for (i = 0; (i < ntop); i++)
    {
        top[i].n = -1;
        top[i].v = 0;
    }
}

static void add5(int ndr, real viol)
{
    int i, mini;

    mini = 0;
    for (i = 1; (i < ntop); i++)
    {
        if (top[i].v < top[mini].v)
        {
            mini = i;
        }
    }
    if (viol > top[mini].v)
    {
        top[mini].v = viol;
        top[mini].n = ndr;
    }
}

static void print5(FILE* fp)
{
    int i;

    std::sort(top, top + ntop, [](const t_toppop& a, const t_toppop& b) { return a.v > b.v; }); // reverse sort
    fprintf(fp, "Index:");
    for (i = 0; (i < ntop); i++)
    {
        fprintf(fp, " %6d", top[i].n);
    }
    fprintf(fp, "\nViol: ");
    for (i = 0; (i < ntop); i++)
    {
        fprintf(fp, " %6.3f", top[i].v);
    }
    fprintf(fp, "\n");
}

static void check_viol(FILE*                          log,
                       const InteractionList&         disres,
                       gmx::ArrayRef<const t_iparams> forceparams,
                       rvec                           x[],
                       rvec4                          f[],
                       t_pbc*                         pbc,
                       t_dr_result                    dr[],
                       int                            clust_id,
                       int                            isize,
                       const int                      index[],
                       real                           vvindex[],
                       t_disresdata*                  disresdata)
{
    int             j, nat, n, type, nviol, ndr, label;
    real            rt, mviol, tviol, viol, lam, dvdl, drt;
    rvec*           fshift;
    static gmx_bool bFirst = TRUE;

    lam   = 0;
    dvdl  = 0;
    tviol = 0;
    nviol = 0;
    mviol = 0;
    ndr   = 0;
    if (ntop)
    {
        reset5();
    }
    gmx::ArrayRef<const int> forceatoms = disres.rawIndices();
    for (j = 0; (j < isize); j++)
    {
        vvindex[j] = 0;
    }
    nat = interaction_function[F_DISRES].nratoms + 1;
    // Check internal consistency of disres.label
    // The label for a distance restraint should be at most one larger
    // than the previous label.
    int label_old = forceparams[forceatoms[0]].disres.label;
    for (gmx::index i = 0; i < forceatoms.ssize(); i += nat)
    {
        type  = forceatoms[i];
        label = forceparams[type].disres.label;
        if ((label == label_old) || (label == label_old + 1))
        {
            label_old = label;
        }
        else
        {
            gmx_fatal(FARGS,
                      "Label mismatch in distance restrains. Label for restraint %d is %d, "
                      "expected it to be either %d or %d",
                      int(i) / nat, label, label_old, label_old + 1);
        }
    }

    // Set up t_fcdata, only needed for calling ta_disres()
    t_fcdata fcd;
    fcd.disres = disresdata;

    // Get offset for label index
    label_old = forceparams[forceatoms[0]].disres.label;
    for (gmx::index i = 0; i < forceatoms.ssize();)
    {
        type  = forceatoms[i];
        n     = 0;
        label = forceparams[type].disres.label - label_old;
        if (debug)
        {
            fprintf(debug, "DISRE: ndr = %d, label = %d  i=%d, n =%d\n", ndr, label, int(i), n);
        }
        do
        {
            n += nat;
        } while (((i + n) < forceatoms.ssize())
                 && (forceparams[forceatoms[i + n]].disres.label == label + label_old));

        calc_disres_R_6(nullptr, nullptr, forceatoms.subArray(i, n), x, pbc, disresdata, nullptr);

        if (disresdata->Rt_6[label] <= 0)
        {
            gmx_fatal(FARGS, "ndr = %d, rt_6 = %f", ndr, disresdata->Rt_6[label]);
        }

        rt = gmx::invsixthroot(disresdata->Rt_6[label]);
        dr[clust_id].aver1[ndr] += rt;
        dr[clust_id].aver2[ndr] += gmx::square(rt);
        drt = 1.0 / gmx::power3(rt);
        dr[clust_id].aver_3[ndr] += drt;
        dr[clust_id].aver_6[ndr] += disresdata->Rt_6[label];

        snew(fshift, SHIFTS);
        ta_disres(n, &forceatoms[i], forceparams.data(), x, f, fshift, pbc, lam, &dvdl, nullptr,
                  &fcd, nullptr);
        sfree(fshift);
        viol = disresdata->sumviol;

        if (viol > 0)
        {
            nviol++;
            if (ntop)
            {
                add5(forceparams[type].disres.label, viol);
            }
            if (viol > mviol)
            {
                mviol = viol;
            }
            tviol += viol;
            for (j = 0; (j < isize); j++)
            {
                if (index[j] == forceparams[type].disres.label)
                {
                    vvindex[j] = gmx::invsixthroot(disresdata->Rt_6[label]);
                }
            }
        }
        ndr++;
        i += n;
    }
    dr[clust_id].nv    = nviol;
    dr[clust_id].maxv  = mviol;
    dr[clust_id].sumv  = tviol;
    dr[clust_id].averv = tviol / ndr;
    dr[clust_id].nframes++;

    if (bFirst)
    {
        fprintf(stderr, "\nThere are %d restraints and %zu pairs\n", ndr, disres.numInteractions());
        bFirst = FALSE;
    }
    if (ntop)
    {
        print5(log);
    }
}

typedef struct
{
    int      label;
    gmx_bool bCore;
    real     up1, r, rT3, rT6, viol, violT3, violT6;
} t_dr_stats;

static void dump_dump(FILE* log, int ndr, t_dr_stats drs[])
{
    static const char* core[] = { "All restraints", "Core restraints" };
    static const char* tp[]   = { "linear", "third power", "sixth power" };
    real               viol_tot, viol_max, viol = 0;
    gmx_bool           bCore;
    int                nviol, nrestr;
    int                i, kkk;

    for (int iCore = 0; iCore < 2; iCore++)
    {
        bCore = (iCore == 1);
        for (kkk = 0; (kkk < 3); kkk++)
        {
            viol_tot = 0;
            viol_max = 0;
            nviol    = 0;
            nrestr   = 0;
            for (i = 0; (i < ndr); i++)
            {
                if (!bCore || drs[i].bCore)
                {
                    switch (kkk)
                    {
                        case 0: viol = drs[i].viol; break;
                        case 1: viol = drs[i].violT3; break;
                        case 2: viol = drs[i].violT6; break;
                        default: gmx_incons("Dumping violations");
                    }
                    viol_max = std::max(viol_max, viol);
                    if (viol > 0)
                    {
                        nviol++;
                    }
                    viol_tot += viol;
                    nrestr++;
                }
            }
            if ((nrestr > 0) || (bCore && (nrestr < ndr)))
            {
                fprintf(log, "\n");
                fprintf(log, "+++++++ %s ++++++++\n", core[bCore]);
                fprintf(log, "+++++++ Using %s averaging: ++++++++\n", tp[kkk]);
                fprintf(log, "Sum of violations: %8.3f nm\n", viol_tot);
                if (nrestr > 0)
                {
                    fprintf(log, "Average violation: %8.3f nm\n", viol_tot / nrestr);
                }
                fprintf(log, "Largest violation: %8.3f nm\n", viol_max);
                fprintf(log, "Number of violated restraints: %d/%d\n", nviol, nrestr);
            }
        }
    }
}

static void dump_viol(FILE* log, int ndr, t_dr_stats* drs, gmx_bool bLinear)
{
    int i;

    fprintf(log, "Restr. Core     Up1     <r>   <rT3>   <rT6>  <viol><violT3><violT6>\n");
    for (i = 0; (i < ndr); i++)
    {
        if (bLinear && (drs[i].viol == 0))
        {
            break;
        }
        fprintf(log, "%6d%5s%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f\n", drs[i].label,
                yesno_names[drs[i].bCore], drs[i].up1, drs[i].r, drs[i].rT3, drs[i].rT6,
                drs[i].viol, drs[i].violT3, drs[i].violT6);
    }
}

static gmx_bool is_core(int i, int isize, const int index[])
{
    int      kk;
    gmx_bool bIC = FALSE;

    for (kk = 0; !bIC && (kk < isize); kk++)
    {
        bIC = (index[kk] == i);
    }

    return bIC;
}

static void dump_stats(FILE*                          log,
                       int                            nsteps,
                       const t_disresdata&            dd,
                       const InteractionList&         disres,
                       gmx::ArrayRef<const t_iparams> ip,
                       t_dr_result*                   dr,
                       int                            isize,
                       int                            index[],
                       t_atoms*                       atoms)
{
    t_dr_stats* drs;

    fprintf(log, "\n");
    fprintf(log, "++++++++++++++ STATISTICS ++++++++++++++++++++++++\n");
    snew(drs, dd.nres);
    for (const auto entry : disres)
    {
        // Note that the restraint i can be used by multiple pairs
        const int i = entry.parameterType - dd.type_min;
        GMX_RELEASE_ASSERT(i >= 0 && i < dd.nres, "The restraint index should be in range");

        drs[i].label  = ip[entry.parameterType].disres.label;
        drs[i].bCore  = is_core(drs[i].label, isize, index);
        drs[i].up1    = ip[entry.parameterType].disres.up1;
        drs[i].r      = dr->aver1[i] / nsteps;
        drs[i].rT3    = gmx::invcbrt(dr->aver_3[i] / nsteps);
        drs[i].rT6    = gmx::invsixthroot(dr->aver_6[i] / nsteps);
        drs[i].viol   = std::max(0.0, static_cast<double>(drs[i].r - drs[i].up1));
        drs[i].violT3 = std::max(0.0, static_cast<double>(drs[i].rT3 - drs[i].up1));
        drs[i].violT6 = std::max(0.0, static_cast<double>(drs[i].rT6 - drs[i].up1));
        if (atoms)
        {
            int j1 = entry.atoms[0];
            int j2 = entry.atoms[1];
            atoms->pdbinfo[j1].bfac += drs[i].violT3 * 5;
            atoms->pdbinfo[j2].bfac += drs[i].violT3 * 5;
        }
    }
    dump_viol(log, dd.nres, drs, FALSE);

    fprintf(log, "+++ Sorted by linear averaged violations: +++\n");
    std::sort(drs, drs + dd.nres,
              [](const t_dr_stats& a, const t_dr_stats& b) { return a.viol > b.viol; }); // Reverse sort
    dump_viol(log, dd.nres, drs, TRUE);

    dump_dump(log, dd.nres, drs);

    sfree(drs);
}

static void dump_clust_stats(FILE*                          fp,
                             const t_disresdata&            dd,
                             const InteractionList&         disres,
                             gmx::ArrayRef<const t_iparams> ip,
                             t_blocka*                      clust,
                             t_dr_result                    dr[],
                             char*                          clust_name[],
                             int                            isize,
                             int                            index[])
{
    int         k, mmm = 0;
    double      sumV, maxV, sumVT3, sumVT6, maxVT3, maxVT6;
    t_dr_stats* drs;

    fprintf(fp, "\n");
    fprintf(fp, "++++++++++++++ STATISTICS ++++++++++++++++++++++\n");
    fprintf(fp, "Cluster  NFrames    SumV      MaxV     SumVT     MaxVT     SumVS     MaxVS\n");

    snew(drs, dd.nres);

    for (k = 0; (k < clust->nr); k++)
    {
        if (dr[k].nframes == 0)
        {
            continue;
        }
        if (dr[k].nframes != (clust->index[k + 1] - clust->index[k]))
        {
            gmx_fatal(FARGS,
                      "Inconsistency in cluster %s.\n"
                      "Found %d frames in trajectory rather than the expected %d\n",
                      clust_name[k], dr[k].nframes, clust->index[k + 1] - clust->index[k]);
        }
        if (!clust_name[k])
        {
            gmx_fatal(FARGS, "Inconsistency with cluster %d. Invalid name", k);
        }
        sumV = sumVT3 = sumVT6 = maxV = maxVT3 = maxVT6 = 0;

        // Use a map to process each restraint only once while looping over all pairs
        std::unordered_map<int, bool> restraintHasBeenProcessed;
        for (const auto entry : disres)
        {
            // Note that the restraint i can be used by multiple pairs
            const int i = entry.parameterType - dd.type_min;

            if (restraintHasBeenProcessed[i])
            {
                continue;
            }

            drs[i].label = ip[entry.parameterType].disres.label;
            drs[i].bCore = is_core(drs[i].label, isize, index);
            drs[i].up1   = ip[entry.parameterType].disres.up1;
            drs[i].r     = dr[k].aver1[i] / dr[k].nframes;
            if ((dr[k].aver_3[i] <= 0) || !std::isfinite(dr[k].aver_3[i]))
            {
                gmx_fatal(FARGS, "dr[%d].aver_3[%d] = %f", k, i, dr[k].aver_3[i]);
            }
            drs[i].rT3    = gmx::invcbrt(dr[k].aver_3[i] / dr[k].nframes);
            drs[i].rT6    = gmx::invsixthroot(dr[k].aver_6[i] / dr[k].nframes);
            drs[i].viol   = std::max(0.0, static_cast<double>(drs[i].r - drs[i].up1));
            drs[i].violT3 = std::max(0.0, static_cast<double>(drs[i].rT3 - drs[i].up1));
            drs[i].violT6 = std::max(0.0, static_cast<double>(drs[i].rT6 - drs[i].up1));
            sumV += drs[i].viol;
            sumVT3 += drs[i].violT3;
            sumVT6 += drs[i].violT6;
            maxV   = std::max(maxV, static_cast<double>(drs[i].viol));
            maxVT3 = std::max(maxVT3, static_cast<double>(drs[i].violT3));
            maxVT6 = std::max(maxVT6, static_cast<double>(drs[i].violT6));

            // We have processed restraint i, mark it as such
            restraintHasBeenProcessed[i] = true;
        }
        if (std::strcmp(clust_name[k], "1000") == 0)
        {
            mmm++;
        }
        fprintf(fp, "%-10s%6d%8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f\n", clust_name[k],
                dr[k].nframes, sumV, maxV, sumVT3, maxVT3, sumVT6, maxVT6);
    }
    fflush(fp);
    sfree(drs);
}

static void init_dr_res(t_dr_result* dr, int ndr)
{
    snew(dr->aver1, ndr + 1);
    snew(dr->aver2, ndr + 1);
    snew(dr->aver_3, ndr + 1);
    snew(dr->aver_6, ndr + 1);
    dr->nv      = 0;
    dr->nframes = 0;
    dr->sumv    = 0;
    dr->maxv    = 0;
    dr->averv   = 0;
}

static void dump_disre_matrix(const char*                   fn,
                              t_dr_result*                  dr,
                              int                           ndr,
                              int                           nsteps,
                              const InteractionDefinitions& idef,
                              const gmx_mtop_t*             mtop,
                              real                          max_dr,
                              int                           nlevels,
                              gmx_bool                      bThird)
{
    FILE*  fp;
    int*   resnr;
    int    n_res, a_offset, mol, a;
    int    i, j, nra, tp, ri, rj, index, nlabel, label;
    int    ai, aj, *ptr;
    real **matrix, *t_res, hi, *w_dr, rav, rviol;
    t_rgb  rlo = { 1, 1, 1 };
    t_rgb  rhi = { 0, 0, 0 };
    if (fn == nullptr)
    {
        return;
    }

    snew(resnr, mtop->natoms);
    n_res    = 0;
    a_offset = 0;
    for (const gmx_molblock_t& molb : mtop->molblock)
    {
        const t_atoms& atoms = mtop->moltype[molb.type].atoms;
        for (mol = 0; mol < molb.nmol; mol++)
        {
            for (a = 0; a < atoms.nr; a++)
            {
                resnr[a_offset + a] = n_res + atoms.atom[a].resind;
            }
            n_res += atoms.nres;
            a_offset += atoms.nr;
        }
    }

    snew(t_res, n_res);
    for (i = 0; (i < n_res); i++)
    {
        t_res[i] = i + 1;
    }
    snew(matrix, n_res);
    for (i = 0; (i < n_res); i++)
    {
        snew(matrix[i], n_res);
    }
    nra = idef.il[F_DISRES].numInteractions();
    snew(ptr, nra + 1);
    index  = 0;
    nlabel = 0;
    ptr[0] = 0;
    snew(w_dr, ndr);
    int interactionIndex = 0;
    for (const auto entry : idef.il[F_DISRES])
    {
        tp    = entry.parameterType;
        label = idef.iparams[tp].disres.label;

        if (label != index)
        {
            /* Set index pointer */
            ptr[index + 1] = interactionIndex;
            if (nlabel <= 0)
            {
                gmx_fatal(FARGS, "nlabel is %d, label = %d", nlabel, label);
            }
            if (index >= ndr)
            {
                gmx_fatal(FARGS, "ndr = %d, index = %d", ndr, index);
            }
            /* Update the weight */
            w_dr[index] = 1.0 / nlabel;
            index       = label;
            nlabel      = 1;
        }
        else
        {
            nlabel++;
        }
        interactionIndex++;
    }
    printf("nlabel = %d, index = %d, ndr = %d\n", nlabel, index, ndr);
    hi = 0;
    for (i = 0; (i < ndr); i++)
    {
        for (j = ptr[i]; (j < ptr[i + 1]); j++)
        {
            tp = idef.il[F_DISRES][j].parameterType;
            ai = idef.il[F_DISRES][j].atoms[0];
            aj = idef.il[F_DISRES][j].atoms[1];

            ri = resnr[ai];
            rj = resnr[aj];
            if (bThird)
            {
                rav = gmx::invcbrt(dr->aver_3[i] / nsteps);
            }
            else
            {
                rav = dr->aver1[i] / nsteps;
            }
            if (debug)
            {
                fprintf(debug, "DR %d, atoms %d, %d, distance %g\n", i, ai, aj, rav);
            }
            rviol = std::max(0.0_real, rav - idef.iparams[tp].disres.up1);
            matrix[ri][rj] += w_dr[i] * rviol;
            matrix[rj][ri] += w_dr[i] * rviol;
            hi = std::max(hi, matrix[ri][rj]);
            hi = std::max(hi, matrix[rj][ri]);
        }
    }

    sfree(resnr);

    if (max_dr > 0)
    {
        if (hi > max_dr)
        {
            printf("Warning: the maxdr that you have specified (%g) is smaller than\nthe largest "
                   "value in your simulation (%g)\n",
                   max_dr, hi);
        }
        hi = max_dr;
    }
    printf("Highest level in the matrix will be %g\n", hi);
    fp = gmx_ffopen(fn, "w");
    write_xpm(fp, 0, "Distance Violations", "<V> (nm)", "Residue", "Residue", n_res, n_res, t_res,
              t_res, matrix, 0, hi, rlo, rhi, &nlevels);
    gmx_ffclose(fp);
}

int gmx_disre(int argc, char* argv[])
{
    const char* desc[] = {
        "[THISMODULE] computes violations of distance restraints.",
        "The program always",
        "computes the instantaneous violations rather than time-averaged,",
        "because this analysis is done from a trajectory file afterwards",
        "it does not make sense to use time averaging. However,",
        "the time averaged values per restraint are given in the log file.[PAR]",
        "An index file may be used to select specific restraints by index group label for",
        "printing.[PAR]",
        "When the optional [TT]-q[tt] flag is given a [REF].pdb[ref] file coloured by the",
        "amount of average violations.[PAR]",
        "When the [TT]-c[tt] option is given, an index file will be read",
        "containing the frames in your trajectory corresponding to the clusters",
        "(defined in another manner) that you want to analyze. For these clusters",
        "the program will compute average violations using the third power",
        "averaging algorithm and print them in the log file."
    };
    static int      ntop    = 0;
    static int      nlevels = 20;
    static real     max_dr  = 0;
    static gmx_bool bThird  = TRUE;
    t_pargs         pa[]    = {
        { "-ntop",
          FALSE,
          etINT,
          { &ntop },
          "Number of large violations that are stored in the log file every step" },
        { "-maxdr",
          FALSE,
          etREAL,
          { &max_dr },
          "Maximum distance violation in matrix output. If less than or equal to 0 the "
          "maximum will be determined by the data." },
        { "-nlevels", FALSE, etINT, { &nlevels }, "Number of levels in the matrix output" },
        { "-third",
          FALSE,
          etBOOL,
          { &bThird },
          "Use inverse third power averaging or linear for matrix output" }
    };

    FILE *       out = nullptr, *aver = nullptr, *numv = nullptr, *maxxv = nullptr, *xvg = nullptr;
    int          i, j, kkk;
    t_trxstatus* status;
    real         t;
    rvec *       x, *xav = nullptr;
    rvec4*       f;
    matrix       box;
    gmx_bool     bPDB;
    int          isize;
    int *        index = nullptr, *ind_fit = nullptr;
    char*        grpname;
    t_cluster_ndx*    clust = nullptr;
    t_dr_result       dr, *dr_clust = nullptr;
    char**            leg;
    real *            vvindex = nullptr, *w_rls = nullptr;
    t_pbc             pbc, *pbc_null;
    int               my_clust;
    FILE*             fplog;
    gmx_output_env_t* oenv;
    gmx_rmpbc_t       gpbc = nullptr;

    t_filenm fnm[] = { { efTPR, nullptr, nullptr, ffREAD }, { efTRX, "-f", nullptr, ffREAD },
                       { efXVG, "-ds", "drsum", ffWRITE },  { efXVG, "-da", "draver", ffWRITE },
                       { efXVG, "-dn", "drnum", ffWRITE },  { efXVG, "-dm", "drmax", ffWRITE },
                       { efXVG, "-dr", "restr", ffWRITE },  { efLOG, "-l", "disres", ffWRITE },
                       { efNDX, nullptr, "viol", ffOPTRD }, { efPDB, "-q", "viol", ffOPTWR },
                       { efNDX, "-c", "clust", ffOPTRD },   { efXPM, "-x", "matrix", ffOPTWR } };
#define NFILE asize(fnm)

    if (!parse_common_args(&argc, argv, PCA_CAN_TIME | PCA_CAN_VIEW, NFILE, fnm, asize(pa), pa,
                           asize(desc), desc, 0, nullptr, &oenv))
    {
        return 0;
    }

    fplog = ftp2FILE(efLOG, NFILE, fnm, "w");

    if (ntop)
    {
        init5(ntop);
    }

    t_inputrec  irInstance;
    t_inputrec* ir = &irInstance;

    gmx::TopologyInformation topInfo;
    topInfo.fillFromInputFile(ftp2fn(efTPR, NFILE, fnm));
    int          ntopatoms = topInfo.mtop()->natoms;
    AtomsDataPtr atoms;
    bPDB = opt2bSet("-q", NFILE, fnm);
    if (bPDB)
    {
        snew(xav, ntopatoms);
        snew(ind_fit, ntopatoms);
        snew(w_rls, ntopatoms);
        for (kkk = 0; (kkk < ntopatoms); kkk++)
        {
            w_rls[kkk]   = 1;
            ind_fit[kkk] = kkk;
        }

        atoms = topInfo.copyAtoms();

        if (atoms->pdbinfo == nullptr)
        {
            snew(atoms->pdbinfo, atoms->nr);
        }
        atoms->havePdbInfo = TRUE;
    }

    gmx_localtop_t top(topInfo.mtop()->ffparams);
    gmx_mtop_generate_local_top(*topInfo.mtop(), &top, ir->efep != efepNO);
    const InteractionDefinitions& idef = top.idef;

    pbc_null = nullptr;
    if (ir->pbcType != PbcType::No)
    {
        pbc_null = &pbc;
    }

    if (ftp2bSet(efNDX, NFILE, fnm))
    {
        /* TODO: Nothing is written to this file if -c is provided, but it is
         * still opened... */
        rd_index(ftp2fn(efNDX, NFILE, fnm), 1, &isize, &index, &grpname);
        xvg = xvgropen(opt2fn("-dr", NFILE, fnm), "Individual Restraints", "Time (ps)", "nm", oenv);
        snew(vvindex, isize);
        snew(leg, isize);
        for (i = 0; (i < isize); i++)
        {
            index[i]++;
            snew(leg[i], 12);
            sprintf(leg[i], "index %d", index[i]);
        }
        xvgr_legend(xvg, isize, leg, oenv);
    }
    else
    {
        isize = 0;
    }

    ir->dr_tau = 0.0;
    t_disresdata disresdata;
    init_disres(fplog, topInfo.mtop(), ir, DisResRunMode::AnalysisTool, DDRole::Master,
                NumRanks::Single, MPI_COMM_NULL, nullptr, &disresdata, nullptr, FALSE);

    int natoms = read_first_x(oenv, &status, ftp2fn(efTRX, NFILE, fnm), &t, &x, box);
    snew(f, 5 * natoms);

    init_dr_res(&dr, disresdata.nres);
    if (opt2bSet("-c", NFILE, fnm))
    {
        clust = cluster_index(fplog, opt2fn("-c", NFILE, fnm));
        snew(dr_clust, clust->clust->nr + 1);
        for (i = 0; (i <= clust->clust->nr); i++)
        {
            init_dr_res(&dr_clust[i], disresdata.nres);
        }
    }
    else
    {
        out   = xvgropen(opt2fn("-ds", NFILE, fnm), "Sum of Violations", "Time (ps)", "nm", oenv);
        aver  = xvgropen(opt2fn("-da", NFILE, fnm), "Average Violation", "Time (ps)", "nm", oenv);
        numv  = xvgropen(opt2fn("-dn", NFILE, fnm), "# Violations", "Time (ps)", "#", oenv);
        maxxv = xvgropen(opt2fn("-dm", NFILE, fnm), "Largest Violation", "Time (ps)", "nm", oenv);
    }

    auto mdAtoms = gmx::makeMDAtoms(fplog, *topInfo.mtop(), *ir, false);
    atoms2md(topInfo.mtop(), ir, -1, nullptr, ntopatoms, mdAtoms.get());
    update_mdatoms(mdAtoms->mdatoms(), ir->fepvals->init_lambda);
    if (ir->pbcType != PbcType::No)
    {
        gpbc = gmx_rmpbc_init(idef, ir->pbcType, natoms);
    }

    j = 0;
    do
    {
        if (ir->pbcType != PbcType::No)
        {
            if (ir->bPeriodicMols)
            {
                set_pbc(&pbc, ir->pbcType, box);
            }
            else
            {
                gmx_rmpbc(gpbc, natoms, box, x);
            }
        }

        if (clust)
        {
            if (j > clust->maxframe)
            {
                gmx_fatal(FARGS,
                          "There are more frames in the trajectory than in the cluster index file. "
                          "t = %8f\n",
                          t);
            }
            my_clust = clust->inv_clust[j];
            range_check(my_clust, 0, clust->clust->nr);
            check_viol(fplog, idef.il[F_DISRES], idef.iparams, x, f, pbc_null, dr_clust, my_clust,
                       isize, index, vvindex, &disresdata);
        }
        else
        {
            check_viol(fplog, idef.il[F_DISRES], idef.iparams, x, f, pbc_null, &dr, 0, isize, index,
                       vvindex, &disresdata);
        }
        if (bPDB)
        {
            reset_x(atoms->nr, ind_fit, atoms->nr, nullptr, x, w_rls);
            do_fit(atoms->nr, w_rls, x, x);
            if (j == 0)
            {
                /* Store the first frame of the trajectory as 'characteristic'
                 * for colouring with violations.
                 */
                for (kkk = 0; (kkk < atoms->nr); kkk++)
                {
                    copy_rvec(x[kkk], xav[kkk]);
                }
            }
        }
        if (!clust)
        {
            if (isize > 0)
            {
                fprintf(xvg, "%10g", t);
                for (i = 0; (i < isize); i++)
                {
                    fprintf(xvg, "  %10g", vvindex[i]);
                }
                fprintf(xvg, "\n");
            }
            fprintf(out, "%10g  %10g\n", t, dr.sumv);
            fprintf(aver, "%10g  %10g\n", t, dr.averv);
            fprintf(maxxv, "%10g  %10g\n", t, dr.maxv);
            fprintf(numv, "%10g  %10d\n", t, dr.nv);
        }
        j++;
    } while (read_next_x(oenv, status, &t, x, box));
    close_trx(status);
    if (ir->pbcType != PbcType::No)
    {
        gmx_rmpbc_done(gpbc);
    }

    if (clust)
    {
        dump_clust_stats(fplog, disresdata, idef.il[F_DISRES], idef.iparams, clust->clust, dr_clust,
                         clust->grpname, isize, index);
    }
    else
    {
        dump_stats(fplog, j, disresdata, idef.il[F_DISRES], idef.iparams, &dr, isize, index,
                   bPDB ? atoms.get() : nullptr);
        if (bPDB)
        {
            write_sto_conf(opt2fn("-q", NFILE, fnm), "Coloured by average violation in Angstrom",
                           atoms.get(), xav, nullptr, ir->pbcType, box);
        }
        dump_disre_matrix(opt2fn_null("-x", NFILE, fnm), &dr, disresdata.nres, j, idef,
                          topInfo.mtop(), max_dr, nlevels, bThird);
        xvgrclose(out);
        xvgrclose(aver);
        xvgrclose(numv);
        xvgrclose(maxxv);
        do_view(oenv, opt2fn("-dn", NFILE, fnm), "-nxy");
        do_view(oenv, opt2fn("-da", NFILE, fnm), "-nxy");
        do_view(oenv, opt2fn("-ds", NFILE, fnm), "-nxy");
        do_view(oenv, opt2fn("-dm", NFILE, fnm), "-nxy");
    }
    if (isize > 0)
    {
        xvgrclose(xvg);
        if (!clust)
        {
            do_view(oenv, opt2fn("-dr", NFILE, fnm), "-nxy");
        }
    }

    gmx_ffclose(fplog);

    return 0;
}
