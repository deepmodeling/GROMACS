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

#include "gmxpre.h"

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/enerdata_utils.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/sits/sits_gpu_data_mgmt.h"
#include "gromacs/sits/sits.h"
#include "gromacs/simd/simd.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"

void sits_t::sits_enhance_force(int step)
{
    if (gpu_sits)
    {
        Sits::gpu_enhance_force(gpu_sits, step);
    }
    else
    {
        // nothing
        GMX_RELEASE_ASSERT(false, "Invalid sits enhance_force kernel type passed!");
    }
}

void sits_t::sits_update_params(int step)
{
    if (gpu_sits)
    {
        Sits::gpu_update_params(gpu_sits, step, sits_at->nk_traj_file, sits_at->norm_traj_file, sits_at->pk_traj_file);
    }
    else
    {
        // nothing
        GMX_RELEASE_ASSERT(false, "Invalid sits update_params kernel type passed!");
    }
}

void sits_t::clear_sits_energy_force()
{
    Sits::sits_gpu_clear_outputs(gpu_sits, true);
}