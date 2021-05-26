/*
 * This file is part of the GROMACS molecular simulation package.
 *
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
 * \brief Common functions for the different SITS GPU implementations.
 *
 * \author Junhan Chang <changjh@pku.edu.cn>
 *
 * \ingroup module_sits
 */

#include "gmxpre.h"

#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/sits/sits.h"
#include "gromacs/sits/sits_gpu_data_mgmt.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/logger.h"

struct t_sits;

namespace Sits
{

/*! \brief Resources that can be used to execute non-bonded kernels on */
enum class SitsResource : int
{
    Cpu,
    Gpu
};

bool Open_File_Safely(FILE** file, const char* file_name, const char* open_type)
{
    file[0] = NULL;
    file[0] = fopen(file_name, open_type);
    if (file[0] == NULL)
    {
        printf("Open file %s failed.\n", file_name);
        getchar();
        return false;
    }
    else
    {
        return true;
    }
}

/* Initializes an sits_atomdata_t data structure */
void sits_atomdata_init(
                        // const gmx::MDLogger&    mdlog,
                        sits_atomdata_t*        sits_at,
                        int                     n_energygroups,
                        t_sits*                 sitsvals)
{
    // sits_atomdata_params_init(mdlog, &nbat->paramsDeprecated(), kernelType, enbnxninitcombrule,
    //                            ntype, nbfp, n_energygroups);

    sits_at->sits_cal_mode = sitsvals->sits_cal_mode;
    sits_at->sits_enh_mode = sitsvals->sits_enh_mode;
    sits_at->sits_enh_bias = sitsvals->sits_enh_bias;
    sits_at->pw_enh_factor = sitsvals->pw_enh_factor;

    sits_at->k_numbers = sitsvals->k_numbers;

    sits_at->beta_k.resize(sits_at->k_numbers);
    sits_at->wt_beta_k.resize(sits_at->k_numbers);
    sits_at->sum_beta_factor = 0.0;
    sits_at->ene_recorded = 0.0;
	sits_at->gf.resize(sits_at->k_numbers);
	sits_at->gfsum = 0.0;
	sits_at->log_weight.resize(sits_at->k_numbers);
	sits_at->log_mk_inv.resize(sits_at->k_numbers);
	sits_at->log_norm_old.resize(sits_at->k_numbers);
	sits_at->log_norm.resize(sits_at->k_numbers);
	sits_at->log_pk.resize(sits_at->k_numbers);
	sits_at->log_nk.resize(sits_at->k_numbers);

    for (int i = 0; i< sits_at->k_numbers; i++)
    {
        sits_at->beta_k[i] = sitsvals->beta_k[i];
        sits_at->log_nk[i] = sitsvals->log_nk[i];
        sits_at->log_norm[i] = sitsvals->log_norm[i];
        sits_at->log_norm_old[i] = sitsvals->log_norm_old[i];
    }

    sits_at->output_interval = sitsvals->nst_sits_enerd_out;
    Open_File_Safely(&(sits_at->sits_enerd_out), sitsvals->sits_enerd_out, "wb");

    sits_at->beta0         = sitsvals->beta0;
    sits_at->constant_nk   = sitsvals->constant_nk;   // sits是否迭代更新nk

    if (!sits_at->constant_nk)
    {
        Open_File_Safely(&(sits_at->nk_traj_file), sitsvals->nk_traj_file, "wb"); //记录nk变化的文件
        Open_File_Safely(&(sits_at->norm_traj_file), sitsvals->norm_traj_file, "wb");      //记录log_norm变化的文件
        Open_File_Safely(&(sits_at->pk_traj_file), sitsvals->pk_traj_file, "wb");      //记录log_norm变化的文件
        sits_at->record_interval = sitsvals->nstsitsrecord;
        sits_at->update_interval = sitsvals->nstsitsupdate;
        sits_at->niter           = sitsvals->niter;
    }
    sits_at->nk_rest_file  = sitsvals->nk_rest_file;   //记录最后一帧nk的文件
    sits_at->norm_rest_file= sitsvals->norm_rest_file; //记录最后一帧log_norm的文件

    //计算时，可以对fc_ball直接修正，+ fb_shift进行调节，
    sits_at->fb_shift      = sitsvals->fb_shift;
    //也可以对进行修正，使加强计算能量时值为 energy_multiple * 原始能量 + energy_shift;
    sits_at->energy_multiple = sitsvals->energy_multiple;
    sits_at->energy_shift  = sitsvals->energy_shift;

    sits_at->natoms = 0;
}

std::unique_ptr<sits_t> init_sits(
                                //   const gmx::MDLogger&     mdlog,
                                  gmx_bool                 bFEP_SITS,
                                  const t_inputrec*        ir,
                                  const t_forcerec*        fr,
                                  const t_commrec*         cr,
                                  const gmx_hw_info_t&     hardwareInfo,
                                  const gmx_device_info_t* deviceInfo,
                                  const gmx_mtop_t*        mtop,
                                  gmx_wallcycle*           wcycle)
{
    const bool useGpu     = deviceInfo != nullptr;

    SitsResource sitsResource;
    if (useGpu)
    {
        sitsResource = SitsResource::Gpu;
    }
    else
    {
        sitsResource = SitsResource::Cpu;
    }

    // Nbnxm::KernelSetup kernelSetup = pick_nbnxn_kernel(mdlog, fr->use_simd_kernels, hardwareInfo,
    //                                                    nonbondedResource, ir, fr->bNonbonded);

    const bool haveMultipleDomains = (DOMAINDECOMP(cr) && cr->dd->nnodes > 1);

    // auto pinPolicy = (useGpu ? gmx::PinningPolicy::PinnedIfSupported : gmx::PinningPolicy::CannotBePinned);

    auto sits_at = std::make_unique<sits_atomdata_t>();

    int mimimumNumEnergyGroupNonbonded = ir->opts.ngener;
    if (ir->opts.ngener - ir->nwall == 1)
    {
        /* We have only one non-wall energy group, we do not need energy group
         * support in the non-bondeds kernels, since all non-bonded energy
         * contributions go to the first element of the energy group matrix.
         */
        mimimumNumEnergyGroupNonbonded = 1;
    }
    sits_atomdata_init(sits_at.get(), mimimumNumEnergyGroupNonbonded, ir->sitsvals);

    gmx_sits_cuda_t* gpu_sits = nullptr;
    if (useGpu)
    {
        /* init the NxN GPU data; the last argument tells whether we'll have
         * both local and non-local NB calculation on GPU */
        gpu_sits = gpu_init_sits(deviceInfo, sits_at.get(), 0);
    }

    return std::make_unique<sits_t>(std::move(sits_at), gpu_sits);
}

} // namespace Sits

sits_atomdata_t::sits_atomdata_t() :
    natoms(0),
    k_numbers(0)
{
}

sits_t::sits_t(std::unique_ptr<sits_atomdata_t>  sits_at_in,
                gmx_sits_cuda_t*                  gpu_sits_ptr) :
    sits_at(std::move(sits_at_in)),
    gpu_sits(gpu_sits_ptr)
{
    GMX_RELEASE_ASSERT(sits_at, "Need valid atomdata object");
}

sits_t::~sits_t()
{
    Sits::gpu_free(gpu_sits);
}

void sits_t::sits_atomdata_set_energygroups(std::vector<int> cginfo)
{
    sits_at->natoms = cginfo.size();
    sits_at->energrp.resize(sits_at->natoms);
    for (int i = 0; i < sits_at->natoms; i++)
    {
        sits_at->energrp[i] = cginfo[i]&255;
    }
}

void sits_t::print_sitsvals(bool bFirstTime, int step)
{
    if (bFirstTime)
    {
        printf("\n############# SITS ############\n");
        printf("natoms = %d\n", sits_at->natoms);
        printf("k_num = %d,\nbeta_k = ", sits_at->k_numbers);
        for (int i=0; i<sits_at->k_numbers; i++)
        {
            printf("%.3f ", sits_at->beta_k[i]);
        }
        printf("\n############# SITS ############\n");
    }
    else
    {
        if (step % sits_at->output_interval == 0)
        {
            if (gpu_sits)
            {
                Sits::gpu_sitsvals_cpyback(gpu_sits, sits_at.get());
            }
            float weight = exp(-sits_at->beta0 * sits_at->ene_recorded - sits_at->gfsum);
            if (sits_at->sits_enerd_out)
            {
                fprintf(sits_at->sits_enerd_out, "%14.4f %14.4f %14.4f %14.4f %14.4f %14e %7.5f\n", 
                        sits_at->enerd[0], sits_at->enerd[1], sits_at->enerd[2], sits_at->ene_recorded, 
                        -sits_at->gfsum / sits_at->beta0, weight, sits_at->factor[0]);
            }
            else
            {
                printf("\n______AA______ ______AB______ ______BB______ _____E_enh____ _____E_eff____ ___reweight___   factor\n");
                printf("%14.4f %14.4f %14.4f %14.4f %14.4f %14e %7.5f\n",
                        sits_at->enerd[0], sits_at->enerd[1], sits_at->enerd[2], sits_at->ene_recorded, 
                        -sits_at->gfsum / sits_at->beta0, weight, sits_at->factor[0]);
            }
        }
    }
}