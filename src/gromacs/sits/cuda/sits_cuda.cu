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
 *  \brief Define CUDA implementation of sits_gpu.h
 *
 *  \author Junhan Chang <changjh@pku.edu.cn>
 */
#include "gmxpre.h"

#include "config.h"

#include <assert.h>
#include <stdlib.h>

#include "gromacs/sits/sits_gpu.h"

#if defined(_MSVC)
#    include <limits>
#endif

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/gpueventsynchronizer.cuh"
#include "gromacs/gpu_utils/vectype_ops.cuh"
#include "gromacs/mdtypes/simulation_workload.h"

#include "gromacs/sits/sits_gpu_data_mgmt.h"
#include "gromacs/sits/sits.h"

#include "gromacs/timing/gpu_timing.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/gmxassert.h"

#include "gromacs/math/utilities.h"

#include "sits_cuda_types.h"

#define FLT_MAX 10e8;

/*! As we execute nonbonded workload in separate streams, before launching
   the kernel we need to make sure that he following operations have completed:
   - atomdata allocation and related H2D transfers (every nstlist step);
   - pair list H2D transfer (every nstlist step);
   - shift vector H2D transfer (every nstlist step);
   - force (+shift force and energy) output clearing (every step).

   These operations are issued in the local stream at the beginning of the step
   and therefore always complete before the local kernel launch. The non-local
   kernel is launched after the local on the same device/context hence it is
   inherently scheduled after the operations in the local stream (including the
   above "misc_ops") on pre-GK110 devices with single hardware queue, but on later
   devices with multiple hardware queues the dependency needs to be enforced.
   We use the misc_ops_and_local_H2D_done event to record the point where
   the local x+q H2D (and all preceding) tasks are complete and synchronize
   with this event in the non-local stream before launching the non-bonded kernel.
 */
// void gpu_enhance_force(gmx_sits_cuda_t* gpu_sits)
// {
//     cu_sits_atdat_t* atdat  = gpu_sits->sits_atdat;
//     cu_sits_param_t* param  = gpu_sits->sits_param;
//     cudaStream_t     stream = *(gpu_sits->stream);

//     /* Kernel launch config:
//      * - The thread block dimensions match the size of i-clusters, j-clusters,
//      *   and j-cluster concurrency, in x, y, and z, respectively.
//      * - The 1D block-grid contains as many blocks as super-clusters.
//      */
//     int num_threads_z = 1;
//     if (nb->dev_info->prop.major == 3 && nb->dev_info->prop.minor == 7)
//     {
//         num_threads_z = 2;
//     }
//     int nblock = calc_nb_kernel_nblock(plist->nsci, nb->dev_info);


//     KernelLaunchConfig config;
//     config.blockSize[0]     = c_clSize;
//     config.blockSize[1]     = c_clSize;
//     config.blockSize[2]     = num_threads_z;
//     config.gridSize[0]      = nblock;
//     config.sharedMemorySize = calc_shmem_required_nonbonded(num_threads_z, nb->dev_info, nbp);
//     config.stream           = stream;

//     if (debug)
//     {
//         fprintf(debug,
//                 "Non-bonded GPU launch configuration:\n\tThread block: %zux%zux%zu\n\t"
//                 "\tGrid: %zux%zu\n\t#Super-clusters/clusters: %d/%d (%d)\n"
//                 "\tShMem: %zu\n",
//                 config.blockSize[0], config.blockSize[1], config.blockSize[2], config.gridSize[0],
//                 config.gridSize[1], plist->nsci * c_numClPerSupercl, c_numClPerSupercl, plist->na_c,
//                 config.sharedMemorySize);
//     }

//     auto*      timingEvent = bDoTime ? t->interaction[iloc].nb_k.fetchNextEvent() : nullptr;
//     const auto kernel      = select_nbnxn_kernel(
//             nbp->eeltype, nbp->vdwtype, stepWork.computeEnergy,
//             (plist->haveFreshList && !nb->timers->interaction[iloc].didPrune), nb->dev_info);
//     const auto kernelArgs =
//             prepareGpuKernelArguments(kernel, config, adat, nbp, plist, &stepWork.computeVirial);
//     launchGpuKernel(kernel, config, timingEvent, "k_calc_nb", kernelArgs);

//     if (GMX_NATIVE_WINDOWS)
//     {
//         /* Windows: force flushing WDDM queue */
//         cudaStreamQuery(stream);
//     }
// }

static __device__ __host__ float log_add_log(float a, float b)
{
    return fmaxf(a, b) + logf(1.0 + expf(-fabsf(a - b)));
}

static __global__ void Sits_Record_Ene(float*       ene_record,
                                       const float* pw_ene,
                                       const float* pp_ene,
                                       const float  pe_a,
                                       const float  pe_b,
                                       const float  pw_factor)
{
    float temp = *pw_ene * pw_factor + *pp_ene;
    temp       = pe_a * temp + pe_b;

    *ene_record = temp;
    // printf("DEBUG ene_record: %f\n", ene_record[0]);
}

static __global__ void Sits_Update_gf(const int kn, float* gf, const float* ene_record, const float* log_nk, const float* beta_k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
    {
        gf[i] = -beta_k[i] * ene_record[0] + log_nk[i];
        // printf("DEBUG gf: %d %f\n", i, gf[i]);
    }
}

static __global__ void Sits_Update_gfsum(const int kn, float* gfsum, const float* gf)
{
    if (threadIdx.x == 0)
    {
        gfsum[0] = -FLT_MAX;
    }
    for (int i = 0; i < kn; i = i + 1)
    {
        gfsum[0] = log_add_log(gfsum[0], gf[i]);
        // printf("DEBUG gfsum: %d %f %f\n", i, gfsum[0], gf[i]);
    }
}

static __global__ void Sits_Update_log_pk(const int kn, float* log_pk, const float* gf, const float* gfsum, const int reset)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
    {
        if (reset == 1)
        {
            log_pk[i] = -FLT_MAX;
        }
        float gfi = gf[i];
        log_pk[i] = log_add_log(log_pk[i], gfi - gfsum[0]);
        // printf("DEBUG log_pk: %d %f %f\n", i, log_pk[i], gfsum[0]);
    }
}

static __global__ void Sits_Update_log_mk_inv(const int    kn,
                                              float*       log_weight,
                                              float*       log_mk_inv,
                                              float*       log_norm_old,
                                              float*       log_norm,
                                              const float* log_pk,
                                              const float* log_nk)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn - 1)
    {
        log_weight[i] = (log_pk[i] + log_pk[i + 1]) * 0.5;
        // printf("DEBUG log_weight: %d %f %f\n", i, log_pk[i], log_pk[i + 1]);
        log_mk_inv[i]   = log_nk[i] - log_nk[i + 1];
        log_norm_old[i] = log_norm[i];
        log_norm[i]     = log_add_log(log_norm[i], log_weight[i]);
        log_mk_inv[i] =
                log_add_log(log_mk_inv[i] + log_norm_old[i] - log_norm[i],
                            log_pk[i + 1] - log_pk[i] + log_mk_inv[i] + log_weight[i] - log_norm[i]);
        // printf("DEBUG log_norm: %d %f %f\n", i, log_norm[i], log_weight[i]);
    }
}

static __global__ void Sits_Update_log_nk_inv(const int kn, float* log_nk_inv, const float* log_mk_inv)
{
    for (int i = 0; i < kn - 1; i++)
    {
        log_nk_inv[i + 1] = log_nk_inv[i] + log_mk_inv[i];
    }
}

static __global__ void Sits_Update_nk(const int kn, float* log_nk, float* nk, const float* log_nk_inv)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
    {
        log_nk[i] = -log_nk_inv[i];
        nk[i]     = exp(log_nk[i]);
    }
}

__global__ void sits_enhance_force_Calculate_nkExpBetakU_1(const int    k_numbers,
                                                                const float* beta_k,
                                                                const float* nk,
                                                                float*       nkexpbetaku,
                                                                const float  ene)
{
    float lin = beta_k[k_numbers - 1];
    for (int i = threadIdx.x; i < k_numbers; i = i + blockDim.x)
    {
        nkexpbetaku[i] = nk[i] * expf(-(beta_k[i] - lin) * ene);
        // printf("%f %f\n", beta_k[i], nkexpbetaku[i]);
    }
}

__global__ void sits_enhance_force_Calculate_nkExpBetakU_2(const int    k_numbers,
                                                                const float* beta_k,
                                                                const float* nk,
                                                                float*       nkexpbetaku,
                                                                const float  ene)
{
    float lin = beta_k[0];
    for (int i = threadIdx.x; i < k_numbers; i = i + blockDim.x)
    {
        nkexpbetaku[i] = nk[i] * expf(-(beta_k[i] - lin) * ene);
        // printf("%f %f\n", beta_k[i], nkexpbetaku[i]);
    }
}

__global__ void sits_enhance_force_Sum_Of_Above(const int    k_numbers,
                                                     const float* nkexpbetaku,
                                                     const float* beta_k,
                                                     float*       sum_of_above)
{
    if (threadIdx.x == 0)
    {
        sum_of_above[0] = 0.;
    }
    __syncthreads();
    float lin = 0.;
    for (int i = threadIdx.x; i < k_numbers; i = i + blockDim.x)
    {
        lin = lin + beta_k[i] * nkexpbetaku[i];
    }
    atomicAdd(sum_of_above, lin);
}

__global__ void sits_enhance_force_Sum_Of_nkExpBetakU(const int    k_numbers,
                                                           const float* nkexpbetaku,
                                                           float*       sum_of_below)
{
    if (threadIdx.x == 0)
    {
        sum_of_below[0] = 0.;
    }
    __syncthreads();
    float lin = 0.;
    for (int i = threadIdx.x; i < k_numbers; i = i + blockDim.x)
    {
        lin = lin + nkexpbetaku[i];
        // printf("%f\n", nkexpbetaku[i]);
    }
    atomicAdd(sum_of_below, lin);
}

__global__ void sits_enhance_force_update_factor(float*        sum_a,
                                                float*        sum_b,
                                                float*        factor,
                                                const float   beta_0,
                                                const float   fb_bias)
{
    if (threadIdx.x == 0)
    {
        if (isinf(factor[0]) || isnan(factor[0]) || factor[0] == 0.0)
        {
            factor[0] = 1.0;
        }
        if (isinf(factor[1]) || isnan(factor[1]) || factor[1] == 0.0)
        {
            factor[1] = 1.0;
        }
        factor[0] = sum_a[0] / sum_b[0] / beta_0 + fb_bias;
        // avoid crashing caused by sharp fluctuation of fc_ball
        if (!isinf(factor[0]) && !isnan(factor[0]) && (factor[0] > 0.4 * factor[1])
            && (factor[0] < 2 * factor[1]))
        {
            factor[1] = factor[0];
        }
        else
        {
            factor[0] = factor[1];
        }
    }
    // printf("\n| sum_a | sum_b | factor | factor1 |\n");
    // printf(" %7.3f %7.3f %8.3f %8.3f \n", *sum_a, *sum_b, factor[0], factor[1]);
}

static __global__ void sits_enhance_force_Protein(const int     protein_numbers,
                                                  float3*       md_frc,
                                                  const float3* pw_frc,
                                                  const float   fc_ball,
                                                  const float   pw_factor)
{
    for (int i = threadIdx.x; i < protein_numbers; i = i + blockDim.x)
    {
        md_frc[i].x = fc_ball * (md_frc[i].x) + pw_factor * pw_frc[i].x;
        md_frc[i].y = fc_ball * (md_frc[i].y) + pw_factor * pw_frc[i].y;
        md_frc[i].z = fc_ball * (md_frc[i].z) + pw_factor * pw_frc[i].z;
    }
}

static __global__ void sits_enhance_force_Water(const int     protein_numbers,
                                                const int     natoms,
                                                float3*       md_frc,
                                                const float3* pw_frc,
                                                const float   pw_factor)
{
    for (int i = threadIdx.x + protein_numbers; i < natoms; i = i + blockDim.x)
    {
        md_frc[i].x = md_frc[i].x + pw_factor * pw_frc[i].x;
        md_frc[i].y = md_frc[i].y + pw_factor * pw_frc[i].y;
        md_frc[i].z = md_frc[i].z + pw_factor * pw_frc[i].z;
    }
}

static __global__ void sits_enhance_force_by_energrp(const int     natoms,
                                                     int*          energrp,
                                                     float3*       md_frc,
                                                     const float3* pw_frc,
                                                     float*        fc_ball,
                                                     const float   pw_factor)
{
    for (int i = threadIdx.x; i < natoms; i = i + blockDim.x)
    {
        float fc_1 = fc_ball[0] - 1.0;
        if (energrp[i] == 0)
        {
            md_frc[i] *= fc_1;
        }
        else
        {
            md_frc[i] = make_float3(0.0);
        }
        md_frc[i] += fc_1 * pw_factor * pw_frc[i];
    }
}

/*-------------------------------- End CUDA kernels-----------------------------*/

void Sits_Classical_Enhance_Force(const int     natoms,
                                              int*          energrp,
                                              const float   pw_factor,
                                              float3*       md_frc,
                                              const float3* pw_frc,
                                              const float*  pp_ene,
                                              const float*  pw_ene,
                                              const int     k_numbers,
                                              float*        nkexpbetaku,
                                              const float*  beta_k,
                                              const float*  n_k,
                                              float*        sum_a,
                                              float*        sum_b,
                                              float*        factor,
                                              const float   beta_0,
                                              const float   pe_a,
                                              const float   pe_b,
                                              const float   fb_bias)
{
    float* h_E_pp;
    float* h_E_pw;

    h_E_pp = (float *) malloc(sizeof(float));
    h_E_pw = (float *) malloc(sizeof(float));
    cudaMemcpy(h_E_pp, pp_ene, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E_pw, pw_ene, sizeof(float), cudaMemcpyDeviceToHost);
    float ene = *(h_E_pp) + pw_factor * *(h_E_pw);
    ene       = pe_a * ene + pe_b;
    if (ene > 0)
    {
        sits_enhance_force_Calculate_nkExpBetakU_1<<<1, 64>>>(k_numbers, beta_k, n_k,
                                                                   nkexpbetaku, ene);
    }
    else
    {
        sits_enhance_force_Calculate_nkExpBetakU_2<<<1, 64>>>(k_numbers, beta_k, n_k,
                                                                   nkexpbetaku, ene);
    }

    sits_enhance_force_Sum_Of_nkExpBetakU<<<1, 128>>>(k_numbers, nkexpbetaku, sum_b);

    sits_enhance_force_Sum_Of_Above<<<1, 128>>>(k_numbers, nkexpbetaku, beta_k, sum_a);

    sits_enhance_force_update_factor<<<1, 1>>>(sum_a, sum_b, factor, beta_0, fb_bias);
    //	printf("factor %e sum0 %e %e ene %f lfactor %e\n", fc, sum_a[0], sum_b[0], ene, factor[1]);

    // line
    // fc = (ene - 20.) / 80./2. + 0.2;
    sits_enhance_force_by_energrp<<<32, 128>>>(natoms, energrp, md_frc, pw_frc, factor, pw_factor);
}

namespace Sits
{

void gpu_update_params(gmx_sits_cuda_t* gpu_sits, int step, FILE* nklog, FILE* normlog)
{
    cu_sits_atdat_t* atdat = gpu_sits->sits_atdat;
    cu_sits_param_t* param = gpu_sits->sits_param;

    if (!param->constant_nk && step % param->record_interval == 0)
    {
        Sits_Record_Ene<<<1, 1>>>(param->ene_recorded, &(atdat->d_enerd[1]), &(atdat->d_enerd[0]),
                                  param->energy_multiple, param->energy_shift, atdat->pw_enh_factor);

        Sits_Update_gf<<<ceilf((float)param->k_numbers / 32.), 32>>>(
                param->k_numbers, param->gf, param->ene_recorded,
                param->log_nk, param->beta_k);

        Sits_Update_gfsum<<<1, 1>>>(param->k_numbers, param->gfsum, param->gf);

        Sits_Update_log_pk<<<ceilf((float)param->k_numbers / 32.), 32>>>(
                param->k_numbers, param->log_pk, param->gf,
                param->gfsum, param->reset);

        param->reset = 0;
        param->record_count++;

        if ((param->record_count % param->update_interval == 0) && (param->record_count / param->update_interval < param->niter))
        {
            Sits_Update_log_mk_inv<<<ceilf((float)param->k_numbers / 32.), 32>>>(
                    param->k_numbers, param->log_weight, param->log_mk_inv,
                    param->log_norm_old, param->log_norm, param->log_pk,
                    param->log_nk);

            Sits_Update_log_nk_inv<<<1, 1>>>(param->k_numbers, param->log_nk_inv,
                                             param->log_mk_inv);

            Sits_Update_nk<<<ceilf((float)param->k_numbers / 32.), 32>>>(
                    param->k_numbers, param->log_nk, param->nk,
                    param->log_nk_inv);

            // param->record_count = 0;
            param->reset        = 1;

            if (!param->constant_nk)
            {
                float* h_log_nk;
                h_log_nk = (float*) malloc(param->k_numbers * sizeof(float));
                cudaMemcpy(h_log_nk, param->log_nk, sizeof(float) * param->k_numbers, cudaMemcpyDeviceToHost);

                float* h_log_pk;
                h_log_pk = (float*) malloc(param->k_numbers * sizeof(float));
                cudaMemcpy(h_log_pk, param->log_pk, sizeof(float) * param->k_numbers, cudaMemcpyDeviceToHost);

                float* h_log_norm;
                h_log_norm = (float*) malloc(param->k_numbers * sizeof(float));
                cudaMemcpy(h_log_norm, param->log_norm, sizeof(float) * param->k_numbers, cudaMemcpyDeviceToHost);

                if (nklog)
                {
                    for (int i = 0; i < param->k_numbers; i++){
                        fprintf(nklog, "%8.4f ", h_log_nk[i]);
                    }
                    for (int i = 0; i < param->k_numbers; i++){
                        fprintf(nklog, "%8.4f ", h_log_pk[i]);
                    }
                    fprintf(nklog, "\n");
                }

                if (normlog)
                {
                    for (int i = 0; i < param->k_numbers; i++){
                        fprintf(normlog, "%8.4f ", h_log_norm[i]);
                    }
                    fprintf(nklog, "\n");
                }
                // cudaMemcpy(param->log_nk_recorded_cpu, param->nk,
                //            sizeof(float) * param->k_numbers, cudaMemcpyDeviceToHost);
                // fwrite(param->log_nk_recorded_cpu, sizeof(float), param->k_numbers,
                //        param->nk_traj_file);
                // cudaMemcpy(param->log_norm_recorded_cpu, param->log_norm,
                //            sizeof(float) * param->k_numbers, cudaMemcpyDeviceToHost);
                // fwrite(param->log_norm_recorded_cpu, sizeof(float),
                //        param->k_numbers, param->norm_traj_file);
            }
        }
    }
}

void gpu_enhance_force(gmx_sits_cuda_t* gpu_sits, int step)
{
    cu_sits_atdat_t* atdat = gpu_sits->sits_atdat;
    cu_sits_param_t* param = gpu_sits->sits_param;

    if (atdat->sits_cal_mode == 0)
    {
        Sits_Classical_Enhance_Force(
                atdat->natoms, atdat->energrp, atdat->pw_enh_factor, 
                atdat->d_force_tot_nbat, atdat->d_force_pw_nbat, 
                &(atdat->d_enerd[0]), &(atdat->d_enerd[1]),
                param->k_numbers, param->nkExpBetakU, param->beta_k,
                param->nk, param->sum_a, param->sum_b,
                param->factor, param->beta0, param->energy_multiple,
                param->energy_shift, param->fb_shift);
    }
    else if (atdat->sits_cal_mode == 1)
    {
        // Get fc_ball by random walk in given potential to reach certain marginal distribution
        // if (!simple_param->is_constant_fc_ball)
        // {
        //     fc_ball_random_walk();
        // }
        // else
        // {
        //     param->fc_ball = simple_param->constant_fc_ball;
        // }
        // sits_enhance_force_Protein<<<1, 128>>>(
        //         param->protein_natoms, frc, protein_water_frc, param->fc_ball,
        //         param->pwwp_enhance_factor * param->fc_ball + 1.0 - param->pwwp_enhance_factor);
        // sits_enhance_force_Water<<<1, 128>>>(
        //         param->protein_natoms, param->natoms, frc, protein_water_frc,
        //         param->pwwp_enhance_factor * param->fc_ball + 1.0 - param->pwwp_enhance_factor);
    }
    else
    {
        // sits_enhance_force_Protein<<<1, 128>>>(
        //         param->protein_natoms, frc, protein_water_frc, param->fc_ball,
        //         param->pwwp_enhance_factor * param->fc_ball + 1.0 - param->pwwp_enhance_factor);
        // sits_enhance_force_Water<<<1, 128>>>(
        //         param->protein_natoms, param->natoms, frc, protein_water_frc,
        //         param->pwwp_enhance_factor * param->fc_ball + 1.0 - param->pwwp_enhance_factor);
    }
}

// void sits_t::CLASSICAL_Sits_INFORMATION::Export_Restart_Information_To_File()
// {
//     FILE* nk;
//     Open_File_Safely(&nk, nk_rest_file, "w");
//     cudaMemcpy(log_nk_recorded_cpu, nk, sizeof(float) * k_numbers, cudaMemcpyDeviceToHost);
//     for (int i = 0; i < k_numbers; i++)
//     {
//         fprintf(nk, "%f\n", log_nk_recorded_cpu[i]);
//     }
//     fclose(nk);

//     FILE* norm;
//     Open_File_Safely(&norm, norm_rest_file, "w");
//     cudaMemcpy(log_norm_recorded_cpu, log_norm, sizeof(float) * k_numbers, cudaMemcpyDeviceToHost);
//     for (int i = 0; i < k_numbers; i++)
//     {
//         fprintf(norm, "%f\n", log_norm_recorded_cpu[i]);
//     }
//     fclose(norm);
// }

} // namespace Sits