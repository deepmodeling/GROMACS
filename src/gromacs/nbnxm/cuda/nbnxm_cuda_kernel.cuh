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

/*! \internal \file
 *  \brief
 *  CUDA non-bonded kernel used through preprocessor-based code generation
 *  of multiple kernel flavors, see nbnxn_cuda_kernels.cuh.
 *
 *  NOTE: No include fence as it is meant to be included multiple times.
 *
 *  \author Szilárd Páll <pall.szilard@gmail.com>
 *  \author Berk Hess <hess@kth.se>
 *  \ingroup module_nbnxm
 */

#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "gromacs/gpu_utils/cuda_kernel_utils.cuh"
#include "gromacs/math/utilities.h"
#include "gromacs/pbcutil/ishift.h"
/* Note that floating-point constants in CUDA code should be suffixed
 * with f (e.g. 0.5f), to stop the compiler producing intermediate
 * code that is in double precision.
 */

#if defined EL_EWALD_ANA || defined EL_EWALD_TAB
/* Note: convenience macro, needs to be undef-ed at the end of the file. */
#    define EL_EWALD_ANY
#endif

#if defined EL_EWALD_ANY || defined EL_RF || defined LJ_EWALD \
        || (defined EL_CUTOFF && defined CALC_ENERGIES)
/* Macro to control the calculation of exclusion forces in the kernel
 * We do that with Ewald (elec/vdw) and RF. Cut-off only has exclusion
 * energy terms.
 *
 * Note: convenience macro, needs to be undef-ed at the end of the file.
 */
#    define EXCLUSION_FORCES
#endif

#if defined LJ_EWALD_COMB_GEOM || defined LJ_EWALD_COMB_LB
/* Note: convenience macro, needs to be undef-ed at the end of the file. */
#    define LJ_EWALD
#endif

#if defined LJ_COMB_GEOM || defined LJ_COMB_LB
#    define LJ_COMB
#endif

/*
   Kernel launch parameters:
    - #blocks   = #pair lists, blockId = pair list Id
    - #threads  = NTHREAD_Z * c_clSize^2
    - shmem     = see nbnxn_cuda.cu:calc_shmem_required_nonbonded()

    Each thread calculates an i force-component taking one pair of i-j atoms.
 */

/**@{*/
/*! \brief Compute capability dependent definition of kernel launch configuration parameters.
 *
 * NTHREAD_Z controls the number of j-clusters processed concurrently on NTHREAD_Z
 * warp-pairs per block.
 *
 * - On CC 3.0-3.5, and >=5.0 NTHREAD_Z == 1, translating to 64 th/block with 16
 * blocks/multiproc, is the fastest even though this setup gives low occupancy
 * (except on 6.0).
 * NTHREAD_Z > 1 results in excessive register spilling unless the minimum blocks
 * per multiprocessor is reduced proportionally to get the original number of max
 * threads in flight (and slightly lower performance).
 * - On CC 3.7 there are enough registers to double the number of threads; using
 * NTHREADS_Z == 2 is fastest with 16 blocks (TODO: test with RF and other kernels
 * with low-register use).
 *
 * Note that the current kernel implementation only supports NTHREAD_Z > 1 with
 * shuffle-based reduction, hence CC >= 3.0.
 *
 *
 * NOTEs on Volta / CUDA 9 extensions:
 *
 * - While active thread masks are required for the warp collectives
 *   (we use any and shfl), the kernel is designed such that all conditions
 *   (other than the inner-most distance check) including loop trip counts
 *   are warp-synchronous. Therefore, we don't need ballot to compute the
 *   active masks as these are all full-warp masks.
 *
 * - TODO: reconsider the use of __syncwarp(): its only role is currently to prevent
 *   WAR hazard due to the cj preload; we should try to replace it with direct
 *   loads (which may be faster given the improved L1 on Volta).
 */

/* Kernel launch bounds for different compute capabilities. The value of NTHREAD_Z
 * determines the number of threads per block and it is chosen such that
 * 16 blocks/multiprocessor can be kept in flight.
 * - CC 3.0,3.5, and >=5.0: NTHREAD_Z=1, (64, 16) bounds
 * - CC 3.7:                NTHREAD_Z=2, (128, 16) bounds
 *
 * Note: convenience macros, need to be undef-ed at the end of the file.
 */
#if GMX_PTX_ARCH == 370
#    define NTHREAD_Z (2)
#    define MIN_BLOCKS_PER_MP (16)
#else
#    define NTHREAD_Z (1)
#    define MIN_BLOCKS_PER_MP (16)
#endif /* GMX_PTX_ARCH == 370 */
#define THREADS_PER_BLOCK (c_clSize * c_clSize * NTHREAD_Z)

#if GMX_PTX_ARCH >= 350
/**@}*/
__launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
#else
__launch_bounds__(THREADS_PER_BLOCK)
#endif /* GMX_PTX_ARCH >= 350 */
#ifdef PRUNE_NBL
#    ifdef CALC_ENERGIES
        __global__ void NB_KERNEL_FUNC_NAME(nbnxn_kernel, _VF_prune_cuda)
#    else
        __global__ void NB_KERNEL_FUNC_NAME(nbnxn_kernel, _F_prune_cuda)
#    endif /* CALC_ENERGIES */
#else
#    ifdef CALC_ENERGIES
        __global__ void NB_KERNEL_FUNC_NAME(nbnxn_kernel, _VF_cuda)
#    else
        __global__ void NB_KERNEL_FUNC_NAME(nbnxn_kernel, _F_cuda)
#    endif /* CALC_ENERGIES */
#endif     /* PRUNE_NBL */
                (const cu_atomdata_t atdat, const cu_nbparam_t nbparam, const cu_plist_t plist, bool bCalcFshift)
#ifdef FUNCTION_DECLARATION_ONLY
                        ; /* Only do function declaration, omit the function body. */
#else
{
    /* convenience variables */
    const nbnxn_sci_t* pl_sci = plist.sci;
    
    const bool bFEP = nbparam.bFEP;
    const float alpha_coul = nbparam.alpha_coul;
    const float alpha_vdw  = nbparam.alpha_vdw;
    float alpha_coul_eff   = alpha_coul;
    float alpha_vdw_eff    = alpha_vdw;
    const bool useSoftCore = (alpha_vdw != 0.0);
    const bool useScBetaNO = (alpha_vdw == 0.0);
    const float sigma6_def  = nbparam.sc_sigma6;
    const float sigma6_min = nbparam.sc_sigma6_min;
    const float lambda_q   = nbparam.lambda_q;
    const float _lambda_q  = 1 - lambda_q;
    const float lambda_v   = nbparam.lambda_v;
    const float _lambda_v  = 1 - lambda_v;

    const float lfac_coul[2] = {lambda_q, _lambda_q};
    const float lfac_vdw[2]  = {lambda_v, _lambda_v};
#ifndef PRUNE_NBL
    const
#endif
            nbnxn_cj4_t* pl_cj4      = plist.cj4;
    const nbnxn_excl_t*  excl        = plist.excl;
#ifndef LJ_COMB
    const int*           atom_types  = atdat.atom_types;
    const int*           atom_typesB = atdat.atom_typesB;
    int                  ntypes      = atdat.ntypes;
#else
    const float2* lj_comb = atdat.lj_comb;
    const float2* lj_combB= atdat.lj_combB;
    float2        ljcp_i, ljcp_j;
    float2        ljcpB_i, ljcpB_j;
#endif
    const float4*        xq          = atdat.xq;
    const float*         qB          = atdat.qB;
    float3*              f           = atdat.f;
    const float3*        shift_vec   = atdat.shift_vec;
    float                rcoulomb_sq = nbparam.rcoulomb_sq;
#    ifdef VDW_CUTOFF_CHECK
    float                rvdw_sq     = nbparam.rvdw_sq;
    float                vdw_in_range;
#    endif
#    ifdef LJ_EWALD
    float                lje_coeff2, lje_coeff6_6;
#    endif
#    ifdef EL_RF
    float                two_k_rf    = nbparam.two_k_rf;
#    endif
#    ifdef EL_EWALD_ANA
    float                beta2       = nbparam.ewald_beta * nbparam.ewald_beta;
    float                beta3       = nbparam.ewald_beta * nbparam.ewald_beta * nbparam.ewald_beta;
#    endif
#    ifdef PRUNE_NBL
    float                rlist_sq    = nbparam.rlistOuter_sq;
#    endif

#    ifdef CALC_ENERGIES
#        ifdef EL_EWALD_ANY
    float                beta        = nbparam.ewald_beta;
    float                ewald_shift = nbparam.sh_ewald;
#        else
    float c_rf = nbparam.c_rf;
#        endif /* EL_EWALD_ANY */
    float*               e_lj        = atdat.e_lj;
    float*               e_el        = atdat.e_el;
#    endif     /* CALC_ENERGIES */

    /* thread/block/warp id-s */
    unsigned int tidxi = threadIdx.x;
    unsigned int tidxj = threadIdx.y;
    unsigned int tidx  = threadIdx.y * blockDim.x + threadIdx.x;
#    if NTHREAD_Z == 1
    unsigned int tidxz = 0;
#    else
    unsigned int  tidxz = threadIdx.z;
#    endif
    unsigned int bidx  = blockIdx.x;
    unsigned int widx  = tidx / warp_size; /* warp index */

    int          sci, ci, cj, ai, aj, cij4_start, cij4_end;
#ifndef LJ_COMB
    int          typei, typej;
    int          typeBi, typeBj;
#endif
    int          i, jm, j4, wexcl_idx;
    float        qi, qj_f, r2, rpm2, rp, inv_r, inv_r2;
    float        qBi, qBj_f;
// #if !defined LJ_COMB_LB || defined CALC_ENERGIES
    float        inv_r6, c6, c12;
    float        c6B, c12B;
    float        rC[2], rV[2], rinvC[2], rinvV[2], r2invC[2], r2invV[2], rpinvC[2], rpinvV[2];
// #endif
#ifdef LJ_COMB_LB
    float        sigma, epsilon;
    float        sigmaB, epsilonB;
#endif
    float        sigma6[2];

    float        int_bit, F_invr;
#    ifdef CALC_ENERGIES
    float        E_lj, E_el;
#    endif
#    if defined CALC_ENERGIES || defined LJ_POT_SWITCH
    float        E_lj_p;
#    endif
    unsigned int wexcl, imask, mask_ji;
    float4       xqbuf;
    float        qBbuf;
    float3       xi, xj, rv, f_ij, fcj_buf;
    float3       fci_buf[c_numClPerSupercl]; /* i force buffer */
    nbnxn_sci_t  nb_sci;

    /*! i-cluster interaction mask for a super-cluster with all c_numClPerSupercl=8 bits set */
    const unsigned superClInteractionMask = ((1U << c_numClPerSupercl) - 1U);

    /*********************************************************************
     * Set up shared memory pointers.
     * sm_nextSlotPtr should always be updated to point to the "next slot",
     * that is past the last point where data has been stored.
     */
    extern __shared__ char sm_dynamicShmem[];
    char*                  sm_nextSlotPtr = sm_dynamicShmem;
    static_assert(sizeof(char) == 1,
                  "The shared memory offset calculation assumes that char is 1 byte");

    /* shmem buffer for i x+q pre-loading */
    float4* xqib = (float4*)sm_nextSlotPtr;
    sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*xqib));

    float* qBib = (float*)sm_nextSlotPtr;
    sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*qBib));

    /* shmem buffer for cj, for each warp separately */
    int* cjs = (int*)(sm_nextSlotPtr);
    /* the cjs buffer's use expects a base pointer offset for pairs of warps in the j-concurrent execution */
    cjs += tidxz * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize;
    sm_nextSlotPtr += (NTHREAD_Z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize * sizeof(*cjs));

#ifndef LJ_COMB
    /* shmem buffer for i atom-type pre-loading */
    int* atib = (int*)sm_nextSlotPtr;
    sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*atib));

    int* atBib = (int*)sm_nextSlotPtr;
    sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*atBib));
#else
    /* shmem buffer for i-atom LJ combination rule parameters */
    float2* ljcpib = (float2*)sm_nextSlotPtr;
    sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*ljcpib));

    float2* ljcpBib = (float2*)sm_nextSlotPtr;
    sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*ljcpBib));
#endif
    printf("xqib: %p, qBib: %p, cjs: %p, sm_nextSlotPtr: %p\n", xqib, qBib, cjs, sm_nextSlotPtr);
    /*********************************************************************/

    nb_sci     = pl_sci[bidx];         /* my i super-cluster's index = current bidx */
    sci        = nb_sci.sci;           /* super-cluster */
    cij4_start = nb_sci.cj4_ind_start; /* first ...*/
    cij4_end   = nb_sci.cj4_ind_end;   /* and last index of j clusters */

    if (tidxz == 0)
    {
        /* Pre-load i-atom x and q into shared memory */
        ci = sci * c_numClPerSupercl + tidxj;
        ai = ci * c_clSize + tidxi;
        printf("qBib: %.4f\n", qB[ai] * nbparam.epsfac);

        float* shiftptr = (float*)&shift_vec[nb_sci.shift];
        xqbuf = xq[ai] + make_float4(LDG(shiftptr), LDG(shiftptr + 1), LDG(shiftptr + 2), 0.0f);
        xqbuf.w *= nbparam.epsfac;
        xqib[tidxj * c_clSize + tidxi] = xqbuf;

        printf("xqib: %.8f, %.8f, %.8f, %.8f\n", xqbuf.x, xqbuf.y, xqbuf.z, xqbuf.w);

        qBib[tidxj * c_clSize + tidxi] = qB[ai] * nbparam.epsfac;

#ifndef LJ_COMB
        /* Pre-load the i-atom types into shared memory */
        printf("atib: %d\n", atom_types[ai]);
        printf("atBib: %d\n", atom_typesB[ai]);
        atib[tidxj * c_clSize + tidxi] = atom_types[ai];
        atBib[tidxj * c_clSize + tidxi] = atom_typesB[ai];
#else
        /* Pre-load the LJ combination parameters into shared memory */
        printf("ljcpib: %.4f, %.4f\n", lj_comb[ai].x, lj_comb[ai].y);
        printf("ljcpBib: %.4f, %.4f\n", lj_combB[ai].x, lj_combB[ai].y);
        ljcpib[tidxj * c_clSize + tidxi] = lj_comb[ai];
        ljcpBib[tidxj * c_clSize + tidxi] = lj_combB[ai];
#endif
    }
    __syncthreads();

    for (i = 0; i < c_numClPerSupercl; i++)
    {
        fci_buf[i] = make_float3(0.0f);
    }

#    ifdef LJ_EWALD
    /* TODO: we are trading registers with flops by keeping lje_coeff-s, try re-calculating it later */
    lje_coeff2   = nbparam.ewaldcoeff_lj * nbparam.ewaldcoeff_lj;
    lje_coeff6_6 = lje_coeff2 * lje_coeff2 * lje_coeff2 * c_oneSixth;
#    endif


#    ifdef CALC_ENERGIES
    E_lj         = 0.0f;
    E_el         = 0.0f;

#        ifdef EXCLUSION_FORCES /* Ewald or RF */
    if (nb_sci.shift == CENTRAL && pl_cj4[cij4_start].cj[0] == sci * c_numClPerSupercl)
    {
        /* we have the diagonal: add the charge and LJ self interaction energy term */
        for (i = 0; i < c_numClPerSupercl; i++)
        {
#            if defined EL_EWALD_ANY || defined EL_RF || defined EL_CUTOFF
            qi = xqib[i * c_clSize + tidxi].w;
            if (bFEP)
            {
                qBi = qBib[i * c_clSize + tidxi];
                E_el += _lambda_q * qi * qi + lambda_q * qBi * qBi;
            }
            else
            {
                E_el += qi * qi;
            }
#            endif

#            ifdef LJ_EWALD
#                if DISABLE_CUDA_TEXTURES
            if (bFEP)
            {
                E_lj += _lambda_v * LDG(
                        &nbparam.nbfp[atom_types[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2])
                    + lambda_v * LDG(
                        &nbparam.nbfp[atom_typesB[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2]);
            }
            else
            {
                E_lj += LDG(
                        &nbparam.nbfp[atom_types[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2]);
            }
#                else
            if (bFEP)
            {
                E_lj += _lambda_v * tex1Dfetch<float>(
                        nbparam.nbfp_texobj,
                        atom_types[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2);
                    + lambda_v * tex1Dfetch<float>(
                        nbparam.nbfp_texobj,
                        atom_typesB[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2);
            }
            else
            {
                E_lj += tex1Dfetch<float>(
                        nbparam.nbfp_texobj,
                        atom_types[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2);
            }
#                endif
#            endif
        }

        /* divide the self term(s) equally over the j-threads, then multiply with the coefficients. */
#            ifdef LJ_EWALD
        E_lj /= c_clSize * NTHREAD_Z;
        E_lj *= 0.5f * c_oneSixth * lje_coeff6_6;
#            endif

#            if defined EL_EWALD_ANY || defined EL_RF || defined EL_CUTOFF
        /* Correct for epsfac^2 due to adding qi^2 */
        E_el /= nbparam.epsfac * c_clSize * NTHREAD_Z;
#                if defined EL_RF || defined EL_CUTOFF
        E_el *= -0.5f * c_rf;
#                else
        E_el *= -beta * M_FLOAT_1_SQRTPI; /* last factor 1/sqrt(pi) */
#                endif
#            endif /* EL_EWALD_ANY || defined EL_RF || defined EL_CUTOFF */
    }
#        endif     /* EXCLUSION_FORCES */

#    endif /* CALC_ENERGIES */

#    ifdef EXCLUSION_FORCES
    const int nonSelfInteraction = !(nb_sci.shift == CENTRAL & tidxj <= tidxi);
#    endif

    /* loop over the j clusters = seen by any of the atoms in the current super-cluster;
     * The loop stride NTHREAD_Z ensures that consecutive warps-pairs are assigned
     * consecutive j4's entries.
     */
    for (j4 = cij4_start + tidxz; j4 < cij4_end; j4 += NTHREAD_Z)
    {
        wexcl_idx = pl_cj4[j4].imei[widx].excl_ind;
        imask     = pl_cj4[j4].imei[widx].imask;
        wexcl     = excl[wexcl_idx].pair[(tidx) & (warp_size - 1)];

#    ifndef PRUNE_NBL
        if (imask)
#    endif
        {
            /* Pre-load cj into shared memory on both warps separately */
            if ((tidxj == 0 | tidxj == 4) & (tidxi < c_nbnxnGpuJgroupSize))
            {
                cjs[tidxi + tidxj * c_nbnxnGpuJgroupSize / c_splitClSize] = pl_cj4[j4].cj[tidxi];
            }
            __syncwarp(c_fullWarpMask);

            /* Unrolling this loop
               - with pruning leads to register spilling;
               - on Kepler and later it is much slower;
               Tested with up to nvcc 7.5 */
            for (jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
            {
                if (imask & (superClInteractionMask << (jm * c_numClPerSupercl)))
                {
                    mask_ji = (1U << (jm * c_numClPerSupercl));

                    cj = cjs[jm + (tidxj & 4) * c_nbnxnGpuJgroupSize / c_splitClSize];
                    aj = cj * c_clSize + tidxj;

                    /* load j atom data */
                    xqbuf = xq[aj];
                    qBbuf = qB[aj];
                    xj    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);
                    qj_f  = xqbuf.w;
                    qBj_f = qBbuf;
#    ifndef LJ_COMB
                    typej = atom_types[aj];
                    typeBj= atom_typesB[aj];
#    else
                    ljcp_j = lj_comb[aj];
                    ljcpB_j = lj_combB[aj];
#    endif

                    fcj_buf = make_float3(0.0f);

#    if !defined PRUNE_NBL
#        pragma unroll 8
#    endif
                    for (i = 0; i < c_numClPerSupercl; i++)
                    {
                        if (imask & mask_ji)
                        {
                            ci = sci * c_numClPerSupercl + i; /* i cluster index */

                            /* all threads load an atom from i cluster ci into shmem! */
                            xqbuf = xqib[i * c_clSize + tidxi];
                            xi    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);

                            /* distance between i and j atoms */
                            rv = xi - xj;
                            r2 = norm2(rv);

#    ifdef PRUNE_NBL
                            /* If _none_ of the atoms pairs are in cutoff range,
                               the bit corresponding to the current
                               cluster-pair in imask gets set to 0. */
                            if (!__any_sync(c_fullWarpMask, r2 < rlist_sq))
                            {
                                imask &= ~mask_ji;
                            }
#    endif

                            int_bit = (wexcl & mask_ji) ? 1.0f : 0.0f;

                            /* cutoff & exclusion check */
#    ifdef EXCLUSION_FORCES
                            if ((r2 < rcoulomb_sq) * (nonSelfInteraction | (ci != cj)))
#    else
                            if ((r2 < rcoulomb_sq) * int_bit)
#    endif
                            {
                                /* load the rest of the i-atom parameters */
                                qi = xqbuf.w;
                                qBi= qBib[i * c_clSize + tidxi];

#    ifndef LJ_COMB
                                /* LJ 6*C6 and 12*C12 */
                                typei = atib[i * c_clSize + tidxi];
                                fetch_nbfp_c6_c12(c6, c12, nbparam, ntypes * typei + typej);
                                if (bFEP)
                                {
                                    typeBi= atBib[i * c_clSize + tidxi];
                                    fetch_nbfp_c6_c12(c6B, c12B, nbparam, ntypes * typeBi + typeBj);
                                    if (useSoftCore)
                                    {
                                        convert_c6_c12_to_sigma6_epsilon(c6, c12, &(sigma6[0]));
                                        convert_c6_c12_to_sigma6_epsilon(c6B, c12B, &(sigma6[1]));
                                    }
                                }
#    else
                                ljcp_i       = ljcpib[i * c_clSize + tidxi];
                                if (bFEP)
                                {
                                    ljcpB_i       = ljcpBib[i * c_clSize + tidxi];
                                }
#        ifdef LJ_COMB_GEOM
                                c6           = ljcp_i.x * ljcp_j.x;
                                c12          = ljcp_i.y * ljcp_j.y;
                                if (bFEP)
                                {
                                    c6B          = ljcpB_i.x * ljcpB_j.x;
                                    c12B         = ljcpB_i.y * ljcpB_j.y;
                                    if (useSoftCore)
                                    {
                                        convert_c6_c12_to_sigma6_epsilon(c6, c12, &(sigma6[0]));
                                        convert_c6_c12_to_sigma6_epsilon(c6B, c12B, &(sigma6[1]));
                                    }
                                }
#        else
                                /* LJ 2^(1/6)*sigma and 12*epsilon */
                                sigma   = ljcp_i.x + ljcp_j.x;
                                epsilon = ljcp_i.y * ljcp_j.y;
                                convert_sigma_epsilon_to_c6_c12(sigma, epsilon, &c6, &c12);
                                if (bFEP)
                                {
                                    sigmaB  = ljcpB_i.x + ljcpB_j.x;
                                    epsilonB= ljcpB_i.y * ljcpB_j.y;
                                    convert_sigma_epsilon_to_c6_c12(sigmaB, epsilonB, &c6B, &c12B);
                                    if (useSoftCore)
                                    {
                                        float sigma2 = sigma * sigma;
                                        sigma6[0]    = sigma2 * sigma2 * sigma2;
                                        float sigma2B= sigmaB * sigmaB;
                                        sigma6[1]    = sigma2B * sigma2B * sigma2B;
                                    }
                                }
// #            if defined CALC_ENERGIES || defined LJ_FORCE_SWITCH || defined LJ_POT_SWITCH
//                                 convert_sigma_epsilon_to_c6_c12(sigma, epsilon, &c6, &c12);
//                                 if (bFEP)
//                                 {
//                                     convert_sigma_epsilon_to_c6_c12(sigmaB, epsilonB, &c6B, &c12B);
//                                 }
// #            endif
#        endif /* LJ_COMB_GEOM */
#    endif     /* LJ_COMB */
                                rpm2 = r2 * r2;
                                rp = rpm2 * r2;
                                
                                // Ensure distance do not become so small that r^-12 overflows
                                r2 = max(r2, NBNXN_MIN_RSQ);
                                inv_r  = rsqrt(r2);
                                inv_r2 = inv_r * inv_r;
                                if (bFEP)
                                {
                                    if ((c12 > 0) && (c12B > 0) && (useSoftCore))
                                    {
                                        alpha_vdw_eff  = alpha_vdw;
                                        alpha_coul_eff = (useScBetaNO ? alpha_vdw_eff : alpha_coul);
                                        for (int k = 0; k < 2; k++)
                                        {
                                            if (sigma6[k] == 0)          sigma6[k] = sigma6_def;
                                            if (sigma6[k] < sigma6_min)  sigma6[k] = sigma6_min;

                                            if (useScBetaNO)
                                            {
                                                rpinvC[k] = 1.0f / (alpha_coul_eff * lfac_coul[k] * sigma6[k] + rp);
                                            }
                                            // else if (softCoreTreatment == SoftCoreTreatment::RPower6_2)
                                            // {
                                            //     rpinvC = one / (alpha_coul_eff * lfac_coul[i] + rpc);
                                            //     sqRoot<softCoreTreatment>(rpinvC, &rinvC, &rC);
                                            // }
                                            else
                                            {
                                                rpinvC[k] = 1.0f / (alpha_coul_eff * lfac_coul[k] + rp);
                                            }
                                            r2invC[k] = cbrt(rpinvC[k]);
                                            rinvC[k]  = sqrt(r2invC[k]);
                                            rC[k]     = 1.0f / rinvC[k];
                            
                                            if ((alpha_coul_eff != alpha_vdw_eff) || (!useScBetaNO))
                                            {
                                                rpinvV[k] = 1.0f / (alpha_vdw_eff * lfac_vdw[k] * sigma6[k] + rp);
                                                r2invV[k] = cbrt(rpinvV[k]);
                                                rinvV[k]  = sqrt(r2invV[k]);
                                                rV[k]     = 1.0f / rinvV[k];
                                            }
                                            else
                                            {
                                                /* We can avoid one expensive pow and one / operation */
                                                rpinvV[k] = rpinvC[k];
                                                r2invV[k] = r2invC[k];
                                                rinvV[k]  = rinvC[k];
                                                rV[k]     = rC[k];
                                            }
                                        }
                                    }
                                    else
                                    {
                                        rpinvC[0] = 1;           rpinvC[1] = 1;
                                        r2invC[0] = inv_r2;      r2invC[1] = inv_r2;
                                        rinvC[0]  = inv_r;       rinvC[1]  = inv_r;
                                        rC[0]     = r2 * inv_r;  rC[1]     = rC[0];
                                        rpinvV[0] = 1;           rpinvV[1] = 1;
                                        r2invV[0] = inv_r2;      r2invV[1] = inv_r2;
                                        rinvV[0]  = inv_r;       rinvV[1]  = inv_r;
                                        rV[0]     = rC[0];       rV[1]     = rC[0];
                                    }
                                    F_invr = 0.0f;
                                }
                                
// #    if !defined LJ_COMB_LB || defined CALC_ENERGIES
                                inv_r6 = inv_r2 * inv_r2 * inv_r2;
#        ifdef EXCLUSION_FORCES
                                /* We could mask inv_r2, but with Ewald
                                 * masking both inv_r6 and F_invr is faster */
                                inv_r6 *= int_bit;
                                if (bFEP)
                                {
                                    rpinvC[0] *= int_bit;    rpinvC[1] *= int_bit;
                                    rpinvV[0] *= int_bit;    rpinvV[1] *= int_bit;
                                }
#        endif /* EXCLUSION_FORCES */

                                if (!bFEP)
                                    F_invr = inv_r6 * (c12 * inv_r6 - c6) * inv_r2;
                                else
                                {
                                    // float Vvdw6[2] = {c6 * rpinvV[0], c6B * rpinvV[1]};
                                    // float Vvdw12[2]= {c12 * rpinvV[0] * rpinvV[0], c12B * rpinvV[1] * rpinvV[1]};
                                    float FscalV[2]= {c12 * rpinvV[0] - c6, c12B * rpinvV[1] - c6B};
                                    for (int k = 0; k < 2; k++)
                                    {
                                        F_invr += lfac_vdw[1-k] * rpinvV[k] * rpinvV[k] * FscalV[k] * rpm2;
                                    }
                                }
#        if defined CALC_ENERGIES || defined LJ_POT_SWITCH
                                if (!bFEP)
                                    E_lj_p = int_bit
                                         * (c12 * (inv_r6 * inv_r6 + nbparam.repulsion_shift.cpot) * c_oneTwelveth
                                            - c6 * (inv_r6 + nbparam.dispersion_shift.cpot) * c_oneSixth);
                                else
                                {
                                    E_lj_p += int_bit * lfac_vdw[1]
                                         * (c12 * (rpinvV[0] * rpinvV[0] + nbparam.repulsion_shift.cpot) * c_oneTwelveth
                                            - c6 * (rpinvV[0] + nbparam.dispersion_shift.cpot) * c_oneSixth);
                                    E_lj_p += int_bit * lfac_vdw[0]
                                         * (c12B * (rpinvV[1] * rpinvV[1] + nbparam.repulsion_shift.cpot) * c_oneTwelveth
                                            - c6B * (rpinvV[1] + nbparam.dispersion_shift.cpot) * c_oneSixth);
                                }
#        endif
// #    else /* !LJ_COMB_LB || CALC_ENERGIES */
//                                 float sig_r  = sigma * inv_r;
//                                 float sig_r2 = sig_r * sig_r;
//                                 float sig_r6 = sig_r2 * sig_r2 * sig_r2;
// #        ifdef EXCLUSION_FORCES
//                                 sig_r6 *= int_bit;
// #        endif /* EXCLUSION_FORCES */

//                                 F_invr = epsilon * sig_r6 * (sig_r6 - 1.0f) * inv_r2;
// #    endif     /* !LJ_COMB_LB || CALC_ENERGIES */

// TODO: adding fep into vdw modifier with force switch
#    ifdef LJ_FORCE_SWITCH
#        ifdef CALC_ENERGIES
                                if (!bFEP)
                                calculate_force_switch_F_E(nbparam, c6, c12, inv_r, r2, &F_invr, &E_lj_p);
                                else
                                calculate_force_switch_F_E(nbparam, c6, c12, rinvV[0], rV[0] * rV[0], &F_invr, &E_lj_p);
#        else
                                if (!bFEP)
                                calculate_force_switch_F(nbparam, c6, c12, inv_r, r2, &F_invr);
                                else
                                calculate_force_switch_F(nbparam, c6, c12, rinvV[0], rV[0] * rV[0], &F_invr);
#        endif /* CALC_ENERGIES */
#    endif     /* LJ_FORCE_SWITCH */

// TODO: adding fep into vdw Ewald
#    ifdef LJ_EWALD
#        ifdef LJ_EWALD_COMB_GEOM
#            ifdef CALC_ENERGIES
                                if (!bFEP)
                                calculate_lj_ewald_comb_geom_F_E(nbparam, typei, typej, r2, inv_r2,
                                                                 lje_coeff2, lje_coeff6_6, int_bit,
                                                                 &F_invr, &E_lj_p);
                                else
                                calculate_lj_ewald_comb_geom_F_E(nbparam, typei, typej, rV[0] * rV[0], inv_r2,
                                                                 lje_coeff2, lje_coeff6_6, int_bit,
                                                                 &F_invr, &E_lj_p);
#            else
                                if (!bFEP)
                                calculate_lj_ewald_comb_geom_F(nbparam, typei, typej, r2, inv_r2,
                                                               lje_coeff2, lje_coeff6_6, &F_invr);
                                else
                                calculate_lj_ewald_comb_geom_F(nbparam, typei, typej, rV[0] * rV[0], inv_r2,
                                                               lje_coeff2, lje_coeff6_6, &F_invr);
#            endif /* CALC_ENERGIES */
#        elif defined LJ_EWALD_COMB_LB
                                if (!bFEP)
                                calculate_lj_ewald_comb_LB_F_E(nbparam, typei, typej, r2, inv_r2,
                                                               lje_coeff2, lje_coeff6_6,
#            ifdef CALC_ENERGIES
                                                               int_bit, &F_invr, &E_lj_p
#            else
                                                               0, &F_invr, nullptr
#            endif /* CALC_ENERGIES */
                                );
                                else
                                calculate_lj_ewald_comb_LB_F_E(nbparam, typei, typej, rV[0] * rV[0], inv_r2,
                                                               lje_coeff2, lje_coeff6_6,
                                                               0, &F_invr, nullptr
                                );
#        endif     /* LJ_EWALD_COMB_GEOM */
#    endif         /* LJ_EWALD */

// TODO: adding fep into vdw modifier with poteintial switch
#    ifdef LJ_POT_SWITCH
#        ifdef CALC_ENERGIES
                                if (!bFEP)
                                calculate_potential_switch_F_E(nbparam, inv_r, r2, &F_invr, &E_lj_p);
                                else
                                calculate_potential_switch_F_E(nbparam, inv_r, rV[0] * rV[0], &F_invr, &E_lj_p);
#        else
                                if (!bFEP)
                                calculate_potential_switch_F(nbparam, inv_r, r2, &F_invr, &E_lj_p);
                                else
                                calculate_potential_switch_F(nbparam, inv_r, rV[0] * rV[0], &F_invr, &E_lj_p);
#        endif /* CALC_ENERGIES */
#    endif     /* LJ_POT_SWITCH */

#    ifdef VDW_CUTOFF_CHECK
                                /* Separate VDW cut-off check to enable twin-range cut-offs
                                 * (rvdw < rcoulomb <= rlist)
                                 */
                                vdw_in_range = (r2 < rvdw_sq) ? 1.0f : 0.0f;
                                F_invr *= vdw_in_range;
#        ifdef CALC_ENERGIES
                                E_lj_p *= vdw_in_range;
#        endif
#    endif /* VDW_CUTOFF_CHECK */

#    ifdef CALC_ENERGIES
                                E_lj += E_lj_p;
#    endif

                                float qq[2] = {qi * qj_f, qi * qj_f};
                                if (bFEP) qq[1] = qBi * qBj_f;
#    ifdef EL_CUTOFF
#        ifdef EXCLUSION_FORCES
                                if (!bFEP)
                                F_invr += qi * qj_f * int_bit * inv_r2 * inv_r;
                                else
                                    for (int k = 0; k < 2; k++)
                                    {
                                        F_invr += lfac_coul[1-k] * qq[k] * int_bit * rinvC[k] * rpinvC[k] * rpm2;
                                    }
#        else
                                if (!bFEP)
                                F_invr += qi * qj_f * inv_r2 * inv_r;
                                else
                                    for (int k = 0; k < 2; k++)
                                    {
                                        F_invr += lfac_coul[1-k] * qq[k] * rinvC[k] * rpinvC[k] * rpm2;
                                    }
#        endif
#    endif
#    ifdef EL_RF
                                if (!bFEP)
                                F_invr += qi * qj_f * (int_bit * inv_r2 * inv_r - two_k_rf);
                                else
                                    for (int k = 0; k < 2; k++)
                                    {
                                        F_invr += lfac_coul[1-k] * qq[k] * (int_bit * rinvC[k] - two_k_rf * rC[k] * rC[k]) * rpinvC[k] * rpm2;
                                    }
#    endif
#    if defined   EL_EWALD_ANA
                                if (!bFEP)
                                F_invr += qi * qj_f
                                          * (int_bit * inv_r2 * inv_r + pmecorrF(beta2 * r2) * beta3);
                                else
                                    for (int k = 0; k < 2; k++)
                                    {
                                        F_invr += lfac_coul[1-k] * qq[k]
                                          * (int_bit * rinvC[k] + pmecorrF(beta2 * rC[k] * rC[k]) * beta3 * rC[k] * rC[k]) * rpinvC[k] * rpm2;
                                    }
#    elif defined EL_EWALD_TAB
                                if (!bFEP)
                                F_invr += qi * qj_f
                                          * (int_bit * inv_r2
                                             - interpolate_coulomb_force_r(nbparam, r2 * inv_r))
                                          * inv_r;
                                else
                                    for (int k = 0; k < 2; k++)
                                    {
                                        F_invr += lfac_coul[1-k] * qq[k]
                                          * (int_bit
                                             - interpolate_coulomb_force_r(nbparam, rC[k]) * rC[k])
                                          * rpinvC[k] * rpm2;
                                    }
#    endif /* EL_EWALD_ANA/TAB */

#    ifdef CALC_ENERGIES
#        ifdef EL_CUTOFF
                                if (!bFEP)
                                E_el += qi * qj_f * (int_bit * inv_r - c_rf);
                                else
                                    for (int k = 0; k < 2; k++)
                                    {
                                        E_el += lfac_coul[1-k] * qq[k] * (int_bit * rinvC[k] - c_rf);
                                    }
#        endif
#        ifdef EL_RF
                                if (!bFEP)
                                E_el += qi * qj_f * (int_bit * inv_r + 0.5f * two_k_rf * r2 - c_rf);
                                else
                                    for (int k = 0; k < 2; k++)
                                    {
                                        E_el += lfac_coul[1-k] * qq[k] * (int_bit * rinvC[k] + 0.5f * two_k_rf * rC[k] * rC[k] - c_rf);
                                    }
#        endif
#        ifdef EL_EWALD_ANY
                                /* 1.0f - erff is faster than erfcf */
                                if (!bFEP)
                                E_el += qi * qj_f
                                        * (inv_r * (int_bit - erff(r2 * inv_r * beta)) - int_bit * ewald_shift);
                                else
                                    for (int k = 0; k < 2; k++)
                                    {
                                        E_el += lfac_coul[1-k] * qq[k]
                                                * (rinvC[k] * (int_bit - erff(rC[k] * beta)) - int_bit * ewald_shift);
                                    }
#        endif /* EL_EWALD_ANY */
#    endif
                                f_ij = rv * F_invr;

                                /* accumulate j forces in registers */
                                fcj_buf -= f_ij;

                                /* accumulate i forces in registers */
                                fci_buf[i] += f_ij;
                            }
                        }

                        /* shift the mask bit by 1 */
                        mask_ji += mask_ji;
                    }

                    /* reduce j forces */
                    reduce_force_j_warp_shfl(fcj_buf, f, tidxi, aj, c_fullWarpMask);
                }
            }
#    ifdef PRUNE_NBL
            /* Update the imask with the new one which does not contain the
               out of range clusters anymore. */
            pl_cj4[j4].imei[widx].imask = imask;
#    endif
        }
        // avoid shared memory WAR hazards between loop iterations
        __syncwarp(c_fullWarpMask);
    }

    /* skip central shifts when summing shift forces */
    if (nb_sci.shift == CENTRAL)
    {
        bCalcFshift = false;
    }

    float fshift_buf = 0.0f;

    /* reduce i forces */
    for (i = 0; i < c_numClPerSupercl; i++)
    {
        ai = (sci * c_numClPerSupercl + i) * c_clSize + tidxi;
        reduce_force_i_warp_shfl(fci_buf[i], f, &fshift_buf, bCalcFshift, tidxj, ai, c_fullWarpMask);
    }

    /* add up local shift forces into global mem, tidxj indexes x,y,z */
    if (bCalcFshift && (tidxj & 3) < 3)
    {
        atomicAdd(&(atdat.fshift[nb_sci.shift].x) + (tidxj & 3), fshift_buf);
    }

#    ifdef CALC_ENERGIES
    /* reduce the energies over warps and store into global memory */
    reduce_energy_warp_shfl(E_lj, E_el, e_lj, e_el, tidx, c_fullWarpMask);
#    endif
}
#endif /* FUNCTION_DECLARATION_ONLY */

#undef NTHREAD_Z
#undef MIN_BLOCKS_PER_MP
#undef THREADS_PER_BLOCK

#undef EL_EWALD_ANY
#undef EXCLUSION_FORCES
#undef LJ_EWALD

#undef LJ_COMB
