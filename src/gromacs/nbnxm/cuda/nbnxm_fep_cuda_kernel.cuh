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
// #ifdef PRUNE_NBL
// #    ifdef CALC_ENERGIES
//         __global__ void NB_KERNEL_FUNC_NAME(nbnxn_kernel, _VF_prune_cuda)
// #    else
//         __global__ void NB_KERNEL_FUNC_NAME(nbnxn_kernel, _F_prune_cuda)
// #    endif /* CALC_ENERGIES */
// #else
#    ifdef CALC_ENERGIES
        __global__ void NB_KERNEL_FUNC_NAME(nbnxn_fep_kernel, _VF_cuda)
#    else
        __global__ void NB_KERNEL_FUNC_NAME(nbnxn_fep_kernel, _F_cuda)
#    endif /* CALC_ENERGIES */
// #endif     /* PRUNE_NBL */
                (const cu_atomdata_t atdat, const cu_nbparam_t nbparam, const cu_feplist_t feplist, bool bCalcFshift)
#ifdef FUNCTION_DECLARATION_ONLY
                        ; /* Only do function declaration, omit the function body. */
#else
{
    /* convenience variables */
    const nbnxn_sci_t* pl_sci = plist.sci;
    
    const bool bFEP = nbparam.bFEP;
    bool       bFEPpair = 0;
    int nFEP, nnFEP;
    const float alpha_coul = nbparam.alpha_coul;
    const float alpha_vdw  = nbparam.alpha_vdw;
    float alpha_coul_eff   = alpha_coul;
    float alpha_vdw_eff    = alpha_vdw;
    const bool useSoftCore = (alpha_vdw != 0.0);
    const bool useScBetaNO = (alpha_coul == 0.0);
    const float sigma6_def = nbparam.sc_sigma6;
    const float sigma6_min = nbparam.sc_sigma6_min;
    const float lambda_q   = nbparam.lambda_q;
    const float _lambda_q  = 1 - lambda_q;
    const float lambda_v   = nbparam.lambda_v;
    const float _lambda_v  = 1 - lambda_v;

    const float lfac_coul[2] = {lambda_q, _lambda_q};
    const float lfac_vdw[2]  = {lambda_v, _lambda_v};
    const float LFC[2]       = {_lambda_q, lambda_q};
    const float LFV[2]       = {_lambda_v, lambda_v};

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

    float rinvC, rinvV, r2C, r2V, rpinvC, rpinvV, rpinvC_nonzero, rpinvV_nonzero;
    float sigma6[2], c6AB[2], c12AB[2];
    float qq[2];
    float FscalC[2], FscalV[2], Vvdw[2], Vcoul[2];
    int typeiAB[2], typejAB[2];

#    ifdef LJ_COMB_LB
    float sigmaAB[2], epsilonAB[2];
#    endif

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
    unsigned int tidxi_global = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
                                + threadIdx.z * blockDim.y * blockDim.x
                                + threadIdx.y * blockDim.x + threadIdx.x;
   
#    if NTHREAD_Z == 1
    unsigned int tidxz = 0;
#    else
    unsigned int  tidxz = threadIdx.z;
#    endif
    unsigned int bidx  = blockIdx.x;
    unsigned int widx  = tidx / warp_size; /* warp index */

#ifndef LJ_COMB
    int          type, typej;
    int          typeBi, typeBj;
#endif
    int          i, jm, j4, wexcl_idx;
    float        qi, qj_f, r2, rpm2, rp, inv_r, inv_r2;
    float        qBi, qBj_f;
// #if !defined LJ_COMB_LB || defined CALC_ENERGIES
    float        inv_r6, c6, c12;
// #endif
#ifdef LJ_COMB_LB
    float        sigma, epsilon;
#endif

    float        int_bit, F_invr;
#    ifdef CALC_ENERGIES
    float        E_lj, E_el;
#    endif
#    if defined CALC_ENERGIES || defined LJ_POT_SWITCH
    float        E_lj_p;
#    endif
    float4       xqbuf;
    float3       xi, xj, rv, f_ij, fci_buf, fcj_buf;

    /*********************************************************************
     * Set up shared memory pointers.
     * sm_nextSlotPtr should always be updated to point to the "next slot",
     * that is past the last point where data has been stored.
     */
//     extern __shared__ char sm_dynamicShmem[];
//     char*                  sm_nextSlotPtr = sm_dynamicShmem;
//     static_assert(sizeof(char) == 1,
//                   "The shared memory offset calculation assumes that char is 1 byte");

//     /* shmem buffer for i x+q pre-loading */
//     float4* xqib = (float4*)sm_nextSlotPtr;
//     sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*xqib));

//     float* qBib;
//     if (bFEP)
//     {
//         qBib = (float*)sm_nextSlotPtr;
//         sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*qBib));
//     }

//     /* shmem buffer for cj, for each warp separately */
//     int* cjs = (int*)(sm_nextSlotPtr);
//     /* the cjs buffer's use expects a base pointer offset for pairs of warps in the j-concurrent execution */
//     cjs += tidxz * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize;
//     sm_nextSlotPtr += (NTHREAD_Z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize * sizeof(*cjs));

// #ifndef LJ_COMB
//     /* shmem buffer for i atom-type pre-loading */
//     int* atib = (int*)sm_nextSlotPtr;
//     sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*atib));

//     int* atBib;
//     if (bFEP){
//         atBib = (int*)sm_nextSlotPtr;
//         sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*atBib));
//     }
// #else
//     /* shmem buffer for i-atom LJ combination rule parameters */
//     float2* ljcpib = (float2*)sm_nextSlotPtr;
//     sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*ljcpib));

//     float2* ljcpBib;
//     if (bFEP){
//         ljcpBib = (float2*)sm_nextSlotPtr;
//         sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*ljcpBib));
//     }
// #endif
    
    /*********************************************************************/

    // Extract pair list data
    const int  nri    = feplist->nri;
    const int* iinr   = feplist->iinr;
    const int* jindex = feplist->jindex;
    const int* jjnr   = feplist->jjnr;
    const int* shift  = feplist->shift;
    const int* gid    = feplist->gid;

//     if (tidxz == 0)
//     {
//         /* Pre-load i-atom x and q into shared memory */
//         ci = sci * c_numClPerSupercl + tidxj;
//         ai = ci * c_clSize + tidxi;

//         float* shiftptr = (float*)&shift_vec[nb_sci.shift];
//         xqbuf = xq[ai] + make_float4(LDG(shiftptr), LDG(shiftptr + 1), LDG(shiftptr + 2), 0.0f);
//         xqbuf.w *= nbparam.epsfac;
//         xqib[tidxj * c_clSize + tidxi] = xqbuf;
//         qBib[tidxj * c_clSize + tidxi] = qB[ai] * nbparam.epsfac;

// #ifndef LJ_COMB
//         /* Pre-load the i-atom types into shared memory */
//         atib[tidxj * c_clSize + tidxi] = atom_types[ai];
//         atBib[tidxj * c_clSize + tidxi] = atom_typesB[ai];
// #else
//         /* Pre-load the LJ combination parameters into shared memory */
//         ljcpib[tidxj * c_clSize + tidxi] = lj_comb[ai];
//         ljcpBib[tidxj * c_clSize + tidxi] = lj_combB[ai];
// #endif
//     }
//     __syncthreads();

#    ifdef LJ_EWALD
    /* TODO: we are trading registers with flops by keeping lje_coeff-s, try re-calculating it later */
    lje_coeff2   = nbparam.ewaldcoeff_lj * nbparam.ewaldcoeff_lj;
    lje_coeff6_6 = lje_coeff2 * lje_coeff2 * lje_coeff2 * c_oneSixth;
#    endif


#    ifdef CALC_ENERGIES
    E_lj         = 0.0f;
    E_el         = 0.0f;

// #        ifdef EXCLUSION_FORCES /* Ewald or RF */
//     if (nb_sci.shift == CENTRAL && pl_cj4[cij4_start].cj[0] == sci * c_numClPerSupercl)
//     {
//         /* we have the diagonal: add the charge and LJ self interaction energy term */
//         for (i = 0; i < c_numClPerSupercl; i++)
//         {
// #            if defined EL_EWALD_ANY || defined EL_RF || defined EL_CUTOFF
//             qi = xqib[i * c_clSize + tidxi].w;
//             if (bFEP)
//             {
//                 qBi = qBib[i * c_clSize + tidxi];
//                 E_el += _lambda_q * qi * qi + lambda_q * qBi * qBi;
//             }
//             else
//             {
//                 E_el += qi * qi;
//             }
// #            endif

// #            ifdef LJ_EWALD
// #                if DISABLE_CUDA_TEXTURES
//             if (bFEP)
//             {
//                 E_lj += _lambda_v * LDG(
//                         &nbparam.nbfp[atom_types[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2])
//                     + lambda_v * LDG(
//                         &nbparam.nbfp[atom_typesB[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2]);
//             }
//             else
//             {
//                 E_lj += LDG(
//                         &nbparam.nbfp[atom_types[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2]);
//             }
// #                else
//             if (bFEP)
//             {
//                 E_lj += _lambda_v * tex1Dfetch<float>(
//                         nbparam.nbfp_texobj,
//                         atom_types[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2);
//                     + lambda_v * tex1Dfetch<float>(
//                         nbparam.nbfp_texobj,
//                         atom_typesB[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2);
//             }
//             else
//             {
//                 E_lj += tex1Dfetch<float>(
//                         nbparam.nbfp_texobj,
//                         atom_types[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2);
//             }
// #                endif
// #            endif
//         }

//         /* divide the self term(s) equally over the j-threads, then multiply with the coefficients. */
// #            ifdef LJ_EWALD
//         E_lj /= c_clSize * NTHREAD_Z;
//         E_lj *= 0.5f * c_oneSixth * lje_coeff6_6;
// #            endif

// #            if defined EL_EWALD_ANY || defined EL_RF || defined EL_CUTOFF
//         /* Correct for epsfac^2 due to adding qi^2 */
//         E_el /= nbparam.epsfac * c_clSize * NTHREAD_Z;
// #                if defined EL_RF || defined EL_CUTOFF
//         E_el *= -0.5f * c_rf;
// #                else
//         E_el *= -beta * M_FLOAT_1_SQRTPI; /* last factor 1/sqrt(pi) */
// #                endif
// #            endif /* EL_EWALD_ANY || defined EL_RF || defined EL_CUTOFF */
//     }
//     // printf("E_el=%e\n\n", E_el);
// #        endif     /* EXCLUSION_FORCES */

// #    endif /* CALC_ENERGIES */

// #    ifdef EXCLUSION_FORCES
//     const int nonSelfInteraction = !(nb_sci.shift == CENTRAL & tidxj <= tidxi);
// #    endif

    /* loop over the j clusters = seen by any of the atoms in the current super-cluster;
     * The loop stride NTHREAD_Z ensures that consecutive warps-pairs are assigned
     * consecutive j4's entries.
     */
    nFEP = 0;nnFEP = 0;

    int npair_within_cutoff = 0;

    // float* shiftptr = (float*)&shift_vec[nb_sci.shift];

    if (tidxi_global < nri)
    {
        const int  nj0   = jindex[n];
        const int  nj1   = jindex[n + 1];
        const int  ai    = iinr[n];
        const int  n     = tidxi_global;
        float* shiftptr  = (float*)&shift_vec[shift[n]];
        xqbuf = xq[ai] + make_float4(LDG(shiftptr), LDG(shiftptr + 1), LDG(shiftptr + 2), 0.0f);
        xqbuf.w *= nbparam.epsfac;

        xi = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);
        qi = xqbuf.w;
        qBi = qB[ai] * nbparam.epsfac;
        // const int  is3   = 3 * shift[n];
        // const real shX   = shiftvec[is3];
        // const real shY   = shiftvec[is3 + 1];
        // const real shZ   = shiftvec[is3 + 2];
        // const int  ii3   = 3 * ii;
        // const real ix    = shX + x[ii3 + 0];
        // const real iy    = shY + x[ii3 + 1];
        // const real iz    = shZ + x[ii3 + 2];

#ifndef LJ_COMB
        typeiAB[0] = atom_types[ai];  typeiAB[1] = atom_typesB[ai];
#else
        ljcp_iAB[0] = lj_comb[ai];  ljcp_iAB[1] = lj_combB[ai];
#endif
        // int  = {typej, typeBj};
        // const int  ntiA  = 2 * ntype * typeA[ii];
        // const int  ntiB  = 2 * ntype * typeB[ii];
        real       vctot = 0;
        real       vvtot = 0;

        fci_buf = make_float3(0.0f);
        // real       fix   = 0;
        // real       fiy   = 0;
        // real       fiz   = 0;

        for (int k = nj0; k < nj1; k++)
        {
            int        tj[NSTATES];
            const int  aj  = jjnr[k];
            real       c6[NSTATES], c12[NSTATES], qq[NSTATES], Vcoul[NSTATES], Vvdw[NSTATES];
            real       r, rinv, rp, rpm2, rpc, rpcm2;
            real       alpha_vdw_eff, alpha_coul_eff, sigma_pow[NSTATES];
            // const real dx  = ix - x[j3];
            // const real dy  = iy - x[j3 + 1];
            // const real dz  = iz - x[j3 + 2];
            // const real rsq = dx * dx + dy * dy + dz * dz;
            // SCReal     FscalC[NSTATES], FscalV[NSTATES]; /* Needs double for sc_power==48 */

            /* load j atom data */
            xqbuf = xq[aj];
            xj    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);
            qj_f  = xqbuf.w;
            qBj_f = qB[aj];
            qq[0] = qi * qj_f;  qq[1] = qBi * qBj_f;
#    ifndef LJ_COMB
            typejAB[0] = atom_types[aj];    typejAB[1] = atom_typesB[aj];
#    else
            ljcp_jAB[0] = lj_comb[aj];  ljcp_jAB[1] = lj_combB[aj];
#    endif
            fcj_buf = make_float3(0.0f);
            /* distance between i and j atoms */
            rv = xi - xj;
            r2 = norm2(rv);

            if (nlist->excl_fep == nullptr || nlist->excl_fep[k])
            {
                /* cutoff & exclusion check */
                if (r2 < rcoulomb_sq)
                {
                    npair_within_cutoff++;
                    rpm2 = r2 * r2;
                    rp = rpm2 * r2;
                                
                    // Ensure distance do not become so small that r^-12 overflows
                    r2 = max(r2, NBNXN_MIN_RSQ);
                    inv_r  = rsqrt(r2);
                    inv_r2 = inv_r * inv_r;

                    if (bFEP)
                    {
                        F_invr = 0.0f;
                        for (int k = 0; k < 2; k++)
                        {
#    ifndef LJ_COMB
                            /* LJ 6*C6 and 12*C12 */
                            fetch_nbfp_c6_c12(c6AB[k], c12AB[k], nbparam, ntypes * typeiAB[k] + typejAB[k]);
                            if (useSoftCore)
                                convert_c6_c12_to_sigma6_epsilon(c6AB[k], c12AB[k], &(sigma6[k]));
#    else
#        ifdef LJ_COMB_GEOM
                            c6AB[k]  = ljcp_iAB[k].x * ljcp_jAB[k].x;
                            c12AB[k] = ljcp_iAB[k].y * ljcp_jAB[k].y;
                            if (useSoftCore)
                                convert_c6_c12_to_sigma6_epsilon(c6AB[k], c12AB[k], &(sigma6[k]));
#        else
                            /* LJ 2^(1/6)*sigma and 12*epsilon */
                            sigmaAB[k]   = ljcp_iAB[k].x + ljcp_jAB[k].x;
                            epsilonAB[k] = ljcp_iAB[k].y * ljcp_jAB[k].y;
                            convert_sigma_epsilon_to_c6_c12(sigmaAB[k], epsilonAB[k], &(c6AB[k]), &(c12AB[k]));
                            if (useSoftCore)
                            {
                                float sigma2 = sigmaAB[k] * sigmaAB[k];
                                sigma6[k]    = sigma2 * sigma2 * sigma2 * 0.5;
                            }
// #            if defined CALC_ENERGIES || defined LJ_FORCE_SWITCH || defined LJ_POT_SWITCH
//                                 convert_sigma_epsilon_to_c6_c12(sigma, epsilon, &c6, &c12);
//                                 if (bFEP)
//                                 {
//                                     convert_sigma_epsilon_to_c6_c12(sigmaB, epsilonB, &c6B, &c12B);
//                                 }
// #            endif
#        endif /* LJ_COMB_GEOM */
                        }
#    endif     /* LJ_COMB */
                        if (qq[0] == qq[1] && c6AB[0] == c6AB[1] && c12AB[0] == c12AB[1]) bFEPpair = 0;
                        else bFEPpair = 1;
                    }
                    else
                        bFEPpair = 0;

                    if (bFEPpair)
                    {
                        nFEP++;
                        inv_r6 = inv_r2 * inv_r2 * inv_r2;
#        ifdef EXCLUSION_FORCES
                        /* We could mask inv_r2, but with Ewald
                         * masking both inv_r6 and F_invr is faster */
                        // inv_r6 *= int_bit;
#        endif /* EXCLUSION_FORCES */
                        for (int k = 0; k < 2; k++)
                        {
                            FscalC[k] = 0;
                            FscalV[k] = 0;
                            Vcoul[k]  = 0;
                            Vvdw[k]   = 0;
                            if ((qq[k] != 0) || (c6AB[k] != 0) || (c12AB[k] != 0))
                            {
                                if ((c12AB[0] == 0 || c12AB[1] == 0) && (useSoftCore))
                                {
                                    alpha_vdw_eff  = alpha_vdw;
                                    alpha_coul_eff = (useScBetaNO ? alpha_vdw_eff : alpha_coul);
                                    if (sigma6[k] == 0)          sigma6[k] = sigma6_def;
                                    if (sigma6[k] < sigma6_min)  sigma6[k] = sigma6_min;

                                    if (useScBetaNO)
                                    {
                                        rpinvC = 1.0f / (alpha_coul_eff * lfac_coul[k] * sigma6[k] + rp);
                                    }
                                    // else if (softCoreTreatment == SoftCoreTreatment::RPower6_2)
                                    // {
                                    //     rpinvC = one / (alpha_coul_eff * lfac_coul[i] + rpc);
                                    //     sqRoot<softCoreTreatment>(rpinvC, &rinvC, &rC);
                                    // }
                                    else
                                    {
                                        rpinvC = 1.0f / (alpha_coul_eff * lfac_coul[k] + rp);
                                    }
                                    r2C    = rcbrt(rpinvC);
                                    rinvC  = rsqrt(r2C);
                            
                                    if ((alpha_coul_eff != alpha_vdw_eff) || (!useScBetaNO))
                                    {
                                        rpinvV = 1.0f / (alpha_vdw_eff * lfac_vdw[k] * sigma6[k] + rp);
                                        r2V    = rcbrt(rpinvV);
                                        rinvV  = rsqrt(r2V);
                                    }
                                    else
                                    {
                                        /* We can avoid one expensive pow and one / operation */
                                        rpinvV = rpinvC;
                                        r2V    = r2C;
                                        rinvV  = rinvC;
                                    }
                                }
                                else
                                {
                                    rpinvC = inv_r6;
                                    r2C    = r2;
                                    rinvC  = inv_r;
                                    rpinvV = inv_r6;
                                    r2V    = r2;
                                    rinvV  = inv_r;
                                }
#        endif /* EXCLUSION_FORCES */
                                // #    if !defined LJ_COMB_LB || defined CALC_ENERGIES
                                    
                                if (c6AB[k] != 0 || c12AB[k] != 0)
                                {
                                    float Vvdw6 = c6AB[k] * rpinvV;
                                    float Vvdw12= c12AB[k] * rpinvV * rpinvV;
                                    FscalV[k] = Vvdw12 - Vvdw6;
#        if defined CALC_ENERGIES || defined LJ_POT_SWITCH
                                    Vvdw[k] = int_bit * 
                                            ((Vvdw12 + c12AB[k] * nbparam.repulsion_shift.cpot) * c_oneTwelveth - (Vvdw6 + c6AB[k] * nbparam.dispersion_shift.cpot) * c_oneSixth);
#        endif
// TODO: adding fep into vdw modifier with force switch
#    ifdef LJ_FORCE_SWITCH 
#        ifdef CALC_ENERGIES
                                    calculate_force_switch_F_E(nbparam, c6AB[k], c12AB[k], rinvV, r2V, &(FscalV[k]), &(Vvdw[k]));
#        else
                                    calculate_force_switch_F(nbparam, c6AB[k], c12AB[k], rinvV, r2V, &(FscalV[k]));
#        endif /* CALC_ENERGIES */
#    endif     /* LJ_FORCE_SWITCH */

// TODO: adding fep into vdw Ewald
#    ifdef LJ_EWALD
#        ifdef LJ_EWALD_COMB_GEOM
#            ifdef CALC_ENERGIES
                                    calculate_lj_ewald_comb_geom_F_E(nbparam, typeiAB[0], typejAB[0], r2V, rinvV * rinvV,
                                                                     lje_coeff2, lje_coeff6_6, int_bit,
                                                                     &(FscalV[k]), &(Vvdw[k]));
#            else
                                    calculate_lj_ewald_comb_geom_F(nbparam, typeiAB[0], typejAB[0], r2V, rinvV * rinvV,
                                                                   lje_coeff2, lje_coeff6_6, &(FscalV[k]));
#            endif /* CALC_ENERGIES */
#        elif defined LJ_EWALD_COMB_LB
                                    calculate_lj_ewald_comb_LB_F_E(nbparam, typeiAB[0], typejAB[0], r2V, rinvV * rinvV,
                                                                   lje_coeff2, lje_coeff6_6,
#            ifdef CALC_ENERGIES
                                                                   int_bit, &(FscalV[k]), &(Vvdw[k])
#            else
                                                                   0, &(FscalV[k]), nullptr
#            endif /* CALC_ENERGIES */
                                    );
#        endif     /* LJ_EWALD_COMB_GEOM */
#    endif         /* LJ_EWALD */

#    ifdef LJ_POT_SWITCH
#        ifdef CALC_ENERGIES
                                    calculate_potential_switch_F_E(nbparam, rinvC, r2V, &(FscalV[i]), &(Vvdw[i]));
#        else
                                    calculate_potential_switch_F(nbparam, rinvC, r2V, &(FscalV[i]), &(Vvdw[i]));
#        endif /* CALC_ENERGIES */
#    endif     /* LJ_POT_SWITCH */

#    ifdef VDW_CUTOFF_CHECK
                                    /* Separate VDW cut-off check to enable twin-range cut-offs
                                     * (rvdw < rcoulomb <= rlist)
                                     */
                                    vdw_in_range = (r2V < rvdw_sq) ? 1.0f : 0.0f;
                                    FscalV[k] *= vdw_in_range;
#        ifdef CALC_ENERGIES
                                    Vvdw[k] *= vdw_in_range;
#        endif
#    endif /* VDW_CUTOFF_CHECK */
                                }

                                if (qq[k] != 0)
                                {
#    ifdef EL_CUTOFF
#        ifdef EXCLUSION_FORCES
                                    FscalC[k] = qq[k] * int_bit * rinvC;
#        else
                                    FscalC[k] = qq[k] * rinvC;
#        endif
#    endif
#    ifdef EL_RF
                                    FscalC[k] = qq[k] * (int_bit * rinvC - two_k_rf * r2C);
#    endif
#    if defined   EL_EWALD_ANA
// #ifdef CALC_ENERGIES
//                     if ((c6AB[0] != 0 || c6AB[1] != 0 || c12AB[0]!=0 || c12AB[1]!=0 || qq[0]!=0 || qq[1]!=0) && (k == 1))
//                     printf("interaction [%d-%d], r2=[%e], rinvC=[%e], ewald corr.F=[%.5f], ewald corr.V=[%.5f], qq=[%e, %e], c6=[%e, %e], c12=[%e, %e], FscalC=[%e, %e], FscalV=[%e, %e], Vcoul=[%e, %e], Vvdw=[%e, %e], mask=%f\n", ai, aj, r2, rinvC, pmecorrF(beta2 * r2) * beta3, inv_r * erff(r2 * inv_r * beta),
//                     qq[0], qq[1], c6AB[0], c6AB[1], c12AB[0], c12AB[1], 
//                     qq[0] * int_bit * rinvC0 * rpinvC[0] * rpm2, qq[1] * int_bit * rinvC * rpinvC[1] * rpm2, 
//                     FscalV[0] * rpm2, FscalV[1] * rpinvV * rpm2, qq[0] * int_bit * (rinvC0 - ewald_shift), qq[1] * int_bit * (rinvC - ewald_shift), Vvdw[0], Vvdw[1], int_bit);
// #        endif
                                    FscalC[k] = qq[k] * int_bit * rinvC
                                                + qq[k] * pmecorrF(beta2 * r2) * beta3 * inv_r2 * inv_r2 * r2C * r2C * r2C;
#    elif defined EL_EWALD_TAB
                                    FscalC[k] = qq[k] * int_bit * rinvC 
                                                - qq[k] * interpolate_coulomb_force_r(nbparam, r2 * inv_r) * inv_r * r2 * inv_r2 * inv_r2 * r2C * r2C * r2C;
#    endif /* EL_EWALD_ANA/TAB */

#    ifdef CALC_ENERGIES
#        ifdef EL_CUTOFF
                                    Vcoul[k] = qq[k] * (int_bit * rinvC - c_rf);
#        endif
#        ifdef EL_RF
                                    Vcoul[k] = qq[k] * (int_bit * rinvC + 0.5f * two_k_rf * r2C - c_rf);
#        endif
#        ifdef EL_EWALD_ANY
                                    /* 1.0f - erff is faster than erfcf */
                                    Vcoul[k] = qq[k] * int_bit * (rinvC - ewald_shift) - qq[k] * inv_r * erff(r2 * inv_r * beta);
#        endif /* EL_EWALD_ANY */
#    endif
                                }
                                        // if ((c6AB[0] != 0 || c6AB[1] != 0 || c12AB[0]!=0 || c12AB[1]!=0 || qq[0]!=0 || qq[1]!=0) && (k == 1))
                                        // printf("interaction [%d-%d], r2=[%e], mask=%f, FscalC=[%e, %e], FscalV=[%e, %e]\n", ai, aj, r2, int_bit, FscalC[0] * rpm2, FscalC[1] * rpm2, FscalV[0] * rpm2, FscalV[1] * rpm2);
                                FscalC[k] *= rpinvC_nonzero;
                                FscalV[k] *= rpinvV_nonzero;
                            }
                        }
                        for (int k = 0; k < 2; k++)
                        {
#    ifdef CALC_ENERGIES
                            E_el += LFC[k] * Vcoul[k];
                            E_lj += LFV[k] * Vvdw[k];
#    endif
                            F_invr += LFC[k] * FscalC[k] * rpm2;
                            F_invr += LFV[k] * FscalV[k] * rpm2;
                        }
                    }
                    else
                    {
                        nnFEP++;

#    ifndef LJ_COMB
                        /* LJ 6*C6 and 12*C12 */
                        fetch_nbfp_c6_c12(c6, c12, nbparam, ntypes * typeiAB[0] + typejAB[0]);
#    else
#        ifdef LJ_COMB_GEOM
                        c6           = ljcp_iAB[0].x * ljcp_jAB[0].x;
                        c12          = ljcp_iAB[0].y * ljcp_jAB[0].y;
#        else
                        /* LJ 2^(1/6)*sigma and 12*epsilon */
                        sigma   = ljcp_iAB[0].x + ljcp_jAB[0].x;
                        epsilon = ljcp_iAB[0].y * ljcp_jAB[0].y;
#            if defined CALC_ENERGIES || defined LJ_FORCE_SWITCH || defined LJ_POT_SWITCH
                        convert_sigma_epsilon_to_c6_c12(sigma, epsilon, &c6, &c12);
#            endif
#        endif /* LJ_COMB_GEOM */
#    endif     /* LJ_COMB */

#    if !defined LJ_COMB_LB || defined CALC_ENERGIES
                        inv_r6 = inv_r2 * inv_r2 * inv_r2;
#        ifdef EXCLUSION_FORCES
                        /* We could mask inv_r2, but with Ewald
                         * masking both inv_r6 and F_invr is faster */
                        // inv_r6 *= int_bit;
#        endif /* EXCLUSION_FORCES */

                        F_invr = inv_r6 * (c12 * inv_r6 - c6) * inv_r2;
#        if defined CALC_ENERGIES || defined LJ_POT_SWITCH
                        E_lj_p = int_bit
                                 * (c12 * (inv_r6 * inv_r6 + nbparam.repulsion_shift.cpot) * c_oneTwelveth
                                    - c6 * (inv_r6 + nbparam.dispersion_shift.cpot) * c_oneSixth);
#        endif
#    else /* !LJ_COMB_LB || CALC_ENERGIES */
                        float sig_r  = sigma * inv_r;
                        float sig_r2 = sig_r * sig_r;
                        float sig_r6 = sig_r2 * sig_r2 * sig_r2;
#        ifdef EXCLUSION_FORCES
                        sig_r6 *= int_bit;
#        endif /* EXCLUSION_FORCES */

                        F_invr = epsilon * sig_r6 * (sig_r6 - 1.0f) * inv_r2;
#    endif     /* !LJ_COMB_LB || CALC_ENERGIES */

#    ifdef LJ_FORCE_SWITCH
#        ifdef CALC_ENERGIES
                        calculate_force_switch_F_E(nbparam, c6, c12, inv_r, r2, &F_invr, &E_lj_p);
#        else
                        calculate_force_switch_F(nbparam, c6, c12, inv_r, r2, &F_invr);
#        endif /* CALC_ENERGIES */
#    endif     /* LJ_FORCE_SWITCH */


#    ifdef LJ_EWALD
#        ifdef LJ_EWALD_COMB_GEOM
#            ifdef CALC_ENERGIES
                        calculate_lj_ewald_comb_geom_F_E(nbparam, typeiAB[0], typejAB[0], r2, inv_r2,
                                                                 lje_coeff2, lje_coeff6_6, int_bit,
                                                                 &F_invr, &E_lj_p);
#            else
                        calculate_lj_ewald_comb_geom_F(nbparam, typeiAB[0], typejAB[0], r2, inv_r2,
                                                               lje_coeff2, lje_coeff6_6, &F_invr);
#            endif /* CALC_ENERGIES */
#        elif defined LJ_EWALD_COMB_LB
                        calculate_lj_ewald_comb_LB_F_E(nbparam, typeiAB[0], typejAB[0], r2, inv_r2,
                                                       lje_coeff2, lje_coeff6_6,
#            ifdef CALC_ENERGIES
                                                       int_bit, &F_invr, &E_lj_p
#            else
                                                       0, &F_invr, nullptr
#            endif /* CALC_ENERGIES */
                        );
#        endif     /* LJ_EWALD_COMB_GEOM */
#    endif         /* LJ_EWALD */

#    ifdef LJ_POT_SWITCH
#        ifdef CALC_ENERGIES
                        calculate_potential_switch_F_E(nbparam, inv_r, r2, &F_invr, &E_lj_p);
#        else
                        calculate_potential_switch_F(nbparam, inv_r, r2, &F_invr, &E_lj_p);
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


#    ifdef EL_CUTOFF
#        ifdef EXCLUSION_FORCES
                        F_invr += qi * qj_f * int_bit * inv_r2 * inv_r;
#        else
                        F_invr += qi * qj_f * inv_r2 * inv_r;
#        endif
#    endif
#    ifdef EL_RF
                        F_invr += qi * qj_f * (int_bit * inv_r2 * inv_r - two_k_rf);
#    endif
#    if defined   EL_EWALD_ANA
                        F_invr += qi * qj_f
                                  * (inv_r2 * inv_r + pmecorrF(beta2 * r2) * beta3);
#    elif defined EL_EWALD_TAB
                        F_invr += qi * qj_f
                                  * (inv_r2
                                     - interpolate_coulomb_force_r(nbparam, r2 * inv_r))
                                  * inv_r;
#    endif /* EL_EWALD_ANA/TAB */

#    ifdef CALC_ENERGIES
#        ifdef EL_CUTOFF
                        E_el += qi * qj_f * (inv_r - c_rf);
#        endif
#        ifdef EL_RF
                        E_el += qi * qj_f * (inv_r + 0.5f * two_k_rf * r2 - c_rf);
#        endif
#        ifdef EL_EWALD_ANY
                                /* 1.0f - erff is faster than erfcf */
                        E_el += qi * qj_f
                                * (inv_r * (int_bit - erff(r2 * inv_r * beta)) - int_bit * ewald_shift);
#        endif /* EL_EWALD_ANY */
#    endif
                    }
                }
            }
#    ifdef EL_EWALD_ANY
            if (r2 < rcoulomb_sq)
            {
                v_lr = inv_r * erff(r2 * inv_r * beta);
                if (ai == aj) v_lr *= 0.5f;
#        if defined   EL_EWALD_ANA
                f_lr = -pmecorrF(beta2 * r2) * beta3;
#        elif defined EL_EWALD_TAB
                f_lr = interpolate_coulomb_force_r(nbparam, r2 * inv_r);
#        endif
                if (bFEP)
                {
                    for (int k = 0; k < 2; k++)
                    {
#        ifdef CALC_ENERGIES
                        E_el -= LFC[k] * qq[k] * v_lr;
#        endif
                        F_invr -= LFC[k] * qq[k] * f_lr;
                    }
                }
                else
                {
#        ifdef CALC_ENERGIES
                    E_el -= qq[0] * v_lr;
#        endif
                    F_invr -= qq[0] * f_lr;
                }
            }
#    endif

            f_ij = rv * F_invr;

            /* accumulate j forces in registers */
            fcj_buf -= f_ij;

            /* accumulate i forces in registers */
            fci_buf += f_ij;
            /* reduce j forces */
            // reduce_force_j_warp_shfl(fcj_buf, f, tidxi, aj, c_fullWarpMask);
            atomicAdd(&(f[aj]), fcj_buf);
        }
    //     }
    //     // avoid shared memory WAR hazards between loop iterations
    //     __syncwarp(c_fullWarpMask);
    // }

        /* skip central shifts when summing shift forces */
        // if (nb_sci.shift == CENTRAL)
        // {
        //     bCalcFshift = false;
        // }

        // float3 fshift_buf = 0.0f;

        /* reduce i forces */
        // for (i = 0; i < c_numClPerSupercl; i++)
        // {
        //     ai = (sci * c_numClPerSupercl + i) * c_clSize + tidxi;
        //     reduce_force_i_warp_shfl(fci_buf[i], f, &fshift_buf, bCalcFshift, tidxj, ai, c_fullWarpMask);
        // }

        /* add up local shift forces into global mem, tidxj indexes x,y,z */
        // if (bCalcFshift && (tidxj & 3) < 3)
        // {
        //     atomicAdd(&(atdat.fshift[nb_sci.shift].x) + (tidxj & 3), fshift_buf);
        // }
        atomicAdd(&(f[ai]), fci_buf);
        if (bCalcFshift)
        {
            atomicAdd(&(atdat.fshift[shift[n]]), fci_buf);
        }

#    ifdef CALC_ENERGIES
        /* reduce the energies over warps and store into global memory */
        reduce_energy_warp_shfl(E_lj, E_el, e_lj, e_el, tidx, c_fullWarpMask);
#    endif
        printf("nFEP=%d, nnFEP=%d\n", nFEP, nnFEP);
    }
}
#endif /* FUNCTION_DECLARATION_ONLY */

#undef NTHREAD_Z
#undef MIN_BLOCKS_PER_MP
#undef THREADS_PER_BLOCK

#undef EL_EWALD_ANY
#undef EXCLUSION_FORCES
#undef LJ_EWALD

#undef LJ_COMB
