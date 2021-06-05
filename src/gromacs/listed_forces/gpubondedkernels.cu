/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018,2019,2020,2021, by the GROMACS development team, led by
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
 * \brief Implements CUDA bonded functionality
 *
 * \author Jon Vincent <jvincent@nvidia.com>
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 * \author Berk Hess <hess@kth.se>
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \author Alan Gray <alang@nvidia.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_listed_forces
 */

#include "gmxpre.h"

#include <cassert>

#include <math_constants.h>

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/gpu_utils/vectype_ops.cuh"
#include "gromacs/listed_forces/gpubonded.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/pbc_aiuc_cuda.cuh"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/gmxassert.h"

#include "gpubonded_impl.h"

struct BondedFepParameters;

#if defined(_MSVC)
#    include <limits>
#endif

/*-------------------------------- CUDA kernels-------------------------------- */
/*------------------------------------------------------------------------------*/

#define CUDA_DEG2RAD_F (CUDART_PI_F / 180.0f)

/*---------------- BONDED CUDA kernels--------------*/

/* Harmonic */
__device__ __forceinline__ static void
           harmonic_gpu(const float kA, const float xA, const float x, float* V, float* F)
{
    constexpr float half = 0.5f;
    float           dx, dx2;

    dx  = x - xA;
    dx2 = dx * dx;

    *F = -kA * dx;
    *V = half * kA * dx2;
}

__device__ __forceinline__ static void
           harmonic_fep_gpu(const float kA, const float kB, const float xA, const float xB, const float x, const float lambda, float* V, float* F)
{
    constexpr float half = 0.5f;
    float           L1, kk, x0, dx, dx2;
    float           v, f, dvdlambda;

    L1 = 1.0 - lambda;
    kk = L1 * kA + lambda * kB;
    x0 = L1 * xA + lambda * xB;

    dx  = x - x0;
    dx2 = dx * dx;

    f         = -kk * dx;
    v         = half * kk * dx2;
    dvdlambda = half * (kB - kA) * dx2 + (xA - xB) * kk * dx;

    *F = f;
    *V = v;
}

template<bool calcVir, bool calcEner>
__device__ void bonds_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          const float4    gm_xq[],
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int3 bondData = *(int3*)(d_forceatoms + 3 * i);
        int  type     = bondData.x;
        int  ai       = bondData.y;
        int  aj       = bondData.z;

        /* dx = xi - xj, corrected for periodic boundary conditions. */
        float3 dx;
        int    ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dx);

        float dr2 = norm2(dx);
        float dr  = sqrt(dr2);

        float vbond;
        float fbond;
        harmonic_gpu(d_forceparams[type].harmonic.krA, d_forceparams[type].harmonic.rA, dr, &vbond, &fbond);

        if (calcEner)
        {
            *vtot_loc += vbond;
        }

        if (dr2 != 0.0f)
        {
            fbond *= rsqrtf(dr2);

            float3 fij = fbond * dx;
            atomicAdd(&gm_f[ai], fij);
            atomicAdd(&gm_f[aj], -fij);
            if (calcVir && ki != CENTRAL)
            {
                atomicAdd(&sm_fShiftLoc[ki], fij);
                atomicAdd(&sm_fShiftLoc[CENTRAL], -fij);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void bonds_fep_gpu(const int                 i,
                              float*                    vtot_loc,
                              const int                 numBonds,
                              const t_iatom             d_forceatoms[],
                              const t_iparams           d_forceparams[],
                              gmx::BondedFepParameters* d_fepparams,
                              const float4              gm_xq[],
                              float3                      gm_f[],
                              float3                      sm_fShiftLoc[],
                              const PbcAiuc             pbcAiuc)
{
    if (i < numBonds)
    {
        int3 bondData = *(int3*)(d_forceatoms + 3 * i);
        int  type     = bondData.x;
        int  ai       = bondData.y;
        int  aj       = bondData.z;

        /* dx = xi - xj, corrected for periodic boundary conditions. */
        float3 dx;
        int  ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dx);

        float dr2 = norm2(dx);
        float dr  = sqrt(dr2);

        float vbond;
        float fbond;
        harmonic_fep_gpu(d_forceparams[type].harmonic.krA, d_forceparams[type].harmonic.krB,
                         d_forceparams[type].harmonic.rA, d_forceparams[type].harmonic.rB, dr,
                         d_fepparams->lambda_q, &vbond, &fbond);

        if (calcEner)
        {
            *vtot_loc += vbond;
        }

        if (dr2 != 0.0f)
        {
            fbond *= rsqrtf(dr2);

            float3 fij = fbond * dx;
            atomicAdd(&gm_f[ai], fij);
            atomicAdd(&gm_f[aj], -fij);
            if (calcVir && ki != CENTRAL)
            {
                atomicAdd(&sm_fShiftLoc[ki], fij);
                atomicAdd(&sm_fShiftLoc[CENTRAL], -fij);
            }
        }
    }
}

template<bool returnShift>
__device__ __forceinline__ static float bond_angle_gpu(const float4   xi,
                                                       const float4   xj,
                                                       const float4   xk,
                                                       const PbcAiuc& pbcAiuc,
                                                       float3*        r_ij,
                                                       float3*        r_kj,
                                                       float*         costh,
                                                       int*           t1,
                                                       int*           t2)
/* Return value is the angle between the bonds i-j and j-k */
{
    *t1 = pbcDxAiuc<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xj, *r_kj);

    *costh   = cos_angle(*r_ij, *r_kj);
    float th = acosf(*costh);

    return th;
}

template<bool calcVir, bool calcEner>
__device__ void angles_gpu(const int       i,
                           float*          vtot_loc,
                           const int       numBonds,
                           const t_iatom   d_forceatoms[],
                           const t_iparams d_forceparams[],
                           const float4    gm_xq[],
                           float3          gm_f[],
                           float3          sm_fShiftLoc[],
                           const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int4 angleData = *(int4*)(d_forceatoms + 4 * i);
        int  type      = angleData.x;
        int  ai        = angleData.y;
        int  aj        = angleData.z;
        int  ak        = angleData.w;

        float3 r_ij;
        float3 r_kj;
        float  cos_theta;
        int    t1;
        int    t2;
        float  theta = bond_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij,
                                              &r_kj, &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        harmonic_gpu(d_forceparams[type].harmonic.krA,
                     d_forceparams[type].harmonic.rA * CUDA_DEG2RAD_F, theta, &va, &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0f)
        {
            float st    = dVdt * rsqrtf(1.0f - cos_theta2);
            float sth   = st * cos_theta;
            float nrij2 = norm2(r_ij);
            float nrkj2 = norm2(r_kj);

            float nrij_1 = rsqrtf(nrij2);
            float nrkj_1 = rsqrtf(nrkj2);

            float cik = st * nrij_1 * nrkj_1;
            float cii = sth * nrij_1 * nrij_1;
            float ckk = sth * nrkj_1 * nrkj_1;

            float3 f_i = cii * r_ij - cik * r_kj;
            float3 f_k = ckk * r_kj - cik * r_ij;
            float3 f_j = -f_i - f_k;

            atomicAdd(&gm_f[ai], f_i);
            atomicAdd(&gm_f[aj], f_j);
            atomicAdd(&gm_f[ak], f_k);

            if (calcVir)
            {
                atomicAdd(&sm_fShiftLoc[t1], f_i);
                atomicAdd(&sm_fShiftLoc[CENTRAL], f_j);
                atomicAdd(&sm_fShiftLoc[t2], f_k);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void angles_fep_gpu(const int                 i,
                               float*                    vtot_loc,
                               const int                 numBonds,
                               const t_iatom             d_forceatoms[],
                               const t_iparams           d_forceparams[],
                               gmx::BondedFepParameters* d_fepparams,
                               const float4              gm_xq[],
                               float3                      gm_f[],
                               float3                      sm_fShiftLoc[],
                               const PbcAiuc             pbcAiuc)
{
    if (i < numBonds)
    {
        int4 angleData = *(int4*)(d_forceatoms + 4 * i);
        int  type      = angleData.x;
        int  ai        = angleData.y;
        int  aj        = angleData.z;
        int  ak        = angleData.w;

        float3  r_ij;
        float3  r_kj;
        float cos_theta;
        int   t1;
        int   t2;
        float theta = bond_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj,
                                              &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        harmonic_fep_gpu(d_forceparams[type].harmonic.krA, d_forceparams[type].harmonic.krB,
                         d_forceparams[type].harmonic.rA * CUDA_DEG2RAD_F,
                         d_forceparams[type].harmonic.rB * CUDA_DEG2RAD_F, theta,
                         d_fepparams->lambda_q, &va, &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0f)
        {
            float st    = dVdt * rsqrtf(1.0f - cos_theta2);
            float sth   = st * cos_theta;
            float nrij2 = norm2(r_ij);
            float nrkj2 = norm2(r_kj);

            float nrij_1 = rsqrtf(nrij2);
            float nrkj_1 = rsqrtf(nrkj2);

            float cik = st * nrij_1 * nrkj_1;
            float cii = sth * nrij_1 * nrij_1;
            float ckk = sth * nrkj_1 * nrkj_1;

            float3 f_i;
            float3 f_k;
            float3 f_j;

            f_i = cii * r_ij - cik * r_kj;
            f_k = ckk * r_kj - cik * r_ij;
            f_j = -f_i - f_k;
            atomicAdd(&gm_f[ai], f_i);
            atomicAdd(&gm_f[aj], f_j);
            atomicAdd(&gm_f[ak], f_k);
            if (calcVir)
            {
                atomicAdd(&sm_fShiftLoc[t1], f_i);
                atomicAdd(&sm_fShiftLoc[CENTRAL], f_j);
                atomicAdd(&sm_fShiftLoc[t2], f_k);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void urey_bradley_gpu(const int       i,
                                 float*          vtot_loc,
                                 const int       numBonds,
                                 const t_iatom   d_forceatoms[],
                                 const t_iparams d_forceparams[],
                                 const float4    gm_xq[],
                                 float3          gm_f[],
                                 float3          sm_fShiftLoc[],
                                 const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int4 ubData = *(int4*)(d_forceatoms + 4 * i);
        int  type   = ubData.x;
        int  ai     = ubData.y;
        int  aj     = ubData.z;
        int  ak     = ubData.w;

        float th0A = d_forceparams[type].u_b.thetaA * CUDA_DEG2RAD_F;
        float kthA = d_forceparams[type].u_b.kthetaA;
        float r13A = d_forceparams[type].u_b.r13A;
        float kUBA = d_forceparams[type].u_b.kUBA;

        float3 r_ij;
        float3 r_kj;
        float  cos_theta;
        int    t1;
        int    t2;
        float  theta = bond_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij,
                                              &r_kj, &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        harmonic_gpu(kthA, th0A, theta, &va, &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        float3 r_ik;
        int    ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[ak], r_ik);

        float dr2 = norm2(r_ik);
        float dr  = dr2 * rsqrtf(dr2);

        float vbond;
        float fbond;
        harmonic_gpu(kUBA, r13A, dr, &vbond, &fbond);

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0f)
        {
            float st  = dVdt * rsqrtf(1.0f - cos_theta2);
            float sth = st * cos_theta;

            float nrkj2 = norm2(r_kj);
            float nrij2 = norm2(r_ij);

            float cik = st * rsqrtf(nrkj2 * nrij2);
            float cii = sth / nrij2;
            float ckk = sth / nrkj2;

            float3 f_i = cii * r_ij - cik * r_kj;
            float3 f_k = ckk * r_kj - cik * r_ij;
            float3 f_j = -f_i - f_k;

            atomicAdd(&gm_f[ai], f_i);
            atomicAdd(&gm_f[aj], f_j);
            atomicAdd(&gm_f[ak], f_k);

            if (calcVir)
            {
                atomicAdd(&sm_fShiftLoc[t1], f_i);
                atomicAdd(&sm_fShiftLoc[CENTRAL], f_j);
                atomicAdd(&sm_fShiftLoc[t2], f_k);
            }
        }

        /* Time for the bond calculations */
        if (dr2 != 0.0f)
        {
            if (calcEner)
            {
                *vtot_loc += vbond;
            }

            fbond *= rsqrtf(dr2);

            float3 fik = fbond * r_ik;
            atomicAdd(&gm_f[ai], fik);
            atomicAdd(&gm_f[ak], -fik);

            if (calcVir && ki != CENTRAL)
            {
                atomicAdd(&sm_fShiftLoc[ki], fik);
                atomicAdd(&sm_fShiftLoc[CENTRAL], -fik);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void urey_bradley_fep_gpu(const int                 i,
                                     float*                    vtot_loc,
                                     const int                 numBonds,
                                     const t_iatom             d_forceatoms[],
                                     const t_iparams           d_forceparams[],
                                     gmx::BondedFepParameters* d_fepparams,
                                     const float4              gm_xq[],
                                     float3                      gm_f[],
                                     float3                      sm_fShiftLoc[],
                                     const PbcAiuc             pbcAiuc)
{
    if (i < numBonds)
    {
        int4 ubData = *(int4*)(d_forceatoms + 4 * i);
        int  type   = ubData.x;
        int  ai     = ubData.y;
        int  aj     = ubData.z;
        int  ak     = ubData.w;

        float th0A = d_forceparams[type].u_b.thetaA * CUDA_DEG2RAD_F;
        float kthA = d_forceparams[type].u_b.kthetaA;
        float r13A = d_forceparams[type].u_b.r13A;
        float kUBA = d_forceparams[type].u_b.kUBA;

        float th0B = d_forceparams[type].u_b.thetaB * CUDA_DEG2RAD_F;
        float kthB = d_forceparams[type].u_b.kthetaB;
        float r13B = d_forceparams[type].u_b.r13B;
        float kUBB = d_forceparams[type].u_b.kUBB;

        float3  r_ij;
        float3  r_kj;
        float cos_theta;
        int   t1;
        int   t2;
        float theta = bond_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj,
                                              &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        // harmonic_gpu(kthA, th0A, theta, &va, &dVdt);
        harmonic_fep_gpu(kthA, kthB, th0A, th0B, theta, d_fepparams->lambda_q, &va, &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        float3 r_ik;
        int  ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[ak], r_ik);

        float dr2 = norm2(r_ik);
        float dr  = dr2 * rsqrtf(dr2);

        float vbond;
        float fbond;
        // harmonic_gpu(kUBA, r13A, dr, &vbond, &fbond);
        harmonic_fep_gpu(kUBA, kUBB, r13A, r13B, dr, d_fepparams->lambda_q, &vbond, &fbond);

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0f)
        {
            float st  = dVdt * rsqrtf(1.0f - cos_theta2);
            float sth = st * cos_theta;

            float nrkj2 = norm2(r_kj);
            float nrij2 = norm2(r_ij);

            float cik = st * rsqrtf(nrkj2 * nrij2);
            float cii = sth / nrij2;
            float ckk = sth / nrkj2;

            float3 f_i;
            float3 f_j;
            float3 f_k;
            f_i = cii * r_ij - cik * r_kj;
            f_k = ckk * r_kj - cik * r_ij;
            f_j = -f_i - f_k;
            atomicAdd(&gm_f[ai], f_i);
            atomicAdd(&gm_f[aj], f_j);
            atomicAdd(&gm_f[ak], f_k);
            if (calcVir)
            {
                atomicAdd(&sm_fShiftLoc[t1], f_i);
                atomicAdd(&sm_fShiftLoc[CENTRAL], f_j);
                atomicAdd(&sm_fShiftLoc[t2], f_k);
            }
        }

        /* Time for the bond calculations */
        if (dr2 != 0.0f)
        {
            if (calcEner)
            {
                *vtot_loc += vbond;
            }

            fbond *= rsqrtf(dr2);

            float3 fik = fbond * r_ik;
            atomicAdd(&gm_f[ai], fik);
            atomicAdd(&gm_f[ak], -fik);

            if (calcVir && ki != CENTRAL)
            {
                atomicAdd(&sm_fShiftLoc[ki], fik);
                atomicAdd(&sm_fShiftLoc[CENTRAL], -fik);
            }
        }
    }
}

template<bool returnShift, typename T>
__device__ __forceinline__ static float dih_angle_gpu(const T        xi,
                                                      const T        xj,
                                                      const T        xk,
                                                      const T        xl,
                                                      const PbcAiuc& pbcAiuc,
                                                      float3*        r_ij,
                                                      float3*        r_kj,
                                                      float3*        r_kl,
                                                      float3*        m,
                                                      float3*        n,
                                                      int*           t1,
                                                      int*           t2,
                                                      int*           t3)
{
    *t1 = pbcDxAiuc<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xj, *r_kj);
    *t3 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xl, *r_kl);

    *m         = cprod(*r_ij, *r_kj);
    *n         = cprod(*r_kj, *r_kl);
    float phi  = gmx_angle(*m, *n);
    float ipr  = iprod(*r_ij, *n);
    float sign = (ipr < 0.0f) ? -1.0f : 1.0f;
    phi        = sign * phi;

    return phi;
}


__device__ __forceinline__ static void
           dopdihs_gpu(const float cpA, const float phiA, const int mult, const float phi, float* v, float* f)
{
    float mdphi, sdphi;

    mdphi = mult * phi - phiA * CUDA_DEG2RAD_F;
    sdphi = sinf(mdphi);
    *v    = cpA * (1.0f + cosf(mdphi));
    *f    = -cpA * mult * sdphi;
}


__device__ __forceinline__ static void
           dopdihs_fep_gpu(const float cpA, const float cpB, const float phiA, const float phiB, const int mult, const float phi, const float lambda, float* v, float* f)
{
    float mdphiA, sdphiA, mdphiB, sdphiB;
    mdphiA = mult * phi - phiA * CUDA_DEG2RAD_F;
    mdphiB = mult * phi - phiB * CUDA_DEG2RAD_F;
    sdphiA = sinf(mdphiA);
    sdphiB = sinf(mdphiB);
    *v    = (1 - lambda) * cpA * (1.0f + cosf(mdphiA)) + lambda * cpB * (1.0f + cosf(mdphiB));
    *f    = -((1 - lambda) * cpA * mult * sdphiA + lambda * cpB * mult * sdphiB);
}

template<bool calcVir>
__device__ static void do_dih_fup_gpu(const int      i,
                                      const int      j,
                                      const int      k,
                                      const int      l,
                                      const float    ddphi,
                                      const float3   r_ij,
                                      const float3   r_kj,
                                      const float3   r_kl,
                                      const float3   m,
                                      const float3   n,
                                      float3         gm_f[],
                                      float3         sm_fShiftLoc[],
                                      const PbcAiuc& pbcAiuc,
                                      const float4   gm_xq[],
                                      const int      t1,
                                      const int      t2,
                                      const int gmx_unused t3)
{
    float iprm  = norm2(m);
    float iprn  = norm2(n);
    float nrkj2 = norm2(r_kj);
    float toler = nrkj2 * GMX_REAL_EPS;
    if ((iprm > toler) && (iprn > toler))
    {
        float  nrkj_1 = rsqrtf(nrkj2); // replacing std::invsqrt call
        float  nrkj_2 = nrkj_1 * nrkj_1;
        float  nrkj   = nrkj2 * nrkj_1;
        float  a      = -ddphi * nrkj / iprm;
        float3 f_i    = a * m;
        float  b      = ddphi * nrkj / iprn;
        float3 f_l    = b * n;
        float  p      = iprod(r_ij, r_kj);
        p *= nrkj_2;
        float q = iprod(r_kl, r_kj);
        q *= nrkj_2;
        float3 uvec = p * f_i;
        float3 vvec = q * f_l;
        float3 svec = uvec - vvec;
        float3 f_j  = f_i - svec;
        float3 f_k  = f_l + svec;

        atomicAdd(&gm_f[i], f_i);
        atomicAdd(&gm_f[j], -f_j);
        atomicAdd(&gm_f[k], -f_k);
        atomicAdd(&gm_f[l], f_l);

        if (calcVir)
        {
            float3 dx_jl;
            int    t3 = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[l], gm_xq[j], dx_jl);

            atomicAdd(&sm_fShiftLoc[t1], f_i);
            atomicAdd(&sm_fShiftLoc[CENTRAL], -f_j);
            atomicAdd(&sm_fShiftLoc[t2], -f_k);
            atomicAdd(&sm_fShiftLoc[t3], f_l);
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void pdihs_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          const float4    gm_xq[],
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc,
                                           &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        float vpd;
        float ddphi;
        dopdihs_gpu(d_forceparams[type].pdihs.cpA, d_forceparams[type].pdihs.phiA,
                    d_forceparams[type].pdihs.mult, phi, &vpd, &ddphi);

        if (calcEner)
        {
            *vtot_loc += vpd;
        }

        do_dih_fup_gpu<calcVir>(ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc,
                                pbcAiuc, gm_xq, t1, t2, t3);
    }
}

template<bool calcVir, bool calcEner>
__device__ void pdihs_fep_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          gmx::BondedFepParameters* d_fepparams,
                          const float4    gm_xq[],
                          float3            gm_f[],
                          float3            sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3  r_ij;
        float3  r_kj;
        float3  r_kl;
        float3  m;
        float3  n;
        int   t1;
        int   t2;
        int   t3;
        float phi = dih_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc,
                                           &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        float vpd;
        float ddphi;
        dopdihs_fep_gpu(d_forceparams[type].pdihs.cpA, d_forceparams[type].pdihs.cpB, d_forceparams[type].pdihs.phiA, d_forceparams[type].pdihs.phiB,
                    d_forceparams[type].pdihs.mult, phi, d_fepparams->lambda_q, &vpd, &ddphi);

        if (calcEner)
        {
            *vtot_loc += vpd;
        }

        do_dih_fup_gpu<calcVir>(ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc,
                                pbcAiuc, gm_xq, t1, t2, t3);
    }
}

template<bool calcVir, bool calcEner>
__device__ void rbdihs_gpu(const int       i,
                           float*          vtot_loc,
                           const int       numBonds,
                           const t_iatom   d_forceatoms[],
                           const t_iparams d_forceparams[],
                           const float4    gm_xq[],
                           float3          gm_f[],
                           float3          sm_fShiftLoc[],
                           const PbcAiuc   pbcAiuc)
{
    constexpr float c0 = 0.0f, c1 = 1.0f, c2 = 2.0f, c3 = 3.0f, c4 = 4.0f, c5 = 5.0f;

    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc,
                                           &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        /* Change to polymer convention */
        if (phi < c0)
        {
            phi += CUDART_PI_F;
        }
        else
        {
            phi -= CUDART_PI_F;
        }
        float cos_phi = cosf(phi);
        /* Beware of accuracy loss, cannot use 1-sqrt(cos^2) ! */
        float sin_phi = sinf(phi);

        float parm[NR_RBDIHS];
        for (int j = 0; j < NR_RBDIHS; j++)
        {
            parm[j] = d_forceparams[type].rbdihs.rbcA[j];
        }
        /* Calculate cosine powers */
        /* Calculate the energy */
        /* Calculate the derivative */
        float v      = parm[0];
        float ddphi  = c0;
        float cosfac = c1;

        float rbp = parm[1];
        ddphi += rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[2];
        ddphi += c2 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[3];
        ddphi += c3 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[4];
        ddphi += c4 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[5];
        ddphi += c5 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }

        ddphi = -ddphi * sin_phi;

        do_dih_fup_gpu<calcVir>(ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc,
                                pbcAiuc, gm_xq, t1, t2, t3);
        if (calcEner)
        {
            *vtot_loc += v;
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void rbdihs_fep_gpu(const int       i,
                           float*          vtot_loc,
                           const int       numBonds,
                           const t_iatom   d_forceatoms[],
                           const t_iparams d_forceparams[],
                           gmx::BondedFepParameters* d_fepparams,
                           const float4    gm_xq[],
                           float3            gm_f[],
                           float3            sm_fShiftLoc[],
                           const PbcAiuc   pbcAiuc)
{
    constexpr float c0 = 0.0f, c1 = 1.0f, c2 = 2.0f, c3 = 3.0f, c4 = 4.0f, c5 = 5.0f;

    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3  r_ij;
        float3  r_kj;
        float3  r_kl;
        float3  m;
        float3  n;
        int   t1;
        int   t2;
        int   t3;
        float phi = dih_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc,
                                           &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);
        float lambda = d_fepparams->lambda_q;

        /* Change to polymer convention */
        if (phi < c0)
        {
            phi += CUDART_PI_F;
        }
        else
        {
            phi -= CUDART_PI_F;
        }
        float cos_phi = cosf(phi);
        /* Beware of accuracy loss, cannot use 1-sqrt(cos^2) ! */
        float sin_phi = sinf(phi);

        float parm[NR_RBDIHS];
        for (int j = 0; j < NR_RBDIHS; j++)
        {
            parm[j] = (1 - lambda) * d_forceparams[type].rbdihs.rbcA[j] + lambda * d_forceparams[type].rbdihs.rbcB[j];
        }
        /* Calculate cosine powers */
        /* Calculate the energy */
        /* Calculate the derivative */
        float v      = parm[0];
        float ddphi  = c0;
        float cosfac = c1;

        float rbp = parm[1];
        ddphi += rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[2];
        ddphi += c2 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[3];
        ddphi += c3 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[4];
        ddphi += c4 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[5];
        ddphi += c5 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }

        ddphi = -ddphi * sin_phi;

        do_dih_fup_gpu<calcVir>(ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc,
                                pbcAiuc, gm_xq, t1, t2, t3);
        if (calcEner)
        {
            *vtot_loc += v;
        }
    }
}

__device__ __forceinline__ static void make_dp_periodic_gpu(float* dp)
{
    /* dp cannot be outside (-pi,pi) */
    if (*dp >= CUDART_PI_F)
    {
        *dp -= 2.0f * CUDART_PI_F;
    }
    else if (*dp < -CUDART_PI_F)
    {
        *dp += 2.0f * CUDART_PI_F;
    }
}

template<bool calcVir, bool calcEner>
__device__ void idihs_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          const float4    gm_xq[],
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc,
                                           &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        /* phi can jump if phi0 is close to Pi/-Pi, which will cause huge
         * force changes if we just apply a normal harmonic.
         * Instead, we first calculate phi-phi0 and take it modulo (-Pi,Pi).
         * This means we will never have the periodicity problem, unless
         * the dihedral is Pi away from phiO, which is very unlikely due to
         * the potential.
         */
        float kA = d_forceparams[type].harmonic.krA;
        float pA = d_forceparams[type].harmonic.rA;

        float phi0 = pA * CUDA_DEG2RAD_F;

        float dp = phi - phi0;

        make_dp_periodic_gpu(&dp);

        float ddphi = -kA * dp;

        do_dih_fup_gpu<calcVir>(ai, aj, ak, al, -ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc,
                                pbcAiuc, gm_xq, t1, t2, t3);

        if (calcEner)
        {
            *vtot_loc += -0.5f * ddphi * dp;
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void idihs_fep_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          gmx::BondedFepParameters* d_fepparams,
                          const float4    gm_xq[],
                          float3            gm_f[],
                          float3            sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3  r_ij;
        float3  r_kj;
        float3  r_kl;
        float3  m;
        float3  n;
        int   t1;
        int   t2;
        int   t3;
        float phi = dih_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc,
                                           &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        /* phi can jump if phi0 is close to Pi/-Pi, which will cause huge
         * force changes if we just apply a normal harmonic.
         * Instead, we first calculate phi-phi0 and take it modulo (-Pi,Pi).
         * This means we will never have the periodicity problem, unless
         * the dihedral is Pi away from phiO, which is very unlikely due to
         * the potential.
         */
        float kA = d_forceparams[type].harmonic.krA;
        float pA = d_forceparams[type].harmonic.rA;
        float kB = d_forceparams[type].harmonic.krB;
        float pB = d_forceparams[type].harmonic.rB;

        float phi0A = pA * CUDA_DEG2RAD_F;
        float phi0B = pB * CUDA_DEG2RAD_F;

        float dpA = phi - phi0A;
        float dpB = phi - phi0B;

        make_dp_periodic_gpu(&dpA);
        make_dp_periodic_gpu(&dpB);

        float lambda = d_fepparams->lambda_q;

        float ddphi = -((1 - lambda) * kA * dpA + lambda * kB * dpB);

        do_dih_fup_gpu<calcVir>(ai, aj, ak, al, -ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc,
                                pbcAiuc, gm_xq, t1, t2, t3);

        if (calcEner)
        {
            *vtot_loc += -0.5f * ddphi * dpA;
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void pairs_gpu(const int       i,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams iparams[],
                          const float4    gm_xq[],
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc,
                          const float     scale_factor,
                          float*          vtotVdw_loc,
                          float*          vtotElec_loc)
{
    if (i < numBonds)
    {
        // TODO this should be made into a separate type, the GPU and CPU sizes should be compared
        int3 pairData = *(int3*)(d_forceatoms + 3 * i);
        int  type     = pairData.x;
        int  ai       = pairData.y;
        int  aj       = pairData.z;

        float qq  = gm_xq[ai].w * gm_xq[aj].w;
        float c6  = iparams[type].lj14.c6A;
        float c12 = iparams[type].lj14.c12A;

        /* Do we need to apply full periodic boundary conditions? */
        float3 dr;
        int    fshift_index = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dr);

        float r2    = norm2(dr);
        float rinv  = rsqrtf(r2);
        float rinv2 = rinv * rinv;
        float rinv6 = rinv2 * rinv2 * rinv2;

        /* Calculate the Coulomb force * r */
        float velec = scale_factor * qq * rinv;

        /* Calculate the LJ force * r and add it to the Coulomb part */
        float fr = (12.0f * c12 * rinv6 - 6.0f * c6) * rinv6 + velec;

        float  finvr = fr * rinv2;
        float3 f     = finvr * dr;

        /* Add the forces */
        atomicAdd(&gm_f[ai], f);
        atomicAdd(&gm_f[aj], -f);
        if (calcVir && fshift_index != CENTRAL)
        {
            atomicAdd(&sm_fShiftLoc[fshift_index], f);
            atomicAdd(&sm_fShiftLoc[CENTRAL], -f);
        }

        if (calcEner)
        {
            *vtotVdw_loc += (c12 * rinv6 - c6) * rinv6;
            *vtotElec_loc += velec;
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void pairs_fep_gpu(const int       i,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams iparams[],
                          gmx::BondedFepParameters* d_fepparams,
                          const float4    gm_xq[],
                          const float     gm_qA[],
                          const float     gm_qB[],
                          float3            gm_f[],
                          float3            sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc,
                          const float     scale_factor,
                          float*          vtotVdw_loc,
                          float*          vtotElec_loc)
{
    const bool bFEP = d_fepparams->bFEP;
    bool       bFEPpair = 0;
    const float alpha_coul = d_fepparams->alpha_coul;
    const float alpha_vdw  = d_fepparams->alpha_vdw;
    float alpha_coul_eff   = alpha_coul;
    float alpha_vdw_eff    = alpha_vdw;
    const bool useSoftCore = (alpha_vdw != 0.0);
    const bool useScBetaNO = (alpha_coul == 0.0);
    const float sigma6_def = d_fepparams->sc_sigma6;
    const float sigma6_min = d_fepparams->sc_sigma6_min;
    const float lambda_q   = d_fepparams->lambda_q;
    const float _lambda_q  = 1 - lambda_q;
    const float lambda_v   = d_fepparams->lambda_v;
    const float _lambda_v  = 1 - lambda_v;

    const float lfac_coul[2] = {lambda_q, _lambda_q};
    const float lfac_vdw[2]  = {lambda_v, _lambda_v};
    const float LFC[2]       = {_lambda_q, lambda_q};
    const float LFV[2]       = {_lambda_v, lambda_v};

    float FscalC[2], FscalV[2], Vcoul[2], Vvdw[2];
    float rinvC, rinvV, r2C, r2V, rpinvC, rpinvV;

    if (i < numBonds)
    {
        int3 pairData = *(int3*)(d_forceatoms + 3 * i);
        int  type     = pairData.x;
        int  ai       = pairData.y;
        int  aj       = pairData.z;

        float qq[2]  = {gm_qA[ai] * gm_qA[aj], gm_qB[ai] * gm_qB[aj]};
        float c6AB[2]  = {iparams[type].lj14.c6A, iparams[type].lj14.c6B};
        float c12AB[2] = {iparams[type].lj14.c12A, iparams[type].lj14.c12B};
        float sigma6[2];
        float velec = 0;
        float vlj   = 0;
        float finvr = 0;

        if (qq[0] == qq[1] && c6AB[0] == c6AB[1] && c12AB[0] == c12AB[1]) bFEPpair = 0;
        else bFEPpair = 1;

        /* Do we need to apply full periodic boundary conditions? */
        float3 dr;
        int  fshift_index = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dr);

        float r2    = norm2(dr);
        float rpm2  = r2 * r2;
        float rp    = rpm2 * r2;
        float rinv  = rsqrtf(r2);
        float rinv2 = rinv * rinv;
        float rinv6 = rinv2 * rinv2 * rinv2;

        if (bFEPpair)
        {
            for (int k = 0; k < 2; k++)
            {
                FscalC[k] = 0;
                FscalV[k] = 0;
                if (calcEner)
                {
                    Vcoul[k] = 0;
                    Vvdw[k]  = 0;
                }
                if ((qq[k] != 0) || (c6AB[k] != 0) || (c12AB[k] != 0))
                {
                    if ((c12AB[0] == 0 || c12AB[1] == 0) && (useSoftCore))
                    {
                        if (c6AB[k] == 0)
                            sigma6[k] = 0;
                        else
                            sigma6[k] = c12AB[k] / c6AB[k];
                        alpha_vdw_eff  = alpha_vdw;
                        alpha_coul_eff = (useScBetaNO ? alpha_vdw_eff : alpha_coul);
                        if (sigma6[k] == 0)
                            sigma6[k] = sigma6_def;
                        if (sigma6[k] < sigma6_min)
                            sigma6[k] = sigma6_min;

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
                            rpinvC = 1.0f / (alpha_coul_eff * lfac_coul[k] * sigma6[k] + rp);
                        }
                        r2C   = rcbrt(rpinvC);
                        rinvC = rsqrt(r2C);

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
                            rinvV  = rinvC;
                        }
                    }
                    else
                    {
                        rpinvC = rinv6;
                        rinvC  = rinv;
                        rpinvV = rinv6;
                        rinvV  = rinv;
                    }
                    // #        endif /* EXCLUSION_FORCES */
                    // #    if !defined LJ_COMB_LB || defined CALC_ENERGIES

                    if (c6AB[k] != 0 || c12AB[k] != 0)
                    {
                        float Vvdw6  = c6AB[k] * rpinvV;
                        float Vvdw12 = c12AB[k] * rpinvV * rpinvV;
                        FscalV[k]    = 12.0f * Vvdw12 - 6.0f * Vvdw6;
                        if (calcEner)
                        {
                            Vvdw[k] = Vvdw12 - Vvdw6;
                        }
                    }

                    if (qq[k] != 0)
                    {
                        FscalC[k] = scale_factor * qq[k] * rinvC;
                        Vcoul[k]  = FscalC[k];
                    }
                    // if ((c6AB[0] != 0 || c6AB[1] != 0 || c12AB[0]!=0 || c12AB[1]!=0 || qq[0]!=0 || qq[1]!=0) && (k == 1))
                    // printf("interaction [%d-%d], r2=[%e], mask=%f, FscalC=[%e, %e], FscalV=[%e, %e]\n", ai, aj, r2, int_bit, FscalC[0] * rpm2, FscalC[1] * rpm2, FscalV[0] * rpm2, FscalV[1] * rpm2);
                    FscalC[k] *= rpinvC;
                    FscalV[k] *= rpinvV;
                }
            }
            for (int k = 0; k < 2; k++)
            {
                if (calcEner)
                {
                    velec += LFC[k] * Vcoul[k];
                    vlj += LFV[k] * Vvdw[k];
                }
                finvr += LFC[k] * FscalC[k] * rpm2;
                finvr += LFV[k] * FscalV[k] * rpm2;
            }
        }
        else
        {
            /* Calculate the Coulomb force * r */
            velec = scale_factor * qq[0] * rinv;
            vlj   = (c12AB[0] * rinv6 - c6AB[0]) * rinv6;

            /* Calculate the LJ force * r and add it to the Coulomb part */
            float fr = (12.0f * c12AB[0] * rinv6 - 6.0f * c6AB[0]) * rinv6 + velec;
            finvr    = fr * rinv2;
        }

        float3 f = finvr * dr;

        /* Add the forces */
        atomicAdd(&gm_f[ai], f);
        atomicAdd(&gm_f[aj], -f);
        if (calcVir && fshift_index != CENTRAL)
        {
            atomicAdd(&sm_fShiftLoc[fshift_index], f);
            atomicAdd(&sm_fShiftLoc[CENTRAL], -f);
        }

        if (calcEner)
        {
            *vtotVdw_loc += vlj;
            *vtotElec_loc += velec;
        }
    }
}

namespace gmx
{

template<bool calcVir, bool calcEner>
__global__ void exec_kernel_gpu(BondedCudaKernelParameters kernelParams)
{
    assert(blockDim.y == 1 && blockDim.z == 1);
    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    float     vtot_loc     = 0;
    float     vtotVdw_loc  = 0;
    float     vtotElec_loc = 0;

    extern __shared__ char sm_dynamicShmem[];
    char*                  sm_nextSlotPtr = sm_dynamicShmem;
    float3*                sm_fShiftLoc   = (float3*)sm_nextSlotPtr;
    sm_nextSlotPtr += SHIFTS * sizeof(float3);

    if (calcVir)
    {
        if (threadIdx.x < SHIFTS)
        {
            sm_fShiftLoc[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
        }
        __syncthreads();
    }

    int  fType;
    bool threadComputedPotential = false;
#pragma unroll
    for (int j = 0; j < numFTypesOnGpu; j++)
    {
        if (tid >= kernelParams.fTypeRangeStart[j] && tid <= kernelParams.fTypeRangeEnd[j])
        {
            const int      numBonds = kernelParams.numFTypeBonds[j];
            int            fTypeTid = tid - kernelParams.fTypeRangeStart[j];
            const t_iatom* iatoms   = kernelParams.d_iatoms[j];
            fType                   = kernelParams.fTypesOnGpu[j];
            if (calcEner)
            {
                threadComputedPotential = true;
            }

            switch (fType)
            {
                case F_BONDS:
                    // bonds_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                    //                              kernelParams.d_forceParams, kernelParams.d_xq,
                    //                              kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    bonds_fep_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                 kernelParams.d_forceParams, kernelParams.d_fepParams, kernelParams.d_xq,
                                                 kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_ANGLES:
                    // angles_gpu<calcVir, calcEner>(
                    //         fTypeTid, &vtot_loc, numBonds, iatoms, kernelParams.d_forceParams,
                    //         kernelParams.d_xq, kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    angles_fep_gpu<calcVir, calcEner>(
                            fTypeTid, &vtot_loc, numBonds, iatoms, kernelParams.d_forceParams, kernelParams.d_fepParams,
                            kernelParams.d_xq, kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_UREY_BRADLEY:
                    // urey_bradley_gpu<calcVir, calcEner>(
                    //         fTypeTid, &vtot_loc, numBonds, iatoms, kernelParams.d_forceParams,
                    //         kernelParams.d_xq, kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    urey_bradley_fep_gpu<calcVir, calcEner>(
                            fTypeTid, &vtot_loc, numBonds, iatoms, kernelParams.d_forceParams, kernelParams.d_fepParams,
                            kernelParams.d_xq, kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_PDIHS:
                case F_PIDIHS:
                    // pdihs_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                    //                              kernelParams.d_forceParams, kernelParams.d_xq,
                    //                              kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    pdihs_fep_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                 kernelParams.d_forceParams, kernelParams.d_fepParams, kernelParams.d_xq,
                                                 kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_RBDIHS:
                    // rbdihs_gpu<calcVir, calcEner>(
                    //         fTypeTid, &vtot_loc, numBonds, iatoms, kernelParams.d_forceParams,
                    //         kernelParams.d_xq, kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    rbdihs_fep_gpu<calcVir, calcEner>(
                            fTypeTid, &vtot_loc, numBonds, iatoms, kernelParams.d_forceParams, kernelParams.d_fepParams,
                            kernelParams.d_xq, kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_IDIHS:
                    // idihs_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                    //                              kernelParams.d_forceParams, kernelParams.d_xq,
                    //                              kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    idihs_fep_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                 kernelParams.d_forceParams, kernelParams.d_fepParams, kernelParams.d_xq,
                                                 kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_LJ14:
                    // pairs_gpu<calcVir, calcEner>(fTypeTid, numBonds, iatoms, kernelParams.d_forceParams,
                    //                              kernelParams.d_xq, kernelParams.d_f, sm_fShiftLoc,
                    //                              kernelParams.pbcAiuc, kernelParams.electrostaticsScaleFactor,
                    //                              &vtotVdw_loc, &vtotElec_loc);
                    pairs_fep_gpu<calcVir, calcEner>(fTypeTid, numBonds, iatoms, kernelParams.d_forceParams, kernelParams.d_fepParams,
                                                 kernelParams.d_xq, kernelParams.d_qA, kernelParams.d_qB, kernelParams.d_f, sm_fShiftLoc,
                                                 kernelParams.pbcAiuc, kernelParams.electrostaticsScaleFactor,
                                                 &vtotVdw_loc, &vtotElec_loc);
                    break;
            }
            break;
        }
    }

    if (threadComputedPotential)
    {
        float* vtotVdw  = kernelParams.d_vTot + F_LJ14;
        float* vtotElec = kernelParams.d_vTot + F_COUL14;

        // Stage atomic accumulation through shared memory:
        // each warp will accumulate its own partial sum
        // and then a single thread per warp will accumulate this to the global sum

        int numWarps = blockDim.x / warpSize;
        int warpId   = threadIdx.x / warpSize;

        // Shared memory variables to hold block-local partial sum
        float* sm_vTot = (float*)sm_nextSlotPtr;
        sm_nextSlotPtr += numWarps * sizeof(float);
        float* sm_vTotVdw = (float*)sm_nextSlotPtr;
        sm_nextSlotPtr += numWarps * sizeof(float);
        float* sm_vTotElec = (float*)sm_nextSlotPtr;

        if (threadIdx.x % warpSize == 0)
        {
            // One thread per warp initializes to zero
            sm_vTot[warpId]     = 0.;
            sm_vTotVdw[warpId]  = 0.;
            sm_vTotElec[warpId] = 0.;
        }
        __syncwarp(); // All threads in warp must wait for initialization

        // Perform warp-local accumulation in shared memory
        atomicAdd(sm_vTot + warpId, vtot_loc);
        atomicAdd(sm_vTotVdw + warpId, vtotVdw_loc);
        atomicAdd(sm_vTotElec + warpId, vtotElec_loc);

        __syncwarp(); // Ensure all threads in warp have completed
        if (threadIdx.x % warpSize == 0)
        { // One thread per warp accumulates partial sum into global sum
            atomicAdd(kernelParams.d_vTot + fType, sm_vTot[warpId]);
            atomicAdd(vtotVdw, sm_vTotVdw[warpId]);
            atomicAdd(vtotElec, sm_vTotElec[warpId]);
        }
    }
    /* Accumulate shift vectors from shared memory to global memory on the first SHIFTS threads of the block. */
    if (calcVir)
    {
        __syncthreads();
        if (threadIdx.x < SHIFTS)
        {
            atomicAdd(kernelParams.d_fShift[threadIdx.x], sm_fShiftLoc[threadIdx.x]);
        }
    }
}


/*-------------------------------- End CUDA kernels-----------------------------*/


template<bool calcVir, bool calcEner>
void GpuBonded::Impl::launchKernel()
{
    GMX_ASSERT(haveInteractions_,
               "Cannot launch bonded GPU kernels unless bonded GPU work was scheduled");

    wallcycle_start_nocount(wcycle_, ewcLAUNCH_GPU);
    wallcycle_sub_start(wcycle_, ewcsLAUNCH_GPU_BONDED);

    int fTypeRangeEnd = kernelParams_.fTypeRangeEnd[numFTypesOnGpu - 1];

    if (fTypeRangeEnd < 0)
    {
        return;
    }

    auto kernelPtr = exec_kernel_gpu<calcVir, calcEner>;

    const auto kernelArgs = prepareGpuKernelArguments(kernelPtr, kernelLaunchConfig_, &kernelParams_);

    launchGpuKernel(kernelPtr, kernelLaunchConfig_, deviceStream_, nullptr,
                    "exec_kernel_gpu<calcVir, calcEner>", kernelArgs);

    wallcycle_sub_stop(wcycle_, ewcsLAUNCH_GPU_BONDED);
    wallcycle_stop(wcycle_, ewcLAUNCH_GPU);
}

void GpuBonded::launchKernel(const gmx::StepWorkload& stepWork)
{
    if (stepWork.computeEnergy)
    {
        // When we need the energy, we also need the virial
        impl_->launchKernel<true, true>();
    }
    else if (stepWork.computeVirial)
    {
        impl_->launchKernel<true, false>();
    }
    else
    {
        impl_->launchKernel<false, false>();
    }
}

} // namespace gmx
