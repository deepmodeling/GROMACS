#include <cassert>
#include <cinttypes>
#include <csignal>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <memory>

#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/math/utilities.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/range.h"
#include "gromacs/utility/real.h"

#include "gromacs/sits/sits.h"

#ifndef SITS_CUDA_TYPES_H
#define SITS_CUDA_TYPES_H

/** \internal
 * \brief Main data structure for CUDA nonbonded force calculations.
 */

struct cu_sits_atdat_t
{
    int sits_calc_mode = 0;        // sits calculation mode: classical or simple
    int sits_enh_mode = PP_AND_PW; // sits enhancing region: solvate, intramolecular or intermolecular
    bool sits_enh_bias = false;    // whether to enhance the bias

    int natoms; /**< number of atoms                              */
    int nalloc;

    //! array of atom indices
    int* atomIndices;
    //! size of atom indices
    int atomIndicesSize;
    //! size of atom indices allocated in device buffer
    int atomIndicesSize_alloc;

    int nenergrp;
    int neg_2log;
    int* energrp;

    float3* d_enerd; // stores pp, pw, and ww energies in bonded and nonbonded SR kernels

    float3* d_force_tot;
    float3* d_force_pw;

    float3* d_force_tot_nbat;
    float3* d_force_pw_nbat;
}

struct cu_sits_param_t
{
public:
    // SITS energy records
    int record_count = 0; //记录次数
    int reset = 1; // record的时候，第一次和后面公式不一样，这个变量是拿来控制这个的

    // SITS ensemble definition
    int   record_interval = 1;   // interval of energy record
    int   update_interval = 100; // interval of $n_k$ update
    bool  constant_nk     = false;   // whether iteratively update n_k
    int   k_numbers;             // 
    int   k_nalloc;
    float beta0;                 // original temperature \beta

    //计算时，可以对fc_ball直接修正，+ fb_shift进行调节，
    float fb_shift;
    // energy record modifications: energy_record = energy_multiple * U + energy_shift;
    float energy_multiple;
    float energy_shift;

    // Derivations and physical quantities see:
    // \ref A selective integrated tempering method
    // \ref Self-adaptive enhanced sampling in the energy and trajectory spaces : Accelerated thermodynamics and kinetic calculations

    float* beta_k;
    float* nkExpBetakU;
    float* nk;
    float* sum_a;
    float* sum_b;
    float* factor;

    // Details of $n_k$ iteration see:
    // \ref An integrate-over-temperature approach for enhanced sampling

    // |   .cpp var    |  ylj .F90 var  |  Ref var
    // | ene_recorded  | vshift         | U  
    // | gf            | gf             | log( n_k * exp(-beta_k * U) )
    // | gfsum         | gfsum          | log( Sum_(k=1)^N ( log( n_k * exp(-beta_k * U) ) ) )
    // | log_weight    | rb             | log of the weighting function
    // | log_mk_inv    | ratio          | log(m_k^-1)
    // | log_norm_old  | normlold       | W(j-1)
    // | log_norm      | norml          | W(j)
    // | log_pk        | rbfb           | log(p_k)
    // | log_nk_inv    | pratio         | log(n_k^-1)
    // | log_nk        | fb             | log(n_k)

    float* ene_recorded;
    float* gf;
    float* gfsum;
    float* log_weight;
    float* log_mk_inv;
    float* log_norm_old;
    float* log_norm;
    float* log_pk;
    float* log_nk_inv;
    float* log_nk;
}

class GpuEventSynchronizer;

/** \internal
 * \brief Main data structure for CUDA nonbonded force calculations.
 */
struct gmx_sits_cuda_t
{
    //! CUDA device information
    const gmx_device_info_t* dev_info;
    //! atom data
    cu_sits_atdat_t* sits_atdat;

    //! parameters required for the sits calc.
    cu_sits_param_t* sits_param;
    //! staging area where fshift/energies get downloaded
    // nb_staging_t nbst;
    cudaStream_t* stream;
};

#endif /* SITS_CUDA_TYPES_H */