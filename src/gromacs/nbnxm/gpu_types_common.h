/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2017,2018,2019,2020,2021, by the GROMACS development team, led by
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
 * \brief Implements common internal types for different NBNXN GPU implementations
 *
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \ingroup module_nbnxm
 */

#ifndef GMX_MDLIB_NBNXN_GPU_COMMON_TYPES_H
#define GMX_MDLIB_NBNXN_GPU_COMMON_TYPES_H

#include "config.h"

#include "gromacs/gpu_utils/device_vectypes.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/utility/enumerationhelpers.h"

#include "nbnxm.h"
#include "pairlist.h"

#if GMX_GPU_OPENCL
#    include "gromacs/gpu_utils/gpuregiontimer_ocl.h"
#endif

#if GMX_GPU_CUDA
#    include "gromacs/gpu_utils/gpuregiontimer.cuh"
#endif

/** \internal
 * \brief Staging area for temporary data downloaded from the GPU.
 *
 *  The energies/shift forces get downloaded here first, before getting added
 *  to the CPU-side aggregate values.
 */
struct NBStagingData
{
    //! LJ energy
    float* e_lj = nullptr;
    //! electrostatic energy
    float* e_el = nullptr;
    //! shift forces
    Float3* fshift = nullptr;
};

/** \internal
 * \brief Nonbonded atom data - both inputs and outputs.
 */
struct NBAtomdata
{
    //! number of atoms
    int natoms;
    //! number of local atoms
    int natoms_local;
    //! allocation size for the atom data (xq, f)
    int nalloc;

    //! atom coordinates + charges, size natoms
    DeviceBuffer<Float4> xq;
    //! force output array, size natoms
    DeviceBuffer<Float3> f;

    //! LJ energy output, size 1
    DeviceBuffer<float> e_lj;
    //! Electrostatics energy input, size 1
    DeviceBuffer<float> e_el;

    //! shift forces
    DeviceBuffer<Float3> fshift;

    //! number of atom types
    int ntypes;
    //! atom type indices, size natoms
    DeviceBuffer<int> atom_types;
    //! sqrt(c6),sqrt(c12) size natoms
    DeviceBuffer<Float2> lj_comb;

    //! shifts
    DeviceBuffer<Float3> shift_vec;
    //! true if the shift vector has been uploaded
    bool bShiftVecUploaded;
};

/** \internal
 * \brief Parameters required for the GPU nonbonded calculations.
 */
struct NBParamGpu
{

    //! type of electrostatics
    enum Nbnxm::ElecType elecType;
    //! type of VdW impl.
    enum Nbnxm::VdwType vdwType;

    //! charge multiplication factor
    float epsfac;
    //! Reaction-field/plain cutoff electrostatics const.
    float c_rf;
    //! Reaction-field electrostatics constant
    float two_k_rf;
    //! Ewald/PME parameter
    float ewald_beta;
    //! Ewald/PME correction term substracted from the direct-space potential
    float sh_ewald;
    //! LJ-Ewald/PME correction term added to the correction potential
    float sh_lj_ewald;
    //! LJ-Ewald/PME coefficient
    float ewaldcoeff_lj;

    //! Coulomb cut-off squared
    float rcoulomb_sq;

    //! VdW cut-off squared
    float rvdw_sq;
    //! VdW switched cut-off
    float rvdw_switch;
    //! Full, outer pair-list cut-off squared
    float rlistOuter_sq;
    //! Inner, dynamic pruned pair-list cut-off squared
    float rlistInner_sq;
    //! True if we use dynamic pair-list pruning
    bool useDynamicPruning;

    //! VdW shift dispersion constants
    shift_consts_t dispersion_shift;
    //! VdW shift repulsion constants
    shift_consts_t repulsion_shift;
    //! VdW switch constants
    switch_consts_t vdw_switch;

    /* LJ non-bonded parameters - accessed through texture memory */
    //! nonbonded parameter table with C6/C12 pairs per atom type-pair, 2*ntype^2 elements
    DeviceBuffer<float> nbfp;
    //! texture object bound to nbfp
    DeviceTexture nbfp_texobj;
    //! nonbonded parameter table per atom type, 2*ntype elements
    DeviceBuffer<float> nbfp_comb;
    //! texture object bound to nbfp_comb
    DeviceTexture nbfp_comb_texobj;

    /* Ewald Coulomb force table data - accessed through texture memory */
    //! table scale/spacing
    float coulomb_tab_scale;
    //! pointer to the table in the device memory
    DeviceBuffer<float> coulomb_tab;
    //! texture object bound to coulomb_tab
    DeviceTexture coulomb_tab_texobj;
};

namespace Nbnxm
{

using gmx::AtomLocality;
using gmx::InteractionLocality;

/*! \internal
 * \brief GPU region timers used for timing GPU kernels and H2D/D2H transfers.
 *
 * The two-sized arrays hold the local and non-local values and should always
 * be indexed with eintLocal/eintNonlocal.
 */
struct gpu_timers_t
{
    /*! \internal
     * \brief Timers for local or non-local coordinate/force transfers
     */
    struct XFTransfers
    {
        //! timer for x/q H2D transfers (l/nl, every step)
        GpuRegionTimer nb_h2d;
        //! timer for f D2H transfer (l/nl, every step)
        GpuRegionTimer nb_d2h;
    };

    /*! \internal
     * \brief Timers for local or non-local interaction related operations
     */
    struct Interaction
    {
        //! timer for pair-list H2D transfers (l/nl, every PS step)
        GpuRegionTimer pl_h2d;
        //! true when a pair-list transfer has been done at this step
        bool didPairlistH2D = false;
        //! timer for non-bonded kernels (l/nl, every step)
        GpuRegionTimer nb_k;
        //! timer for the 1st pass list pruning kernel (l/nl, every PS step)
        GpuRegionTimer prune_k;
        //! true when we timed pruning and the timings need to be accounted for
        bool didPrune = false;
        //! timer for rolling pruning kernels (l/nl, frequency depends on chunk size)
        GpuRegionTimer rollingPrune_k;
        //! true when we timed rolling pruning (at the previous step) and the timings need to be accounted for
        bool didRollingPrune = false;
    };

    //! timer for atom data transfer (every PS step)
    GpuRegionTimer atdat;
    //! timers for coordinate/force transfers (every step)
    gmx::EnumerationArray<AtomLocality, XFTransfers> xf;
    //! timers for interaction related transfers
    gmx::EnumerationArray<InteractionLocality, Nbnxm::gpu_timers_t::Interaction> interaction;
};

/*! \internal
 * \brief GPU pair list structure */
struct gpu_plist
{
    //! number of atoms per cluster
    int na_c;

    //! size of sci, # of i clusters in the list
    int nsci;
    //! allocation size of sci
    int sci_nalloc;
    //! list of i-cluster ("super-clusters")
    DeviceBuffer<nbnxn_sci_t> sci;

    //! total # of 4*j clusters
    int ncj4;
    //! allocation size of cj4
    int cj4_nalloc;
    //! 4*j cluster list, contains j cluster number and index into the i cluster list
    DeviceBuffer<nbnxn_cj4_t> cj4;
    //! # of 4*j clusters * # of warps
    int nimask;
    //! allocation size of imask
    int imask_nalloc;
    //! imask for 2 warps for each 4*j cluster group
    DeviceBuffer<unsigned int> imask;
    //! atom interaction bits
    DeviceBuffer<nbnxn_excl_t> excl;
    //! count for excl
    int nexcl;
    //! allocation size of excl
    int excl_nalloc;

    /* parameter+variables for normal and rolling pruning */
    //! true after search, indictes that initial pruning with outer prunning is needed
    bool haveFreshList;
    //! the number of parts/steps over which one cyle of roling pruning takes places
    int rollingPruningNumParts;
    //! the next part to which the roling pruning needs to be applied
    int rollingPruningPart;
};

} // namespace Nbnxm

#endif
