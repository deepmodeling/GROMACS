/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021, by the GROMACS development team, led by
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

#ifndef GMX_NBNXM_PAIRLISTSETHELPERS_H
#define GMX_NBNXM_PAIRLISTSETHELPERS_H


// Resets current flags to 0 and adds more flags if needed.
void resizeAndZeroBufferFlags(std::vector<gmx_bitmask_t>* flags, const int numAtoms);

/* Estimates the average size of a full j-list for super/sub setup */
void get_nsubpair_target(const Nbnxm::GridSet&    gridSet,
                         gmx::InteractionLocality iloc,
                         real                     rlist,
                         int                      min_ci_balanced,
                         int*                     nsubpair_target,
                         float*                   nsubpair_tot_est);

/* Returns the i-zone range for pairlist construction for the give locality */
gmx::Range<int> getIZoneRange(const Nbnxm::GridSet::DomainSetup& domainSetup,
                              const gmx::InteractionLocality     locality);

/* Returns the j-zone range for pairlist construction for the give locality and i-zone */
gmx::Range<int> getJZoneRange(const gmx_domdec_zones_t*      ddZones,
                              const gmx::InteractionLocality locality,
                              const int                      iZone);

int get_ci_block_size(const Nbnxm::Grid& iGrid, bool haveMultipleDomains, int numLists);

/* Generates the part of pair-list nbl assigned to our thread */
template<typename T>
void nbnxn_make_pairlist_part(const Nbnxm::GridSet&        gridSet,
                              const Nbnxm::Grid&           iGrid,
                              const Nbnxm::Grid&           jGrid,
                              PairsearchWork*              work,
                              const nbnxn_atomdata_t*      nbat,
                              const gmx::ListOfLists<int>& exclusions,
                              real                         rlist,
                              PairlistType                 pairlistType,
                              int                          ci_block,
                              gmx_bool                     bFBufferFlag,
                              int                          nsubpair_max,
                              gmx_bool                     progBal,
                              float                        nsubpair_tot_est,
                              int                          th,
                              int                          nth,
                              T*                           nbl,
                              t_nblist*                    nbl_fep);

extern template void nbnxn_make_pairlist_part<NbnxnPairlistCpu>(const Nbnxm::GridSet&   gridSet,
                                                                const Nbnxm::Grid&      iGrid,
                                                                const Nbnxm::Grid&      jGrid,
                                                                PairsearchWork*         work,
                                                                const nbnxn_atomdata_t* nbat,
                                                                const gmx::ListOfLists<int>& exclusions,
                                                                real                         rlist,
                                                                PairlistType      pairlistType,
                                                                int               ci_block,
                                                                gmx_bool          bFBufferFlag,
                                                                int               nsubpair_max,
                                                                gmx_bool          progBal,
                                                                float             nsubpair_tot_est,
                                                                int               th,
                                                                int               nth,
                                                                NbnxnPairlistCpu* nbl,
                                                                t_nblist*         nbl_fep);

extern template void nbnxn_make_pairlist_part<NbnxnPairlistGpu>(const Nbnxm::GridSet&   gridSet,
                                                                const Nbnxm::Grid&      iGrid,
                                                                const Nbnxm::Grid&      jGrid,
                                                                PairsearchWork*         work,
                                                                const nbnxn_atomdata_t* nbat,
                                                                const gmx::ListOfLists<int>& exclusions,
                                                                real                         rlist,
                                                                PairlistType      pairlistType,
                                                                int               ci_block,
                                                                gmx_bool          bFBufferFlag,
                                                                int               nsubpair_max,
                                                                gmx_bool          progBal,
                                                                float             nsubpair_tot_est,
                                                                int               th,
                                                                int               nth,
                                                                NbnxnPairlistGpu* nbl,
                                                                t_nblist*         nbl_fep);

/* Combine pair lists *nbl generated on multiple threads nblc */
void combine_nblists(gmx::ArrayRef<const NbnxnPairlistGpu> nbls, NbnxnPairlistGpu* nblc);

//! Prepares CPU lists produced by the search for dynamic pruning
void prepareListsForDynamicPruning(gmx::ArrayRef<NbnxnPairlistCpu> lists);

void print_reduction_cost(gmx::ArrayRef<const gmx_bitmask_t> flags, int nout);

/* Debug list print function */
void print_nblist_sci_cj(FILE* fp, const NbnxnPairlistGpu& nbl);

/* Returns if the pairlists are so imbalanced that it is worth rebalancing. */
bool checkRebalanceSimpleLists(gmx::ArrayRef<const NbnxnPairlistCpu> lists);

/* Perform a count (linear) sort to sort the smaller lists to the end.
 * This avoids load imbalance on the GPU, as large lists will be
 * scheduled and executed first and the smaller lists later.
 * Load balancing between multi-processors only happens at the end
 * and there smaller lists lead to more effective load balancing.
 * The sorting is done on the cj4 count, not on the actual pair counts.
 * Not only does this make the sort faster, but it also results in
 * better load balancing than using a list sorted on exact load.
 * This function swaps the pointer in the pair list to avoid a copy operation.
 */
void sort_sci(NbnxnPairlistGpu* nbl);

/* Print statistics of a pair list, used for debug output */
void print_nblist_statistics(FILE* fp, const NbnxnPairlistCpu& nbl, const Nbnxm::GridSet& gridSet, real rl);

/* Print statistics of a pair lists, used for debug output */
void print_nblist_statistics(FILE* fp, const NbnxnPairlistGpu& nbl, const Nbnxm::GridSet& gridSet, real rl);

/* This routine re-balances the pairlists such that all are nearly equally
 * sized. Only whole i-entries are moved between lists. These are moved
 * between the ends of the lists, such that the buffer reduction cost should
 * not change significantly.
 * Note that all original reduction flags are currently kept. This can lead
 * to reduction of parts of the force buffer that could be avoided. But since
 * the original lists are quite balanced, this will only give minor overhead.
 */
void rebalanceSimpleLists(gmx::ArrayRef<const NbnxnPairlistCpu> srcSet,
                          gmx::ArrayRef<NbnxnPairlistCpu>       destSet,
                          gmx::ArrayRef<PairsearchWork>         searchWork);

void reduce_buffer_flags(gmx::ArrayRef<PairsearchWork> searchWork, int nsrc, gmx::ArrayRef<gmx_bitmask_t> dest);

void balance_fep_lists(gmx::ArrayRef<std::unique_ptr<t_nblist>> fepLists,
                       gmx::ArrayRef<PairsearchWork>            work);

/* Debug list print function */
void print_nblist_ci_cj(FILE* fp, const NbnxnPairlistCpu& nbl);

#endif // GMX_NBNXM_PAIRLISTSETHELPERS_H
