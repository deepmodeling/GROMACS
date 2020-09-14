/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2017,2018,2019,2020, by the GROMACS development team, led by
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
 * \brief
 * Implements bias sharing checking functionality.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_awh
 */

#include "gmxpre.h"

#include "biassharing.h"

#include "config.h"

#include <algorithm>
#include <set>
#include <vector>

#include "gromacs/gmxlib/network.h"
#include "gromacs/mdrunutility/multisim.h"
#include "gromacs/mdtypes/awh_params.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

namespace
{

std::multiset<int> getGlobalShareIndices(ArrayRef<const int> localShareIndices, MPI_Comm simulationMastersComm)
{
#if GMX_MPI
    int numSimulations;
    MPI_Comm_size(simulationMastersComm, &numSimulations);
    int ourRank;
    MPI_Comm_rank(simulationMastersComm, &ourRank);
    std::vector<int> biasCountsIn(numSimulations, 0);
    std::vector<int> biasCounts(numSimulations, 0);
    biasCountsIn[ourRank] = localShareIndices.size();
    MPI_Allreduce(biasCountsIn.data(), biasCounts.data(), numSimulations, MPI_INT, MPI_SUM,
                  simulationMastersComm);
    // Now we need to gather the share indices to all (master) ranks.
    // We could use MPI_Allgatherv, but thread-MPI does not support that and using
    // MPI_Allreduce produces simpler code, so we use that.
    int totNumBiases = 0;
    int ourOffset    = 0;
    for (int rank = 0; rank < numSimulations; rank++)
    {
        if (rank == ourRank)
        {
            ourOffset = totNumBiases;
        }
        totNumBiases += biasCounts[rank];
    }
    // Fill a buffer with zeros and our part of sharing indices
    std::vector<int> shareIndicesAllIn(totNumBiases, 0);
    std::copy(localShareIndices.begin(), localShareIndices.end(), shareIndicesAllIn.begin() + ourOffset);
    // Gather all sharing indices to all (master) ranks
    std::vector<int> shareIndicesAll(totNumBiases);
    MPI_Allreduce(shareIndicesAllIn.data(), shareIndicesAll.data(), totNumBiases, MPI_INT, MPI_SUM,
                  simulationMastersComm);
#else
    GMX_UNUSED_VALUE(simulationMastersComm);

    ArrayRef<const int> shareIndicesAll = localShareIndices;
#endif // GMX_MPI

    std::multiset<int> shareIndicesSet;
    for (int shareIndex : shareIndicesAll)
    {
        if (shareIndex > 0)
        {
            shareIndicesSet.insert(shareIndex);
        }
    }

    return shareIndicesSet;
}

} // namespace

BiasSharing::BiasSharing(const AwhParams& awhParams, const t_commrec& commRecord, MPI_Comm simulationMastersComm) :
    commRecord_(commRecord)
{
    if (MASTER(&commRecord))
    {
        std::vector<int> localShareIndices;
        int              shareGroupPrev = 0;
        for (int k = 0; k < awhParams.numBias; k++)
        {
            const int shareGroup = awhParams.awhBiasParams[k].shareGroup;
            GMX_RELEASE_ASSERT(shareGroup >= 0, "Bias share group values should be >= 0");
            localShareIndices.push_back(shareGroup);
            if (shareGroup > 0 && shareGroup <= shareGroupPrev)
            {
                GMX_THROW(
                        InvalidInputError("AWH biases that are shared should use increasing "
                                          "share-group values"));
                shareGroupPrev = shareGroup;
            }
        }
        std::multiset<int> globalShareIndices =
                getGlobalShareIndices(localShareIndices, simulationMastersComm);

        int numSimulations = 1;
#if GMX_MPI
        MPI_Comm_size(simulationMastersComm, &numSimulations);
        int myRank;
        MPI_Comm_rank(simulationMastersComm, &myRank);
#endif // GMX_MPI

        numSharingSimulations_.resize(awhParams.numBias, 1);
        sharingSimulationIndices_.resize(awhParams.numBias, 0);
        multiSimCommPerBias_.resize(awhParams.numBias, MPI_COMM_NULL);

        for (int shareIndex : globalShareIndices)
        {
            if (globalShareIndices.count(shareIndex) > 1)
            {
                const auto& findBiasIndex =
                        std::find(localShareIndices.begin(), localShareIndices.end(), shareIndex);
                const index localBiasIndex = (findBiasIndex == localShareIndices.end()
                                                      ? -1
                                                      : findBiasIndex - localShareIndices.begin());
                MPI_Comm    splitComm;
                if (static_cast<int>(globalShareIndices.count(shareIndex)) == numSimulations)
                {
                    splitComm = simulationMastersComm;
                }
                else
                {
#if GMX_MPI
                    const int haveLocally = (localBiasIndex >= 0 ? 1 : 0);
                    MPI_Comm_split(simulationMastersComm, haveLocally, myRank, &splitComm);
                    createdCommList_.push_back(splitComm);
#else
                    GMX_RELEASE_ASSERT(false, "Can not have sharing without MPI");
#endif // GMX_MPI
                }
                if (localBiasIndex >= 0)
                {
                    numSharingSimulations_[localBiasIndex] = globalShareIndices.count(shareIndex);
#if GMX_MPI
                    MPI_Comm_rank(splitComm, &sharingSimulationIndices_[localBiasIndex]);
#endif // GMX_MPI
                    multiSimCommPerBias_[localBiasIndex] = splitComm;
                }
            }
        }
    }

#if GMX_MPI
    if (commRecord.nnodes > 1)
    {
        numSharingSimulations_.resize(awhParams.numBias);
        MPI_Bcast(numSharingSimulations_.data(), numSharingSimulations_.size(), MPI_INT, 0,
                  commRecord.mpi_comm_mygroup);
    }
#endif // GMX_MPI
}

BiasSharing::~BiasSharing()
{
#if GMX_MPI
    for (MPI_Comm comm : createdCommList_)
    {
        MPI_Comm_free(&comm);
    }
#endif // GMX_MPI
}

namespace
{

#if GMX_MPI

template<typename T>
std::enable_if_t<std::is_same_v<T, int>, MPI_Datatype> mpiType()
{
    return MPI_INT;
}

template<typename T>
std::enable_if_t<std::is_same_v<T, long>, MPI_Datatype> mpiType()
{
    return MPI_LONG;
}

template<typename T>
std::enable_if_t<std::is_same_v<T, double>, MPI_Datatype> mpiType()
{
    return MPI_DOUBLE;
}

#endif // GMX_MPI

} // namespace

/*! \brief
 * Sum an array over all simulations on all ranks of each simulation.
 *
 * This assumes the data is identical on all ranks within each simulation.
 *
 * \param[in,out] data          The data to sum.
 * \param[in]     commRecord    Struct for intra-simulation communication.
 * \param[in]     multiSimComm  Communicator for the master rank of sharing simulations.
 */
template<typename T>
void sumOverSimulations(ArrayRef<T> data, const t_commrec& commRecord, const MPI_Comm multiSimComm)
{
#if GMX_MPI
    if (MASTER(&commRecord))
    {
#    if MPI_IN_PLACE_EXISTS
        MPI_Allreduce(MPI_IN_PLACE, data.data(), data.size(), mpiType<T>(), MPI_SUM, multiSimComm);
#    else
        std::vector<T> buffer(data.size());
        MPI_Allreduce(data.data(), buffer.data(), data.size(), mpiType<T>(), MPI_SUM, multiSimComm);
        std::copy(buffer.begin(), buffer.end(), data.begin());
#    endif
    }
    if (commRecord.nnodes > 1)
    {
        gmx_bcast(data.size() * sizeof(T), data.data(), commRecord.mpi_comm_mygroup);
    }
#else
    GMX_UNUSED_VALUE(data);
    GMX_UNUSED_VALUE(commRecord);
    GMX_UNUSED_VALUE(multiSimComm);
#endif // GMX_MPI
}

void BiasSharing::sum(ArrayRef<int> data, const int biasIndex) const
{
    sumOverSimulations(data, commRecord_, multiSimCommPerBias_[biasIndex]);
}

void BiasSharing::sum(ArrayRef<long> data, const int biasIndex) const
{
    sumOverSimulations(data, commRecord_, multiSimCommPerBias_[biasIndex]);
}

void BiasSharing::sum(ArrayRef<double> data, const int biasIndex) const
{
    sumOverSimulations(data, commRecord_, multiSimCommPerBias_[biasIndex]);
}

bool haveBiasSharingWithinSimulation(const AwhParams& awhParams)
{
    bool haveSharing = false;

    for (int k = 0; k < awhParams.numBias; k++)
    {
        int shareGroup = awhParams.awhBiasParams[k].shareGroup;
        if (shareGroup > 0)
        {
            for (int i = k + 1; i < awhParams.numBias; i++)
            {
                if (awhParams.awhBiasParams[i].shareGroup == shareGroup)
                {
                    haveSharing = true;
                }
            }
        }
    }

    return haveSharing;
}

void biasesAreCompatibleForSharingBetweenSimulations(const AwhParams&           awhParams,
                                                     const std::vector<size_t>& pointSize,
                                                     const BiasSharing&         biasSharing)
{
    /* We currently enforce subsequent shared biases to have consecutive
     * share-group values starting at 1. This means we can reduce shared
     * biases in order over the ranks and it does not restrict possibilities.
     */
    int numShare = 0;
    for (int b = 0; b < awhParams.numBias; b++)
    {
        int group = awhParams.awhBiasParams[b].shareGroup;
        if (group > 0)
        {
            numShare++;
            if (group != numShare)
            {
                GMX_THROW(
                        InvalidInputError("AWH biases that are shared should use increasing "
                                          "share-group values"));
            }
        }
    }

    /* Check the point sizes. This is a sufficient condition for running
     * as shared multi-sim run. No physics checks are performed here.
     */
    for (int b = 0; b < awhParams.numBias; b++)
    {
        if (awhParams.awhBiasParams[b].shareGroup > 0)
        {
            const int        numSim   = biasSharing.numSharingSimulations(b);
            const int        simIndex = biasSharing.sharingSimulationIndex(b);
            std::vector<int> intervals(numSim * 2);
            intervals[numSim * 0 + simIndex] = awhParams.nstSampleCoord;
            intervals[numSim * 1 + simIndex] = awhParams.numSamplesUpdateFreeEnergy;
            biasSharing.sum(intervals, b);
            for (int sim = 1; sim < numSim; sim++)
            {
                if (intervals[sim] != intervals[0])
                {
                    GMX_THROW(InvalidInputError(
                            "All simulations should have the same AWH sample interval"));
                }
                if (intervals[numSim + sim] != intervals[numSim])
                {
                    GMX_THROW(
                            InvalidInputError("All simulations should have the same AWH "
                                              "free-energy update interval"));
                }
            }

            std::vector<long> pointSizes(numSim);
            pointSizes[simIndex] = pointSize[b];
            biasSharing.sum(pointSizes, b);
            for (int sim = 1; sim < numSim; sim++)
            {
                if (pointSizes[sim] != pointSizes[0])
                {
                    GMX_THROW(InvalidInputError(
                            gmx::formatString("Shared AWH bias %d has different grid sizes in "
                                              "different simulations\n",
                                              b + 1)));
                }
            }
        }
    }
}

} // namespace gmx
