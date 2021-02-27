/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020,2021, by the GROMACS development team, led by
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
 * \brief Implements PME-PP communication using CUDA
 *
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "pme_force_sender_gpu_impl.h"

#include "config.h"

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/gpueventsynchronizer.cuh"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/utility/gmxmpi.h"

namespace gmx
{

/*! \brief Create PME-PP GPU communication object */
PmeForceSenderGpu::Impl::Impl(const DeviceStream& pmeStream, MPI_Comm comm, gmx::ArrayRef<PpRanks> ppRanks) :
    pmeStream_(pmeStream),
    comm_(comm),
    ppRanks_(ppRanks)
{
}

PmeForceSenderGpu::Impl::~Impl() {}

/*! \brief  sends force buffer address to PP ranks */
void PmeForceSenderGpu::Impl::sendForceBufferAddressToPpRanks(float3* d_f)
{
#if GMX_MPI
    if (GMX_THREAD_MPI)
    {
        int ind_start = 0;
        int ind_end   = 0;
        for (const auto& receiver : ppRanks_)
        {
            ind_start = ind_end;
            ind_end   = ind_start + receiver.numAtoms;

            // Data will be transferred directly from GPU.
            float3* sendBuf = &d_f[ind_start];

            MPI_Send(&sendBuf, sizeof(float3**), MPI_BYTE, receiver.rankId, 0, comm_);
        }
    }
#else
    GMX_UNUSED_VALUE(d_f);
#endif
}

/*! \brief Send PME data directly using CUDA memory copy */
void PmeForceSenderGpu::Impl::sendFToPpCudaDirect(int ppRank)
{
#if GMX_MPI
    // Data will be pulled directly from PP task

    // Record and send event to ensure PME force calcs are completed before PP task pulls data
    pmeSync_.markEvent(pmeStream_);
    GpuEventSynchronizer* pmeSyncPtr = &pmeSync_;

    MPI_Send(&pmeSyncPtr, sizeof(GpuEventSynchronizer*), MPI_BYTE, ppRank, 0, comm_);
#else
    GMX_UNUSED_VALUE(ppRank);
#endif
}

/*! \brief Send PME data directly using CUDA-aware MPI */
void PmeForceSenderGpu::Impl::sendFToPpCudaMpi(float3* sendbuf, int numBytes, int ppRank, MPI_Request* request)
{
#if GMX_MPI
    GMX_ASSERT(sendCount_ < (int)(ppRanks_.size()), "sendCount_ different from expected values");

    // Ensure PME force calcs are completed before data is sent
    // we need to synchronize PME stream only for the first time
    if (sendCount_ == 0)
    {
        pmeStream_.synchronize();
    }

    // arbitrarily chosen
    const int tag = 101;

    MPI_Isend(sendbuf, numBytes, MPI_BYTE, ppRank, tag, comm_, request);

    if (++sendCount_ == (int)(ppRanks_.size()))
        sendCount_ = 0;
#else
    GMX_UNUSED_VALUE(sendbuf);
    GMX_UNUSED_VALUE(numBytes);
    GMX_UNUSED_VALUE(ppRank);
    GMX_UNUSED_VALUE(request);
#endif
}

/*! \brief Send PME data to PP rank */
void PmeForceSenderGpu::Impl::sendFToPp(float3* sendbuf, int numBytes, int ppRank, MPI_Request* request)
{
    if (GMX_THREAD_MPI)
    {
        sendFToPpCudaDirect(ppRank);
    }
    else
    {
        sendFToPpCudaMpi(sendbuf, numBytes, ppRank, request);
    }
}

PmeForceSenderGpu::PmeForceSenderGpu(const DeviceStream&    pmeStream,
                                     MPI_Comm               comm,
                                     gmx::ArrayRef<PpRanks> ppRanks) :
    impl_(new Impl(pmeStream, comm, ppRanks))
{
}

PmeForceSenderGpu::~PmeForceSenderGpu() = default;

void PmeForceSenderGpu::sendForceBufferAddressToPpRanks(RVec* d_f)
{
    impl_->sendForceBufferAddressToPpRanks(asFloat3(d_f));
}

void PmeForceSenderGpu::sendFToPp(RVec* sendbuf, int numBytes, int ppRank, MPI_Request* request)
{
    impl_->sendFToPp(asFloat3(sendbuf), numBytes, ppRank, request);
}


} // namespace gmx
