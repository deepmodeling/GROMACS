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
 *
 * \brief Implements class which recieves coordinates to GPU memory on PME task using CUDA
 *
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "pme_coordinate_receiver_gpu_impl.h"

#include "config.h"

#include "gromacs/ewald/pme_force_sender_gpu.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/gpueventsynchronizer.cuh"
#include "gromacs/utility/gmxmpi.h"

namespace gmx
{

PmeCoordinateReceiverGpu::Impl::Impl(const DeviceStream&    pmeStream,
                                     MPI_Comm               comm,
                                     gmx::ArrayRef<PpRanks> ppRanks) :
    comm_(comm),
    ppRanks_(ppRanks)
#if GMX_THREAD_MPI
    ,
    pmeStream_(pmeStream)
#endif
{
    request_.resize(ppRanks.size());

#if GMX_THREAD_MPI
    ppSync_.resize(ppRanks.size());
#endif
}

PmeCoordinateReceiverGpu::Impl::~Impl() = default;


void PmeCoordinateReceiverGpu::Impl::sendCoordinateBufferAddressToPpRanks(DeviceBuffer<RVec> d_x)
{
#if GMX_THREAD_MPI
    int ind_start = 0;
    int ind_end   = 0;
    for (const auto& receiver : ppRanks_)
    {
        ind_start = ind_end;
        ind_end   = ind_start + receiver.numAtoms;

        // Data will be transferred directly from GPU.
        void* sendBuf = reinterpret_cast<void*>(&d_x[ind_start]);

        MPI_Send(&sendBuf, sizeof(void**), MPI_BYTE, receiver.rankId, 0, comm_);
    }
#endif
}

/*! \brief Receive coordinate data directly using CUDA memory copy */
void PmeCoordinateReceiverGpu::Impl::launchReceiveCoordinatesFromPp(DeviceBuffer<RVec> recvbuf,
                                                                    int                nat,
                                                                    int                numBytes,
                                                                    int                ppRank)
{
    // Data will be pushed directly from PP task

#if GMX_MPI

#    if GMX_THREAD_MPI
    // Receive event from PP task
    MPI_Irecv(&ppSync_[recvCount_], sizeof(GpuEventSynchronizer*), MPI_BYTE, ppRank, 0, comm_,
              &request_[recvCount_]);

    GMX_UNUSED_VALUE(recvbuf);
    GMX_UNUSED_VALUE(nat);
    GMX_UNUSED_VALUE(numBytes);
#    else

    MPI_Irecv(&recvbuf[nat], numBytes, MPI_BYTE, ppRank, 0, comm_, &request_[recvCount_]);

#    endif // GMX_THREAD_MPI

    recvCount_++;
#else
    GMX_UNUSED_VALUE(recvbuf);
    GMX_UNUSED_VALUE(nat);
    GMX_UNUSED_VALUE(numBytes);
    GMX_UNUSED_VALUE(ppRank);
#endif
}

void PmeCoordinateReceiverGpu::Impl::waitOrEnqueueWaitReceiveCoordinatesFromPp()
{
    if (recvCount_ > 0)
    {
        // ensure PME calculation doesn't commence until coordinate data has been transferred
#if GMX_MPI
        MPI_Waitall(recvCount_, request_.data(), MPI_STATUS_IGNORE);
#endif

#if GMX_THREAD_MPI
        for (int i = 0; i < recvCount_; i++)
        {
            ppSync_[i]->enqueueWaitEvent(pmeStream_);
        }
#endif

        // reset receive counter
        recvCount_ = 0;
    }
}

PmeCoordinateReceiverGpu::PmeCoordinateReceiverGpu(const DeviceStream&    pmeStream,
                                                   MPI_Comm               comm,
                                                   gmx::ArrayRef<PpRanks> ppRanks) :
    impl_(new Impl(pmeStream, comm, ppRanks))
{
}

PmeCoordinateReceiverGpu::~PmeCoordinateReceiverGpu() = default;

void PmeCoordinateReceiverGpu::sendCoordinateBufferAddressToPpRanks(DeviceBuffer<RVec> d_x)
{
    impl_->sendCoordinateBufferAddressToPpRanks(d_x);
}

void PmeCoordinateReceiverGpu::launchReceiveCoordinatesFromPp(DeviceBuffer<RVec> recvbuf,
                                                              int                nat,
                                                              int                numBytes,
                                                              int                ppRank)
{
    impl_->launchReceiveCoordinatesFromPp(recvbuf, nat, numBytes, ppRank);
}

void PmeCoordinateReceiverGpu::waitOrEnqueueWaitReceiveCoordinatesFromPp()
{
    impl_->waitOrEnqueueWaitReceiveCoordinatesFromPp();
}

} // namespace gmx
