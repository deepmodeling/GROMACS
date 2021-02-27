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

#include "pme_pp_comm_gpu_impl.h"

#include "config.h"

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.cuh"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/utility/gmxmpi.h"

namespace gmx
{

PmePpCommGpu::Impl::Impl(MPI_Comm             comm,
                         int                  pmeRank,
                         const DeviceContext& deviceContext,
                         const DeviceStream&  deviceStream) :
    deviceContext_(deviceContext),
    comm_(comm),
    pmeRank_(pmeRank),
    pmePpCommStream_(deviceStream)
{
}

PmePpCommGpu::Impl::~Impl() {}

void PmePpCommGpu::Impl::reinit(int size)
{
    // This rank will access PME rank memory directly, so needs to receive the remote PME buffer addresses.
#if GMX_MPI

    if (GMX_THREAD_MPI)
    {
        MPI_Recv(&remotePmeXBuffer_, sizeof(float3**), MPI_BYTE, pmeRank_, 0, comm_, MPI_STATUS_IGNORE);
        MPI_Recv(&remotePmeFBuffer_, sizeof(float3**), MPI_BYTE, pmeRank_, 0, comm_, MPI_STATUS_IGNORE);
    }

#endif

    // Reallocate buffer used for staging PME force on GPU
    reallocateDeviceBuffer(&d_pmeForces_, size, &d_pmeForcesSize_, &d_pmeForcesSizeAlloc_, deviceContext_);
    return;
}

void PmePpCommGpu::Impl::receiveForceFromPmeCudaDirect(float3* pmeForcePtr, int recvSize, bool receivePmeForceToGpu)
{
    // Receive event from PME task and add to stream, to ensure pull of data doesn't
    // occur before PME force calc is completed

#if GMX_MPI
    GpuEventSynchronizer* pmeSync;
    MPI_Recv(&pmeSync, sizeof(GpuEventSynchronizer*), MPI_BYTE, pmeRank_, 0, comm_, MPI_STATUS_IGNORE);
    pmeSync->enqueueWaitEvent(pmePpCommStream_);
#endif

    // Pull force data from remote GPU
    cudaError_t stat = cudaMemcpyAsync(pmeForcePtr,
                                       remotePmeFBuffer_,
                                       recvSize * DIM * sizeof(float),
                                       cudaMemcpyDefault,
                                       pmePpCommStream_.stream());
    CU_RET_ERR(stat, "cudaMemcpyAsync on Recv from PME CUDA direct data transfer failed");

    if (receivePmeForceToGpu)
    {
        // Record event to be enqueued in the GPU local buffer operations, to
        // satisfy dependency on receiving the PME force data before
        // reducing it with the other force contributions.
        forcesReadySynchronizer_.markEvent(pmePpCommStream_);
    }
    else
    {
        // Ensure CPU waits for PME forces to be copied before reducing
        // them with other forces on the CPU
        pmePpCommStream_.synchronize();
    }
}

void PmePpCommGpu::Impl::receiveForceFromPmeCudaMPI(float3* pmeForcePtr, int recvSize)
{
#if GMX_MPI
    // arbitrarily chosen
    const int tag = 101;

    MPI_Recv(pmeForcePtr, recvSize * DIM, MPI_FLOAT, pmeRank_, tag, comm_, MPI_STATUS_IGNORE);
#else
    GMX_UNUSED_VALUE(pmeForcePtr);
    GMX_UNUSED_VALUE(recvSize);
#endif
}

void PmePpCommGpu::Impl::receiveForceFromPme(float3* recvPtr, int recvSize, bool receivePmeForceToGpu)
{
    float3* pmeForcePtr = receivePmeForceToGpu ? d_pmeForces_ : recvPtr;
    if (GMX_THREAD_MPI)
    {
        receiveForceFromPmeCudaDirect(pmeForcePtr, recvSize, receivePmeForceToGpu);
    }
    else
    {
        receiveForceFromPmeCudaMPI(pmeForcePtr, recvSize);
    }
}

void PmePpCommGpu::Impl::sendCoordinatesToPmeCudaDirect(float3*               sendPtr,
                                                        int                   sendSize,
                                                        GpuEventSynchronizer* coordinatesReadyOnDeviceEvent)
{
    // ensure stream waits until coordinate data is available on device
    coordinatesReadyOnDeviceEvent->enqueueWaitEvent(pmePpCommStream_);

    cudaError_t stat = cudaMemcpyAsync(remotePmeXBuffer_,
                                       sendPtr,
                                       sendSize * DIM * sizeof(float),
                                       cudaMemcpyDefault,
                                       pmePpCommStream_.stream());
    CU_RET_ERR(stat, "cudaMemcpyAsync on Send to PME CUDA direct data transfer failed");

#if GMX_MPI
    // Record and send event to allow PME task to sync to above transfer before commencing force calculations
    pmeCoordinatesSynchronizer_.markEvent(pmePpCommStream_);

    GpuEventSynchronizer* pmeSync = &pmeCoordinatesSynchronizer_;
    MPI_Send(&pmeSync, sizeof(GpuEventSynchronizer*), MPI_BYTE, pmeRank_, 0, comm_);
#endif
}

void PmePpCommGpu::Impl::sendCoordinatesToPmeCudaMPI(float3*               sendPtr,
                                                     int                   sendSize,
                                                     GpuEventSynchronizer* coordinatesReadyOnDeviceEvent)
{
    // ensure coordinate data is available on device before we start transfer
    coordinatesReadyOnDeviceEvent->waitForEvent();

#if GMX_MPI
    float3* sendptr_x = sendPtr;

    MPI_Send(sendptr_x, sendSize * DIM, MPI_FLOAT, pmeRank_, 0, comm_);
#else
    GMX_UNUSED_VALUE(sendPtr);
    GMX_UNUSED_VALUE(sendSize);
#endif
}

void PmePpCommGpu::Impl::sendCoordinatesToPme(float3*               sendPtr,
                                              int                   sendSize,
                                              GpuEventSynchronizer* coordinatesReadyOnDeviceEvent)
{
    if (GMX_THREAD_MPI)
    {
        sendCoordinatesToPmeCudaDirect(sendPtr, sendSize, coordinatesReadyOnDeviceEvent);
    }
    else
    {
        sendCoordinatesToPmeCudaMPI(sendPtr, sendSize, coordinatesReadyOnDeviceEvent);
    }
}
void* PmePpCommGpu::Impl::getGpuForceStagingPtr()
{
    return static_cast<void*>(d_pmeForces_);
}

GpuEventSynchronizer* PmePpCommGpu::Impl::getForcesReadySynchronizer()
{
    if (GMX_THREAD_MPI)
    {
        return &forcesReadySynchronizer_;
    }
    else
    {
        return nullptr;
    }
}

PmePpCommGpu::PmePpCommGpu(MPI_Comm             comm,
                           int                  pmeRank,
                           const DeviceContext& deviceContext,
                           const DeviceStream&  deviceStream) :
    impl_(new Impl(comm, pmeRank, deviceContext, deviceStream))
{
}

PmePpCommGpu::~PmePpCommGpu() = default;

void PmePpCommGpu::reinit(int size)
{
    impl_->reinit(size);
}

void PmePpCommGpu::receiveForceFromPme(RVec* recvPtr, int recvSize, bool receivePmeForceToGpu)
{
    impl_->receiveForceFromPme(asFloat3(recvPtr), recvSize, receivePmeForceToGpu);
}

void PmePpCommGpu::sendCoordinatesToPmeFromGPU(DeviceBuffer<RVec>    sendPtr,
                                               int                   sendSize,
                                               GpuEventSynchronizer* coordinatesReadyOnDeviceEvent)
{
    impl_->sendCoordinatesToPme(asFloat3(sendPtr), sendSize, coordinatesReadyOnDeviceEvent);
}

void PmePpCommGpu::sendCoordinatesToPmeFromCPU(RVec*                 sendPtr,
                                               int                   sendSize,
                                               GpuEventSynchronizer* coordinatesReadyOnDeviceEvent)
{
    impl_->sendCoordinatesToPme(asFloat3(sendPtr), sendSize, coordinatesReadyOnDeviceEvent);
}

void* PmePpCommGpu::getGpuForceStagingPtr()
{
    return impl_->getGpuForceStagingPtr();
}

GpuEventSynchronizer* PmePpCommGpu::getForcesReadySynchronizer()
{
    return impl_->getForcesReadySynchronizer();
}

} // namespace gmx
