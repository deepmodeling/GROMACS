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
    status_(MPI_REQUEST_NULL)
#if GMX_THREAD_MPI
    ,
    pmePpCommStream_(deviceStream)
#endif
{
}

PmePpCommGpu::Impl::~Impl()
{
#if GMX_MPI
#    if GMX_THREAD_MPI
#    else
    // resource clean-up as MPI_waitcall might not be called on this status_
    if (status_ != MPI_REQUEST_NULL)
        MPI_Request_free(&status_);

    // free staging buffer on GPU. This code is workaround for UCX bug
    // https://github.com/openucx/ucx/issues/4722
    freeDeviceBuffer(d_ppCoord_);
#    endif
#endif
}

void PmePpCommGpu::Impl::reinit(int size)
{
    // This rank will access PME rank memory directly, so needs to receive the remote PME buffer addresses.
#if GMX_MPI

#    if GMX_THREAD_MPI

    MPI_Recv(&remotePmeXBuffer_, sizeof(void**), MPI_BYTE, pmeRank_, 0, comm_, MPI_STATUS_IGNORE);
    MPI_Recv(&remotePmeFBuffer_, sizeof(void**), MPI_BYTE, pmeRank_, 0, comm_, MPI_STATUS_IGNORE);
#    else
    // Reallocate buffer used for staging PP co-ordinates on GPU. This is needed only for process-MPI
    // as UCX layer has bug due to which host->device data trasnfer seg faults inside UCX layer.
    // Bug: https://github.com/openucx/ucx/issues/4722
    // ToDo: Evaluate if we really need to create new staging area or some already existing memory can be used
    // like stateGpu->getCoordinates()
    reallocateDeviceBuffer(&d_ppCoord_, size, &d_ppCoordSize_, &d_ppCoordSizeAlloc_, deviceContext_);
#    endif

    // Reallocate buffer used for staging PME force on GPU
    reallocateDeviceBuffer(&d_pmeForces_, size, &d_pmeForcesSize_, &d_pmeForcesSizeAlloc_, deviceContext_);
#else
    GMX_UNUSED_VALUE(size);
#endif
    return;
}

#if GMX_MPI
#    if GMX_THREAD_MPI
// TODO make this asynchronous by splitting into this into
// launchRecvForceFromPmeCudaDirect() and sycnRecvForceFromPmeCudaDirect()
void PmePpCommGpu::Impl::receiveForceFromPmeCUDADirect(void* pmeForcePtr, int recvSize, bool receivePmeForceToGpu)
{
    // Receive event from PME task and add to stream, to ensure pull of data doesn't
    // occur before PME force calc is completed
    GpuEventSynchronizer* pmeSync;
    MPI_Recv(&pmeSync, sizeof(GpuEventSynchronizer*), MPI_BYTE, pmeRank_, 0, comm_, MPI_STATUS_IGNORE);
    pmeSync->enqueueWaitEvent(pmePpCommStream_);

    // Pull force data from remote GPU
    cudaError_t stat = cudaMemcpyAsync(pmeForcePtr, remotePmeFBuffer_, recvSize * DIM * sizeof(float),
                                       cudaMemcpyDefault, pmePpCommStream_.stream());
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
        cudaStreamSynchronize(pmePpCommStream_.stream());
    }
}
#    else

void PmePpCommGpu::Impl::receiveForceFromPmeCUDAMPI(void* pmeForcePtr, int recvSize, bool receivePmeForceToGpu)
{
    if (status_ != MPI_REQUEST_NULL)
    {
        // Resource clean-up
        MPI_Request_free(&status_);
        status_ = MPI_REQUEST_NULL;
    }

    MPI_Irecv(pmeForcePtr, recvSize * DIM, MPI_FLOAT, pmeRank_, 0, comm_, &status_);
}
#    endif
#endif

void PmePpCommGpu::Impl::receiveForceFromPme(void* recvPtr, int recvSize, bool receivePmeForceToGpu)
{
#if GMX_MPI
    void* pmeForcePtr = receivePmeForceToGpu ? static_cast<void*>(d_pmeForces_) : recvPtr;

#    if GMX_THREAD_MPI
    receiveForceFromPmeCUDADirect(pmeForcePtr, recvSize, receivePmeForceToGpu);
#    else
    receiveForceFromPmeCUDAMPI(pmeForcePtr, recvSize, receivePmeForceToGpu);

    if (!receivePmeForceToGpu)
    {
        // Need to wait as host pointer is expected to be used right after this call
        MPI_Wait(&status_, MPI_STATUS_IGNORE);
    }

#    endif // GMX_THREAD_MPI

#else
    GMX_UNUSED_VALUE(recvPtr);
    GMX_UNUSED_VALUE(recvSize);
    GMX_UNUSED_VALUE(receivePmeForceToGpu);
#endif
}

#if GMX_MPI
#    if GMX_THREAD_MPI
void PmePpCommGpu::Impl::sendCoordinatesToPmeCudaDirect(void* sendPtr,
                                                        int   sendSize,
                                                        bool gmx_unused sendPmeCoordinatesFromGpu,
                                                        GpuEventSynchronizer* coordinatesReadyOnDeviceEvent)
{
    // ensure stream waits until coordinate data is available on device
    coordinatesReadyOnDeviceEvent->enqueueWaitEvent(pmePpCommStream_);

    cudaError_t stat = cudaMemcpyAsync(remotePmeXBuffer_, sendPtr, sendSize * DIM * sizeof(float),
                                       cudaMemcpyDefault, pmePpCommStream_.stream());
    CU_RET_ERR(stat, "cudaMemcpyAsync on Send to PME CUDA direct data transfer failed");

    // Record and send event to allow PME task to sync to above transfer before commencing force calculations
    pmeCoordinatesSynchronizer_.markEvent(pmePpCommStream_);
    GpuEventSynchronizer* pmeSync = &pmeCoordinatesSynchronizer_;
    MPI_Send(&pmeSync, sizeof(GpuEventSynchronizer*), MPI_BYTE, pmeRank_, 0, comm_);
}

#    else

void PmePpCommGpu::Impl::sendCoordinatesToPmeCudaMPI(void* sendPtr,
                                                     int sendSize,
                                                     bool gmx_unused sendPmeCoordinatesFromGpu,
                                                     GpuEventSynchronizer* coordinatesReadyOnDeviceEvent)
{
    // ensure coordinate data is available on device before we start transfer
    coordinatesReadyOnDeviceEvent->waitForEvent();

    void* sendptr_x = sendPtr;
    if (!sendPmeCoordinatesFromGpu)
    {
        // need an explcit copy as UCX has a bug due to which sending host buffer
        // to a device buffer cause crash inside UCX. This has been reported to UCX team.
        cudaError_t stat =
                cudaMemcpy(d_ppCoord_, sendPtr, sendSize * DIM * sizeof(float), cudaMemcpyHostToDevice);
        CU_RET_ERR(stat, "cudaMemcpy on Send to PME CUDA data transfer failed");

        sendptr_x = d_ppCoord_;
    }

    MPI_Send(sendptr_x, sendSize * DIM, MPI_FLOAT, pmeRank_, 0, comm_);
}
#    endif
#endif

void PmePpCommGpu::Impl::sendCoordinatesToPme(void* sendPtr,
                                              int   sendSize,
                                              bool gmx_unused       sendPmeCoordinatesFromGpu,
                                              GpuEventSynchronizer* coordinatesReadyOnDeviceEvent)
{
#if GMX_MPI

#    if GMX_THREAD_MPI
    sendCoordinatesToPmeCudaDirect(sendPtr, sendSize, sendPmeCoordinatesFromGpu,
                                   coordinatesReadyOnDeviceEvent);
#    else
    sendCoordinatesToPmeCudaMPI(sendPtr, sendSize, sendPmeCoordinatesFromGpu, coordinatesReadyOnDeviceEvent);
#    endif // GMX_THREAD_MPI

#else
    GMX_UNUSED_VALUE(sendPtr);
    GMX_UNUSED_VALUE(sendSize);
    GMX_UNUSED_VALUE(sendPmeCoordinatesFromGpu);
    GMX_UNUSED_VALUE(coordinatesReadyOnDeviceEvent);
#endif
}
void* PmePpCommGpu::Impl::getGpuForceStagingPtr()
{
    return static_cast<void*>(d_pmeForces_);
}

void* PmePpCommGpu::Impl::waitForcesReadyOrGetSynchronizer()
{
#if GMX_THREAD_MPI
    return static_cast<void*>(&forcesReadySynchronizer_);
#else
    MPI_Wait(&status_, MPI_STATUS_IGNORE);
    return nullptr;
#endif
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

void PmePpCommGpu::receiveForceFromPme(void* recvPtr, int recvSize, bool receivePmeForceToGpu)
{
    impl_->receiveForceFromPme(recvPtr, recvSize, receivePmeForceToGpu);
}

void PmePpCommGpu::sendCoordinatesToPme(void*                 sendPtr,
                                        int                   sendSize,
                                        bool                  sendPmeCoordinatesFromGpu,
                                        GpuEventSynchronizer* coordinatesReadyOnDeviceEvent)
{
    impl_->sendCoordinatesToPme(sendPtr, sendSize, sendPmeCoordinatesFromGpu, coordinatesReadyOnDeviceEvent);
}

void* PmePpCommGpu::getGpuForceStagingPtr()
{
    return impl_->getGpuForceStagingPtr();
}

void* PmePpCommGpu::waitForcesReadyOrGetSynchronizer()
{
    return impl_->waitForcesReadyOrGetSynchronizer();
}

} // namespace gmx
