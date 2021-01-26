/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017,2018,2019,2020,2021, by the GROMACS development team, led by
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
 *  \brief Implements CUDA FFT routines for PME GPU.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *  \ingroup module_ewald
 */

#include "gmxpre.h"

#include "pme_gpu_3dfft.h"

#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/devicebuffer.cuh"

#include "pme.cuh"
#include "pme_gpu_types.h"
#include "pme_gpu_types_host.h"
#include "pme_gpu_types_host_impl.h"

static void handleCufftError(cufftResult_t status, const char* msg)
{
    if (status != CUFFT_SUCCESS)
    {
        gmx_fatal(FARGS, "%s (error code %d)\n", msg, status);
    }
}

#if CUDA_AWARE_MPI

// CUDA block size x and y-dim
constexpr int c_threads = 16;

/*! \brief
 * A CUDA kernel which converts grid from XYZ to YZX layout in case of forward fft
 * and converts from YZX to XYZ layout in case of reverse fft
 *
 * \tparam[in] forward            Forward pass or reverse pass
 *
 * \param[in] gm_arrayIn          Input local grid
 * \param[in] gm_arrayOut         Output local grid in converted layout
 * \param[in] sizeX               Grid size in X-dim.
 * \param[in] sizeY               Grid size in Y-dim.
 * \param[in] sizeZ               Grid size in Z-dim.
 */
template<bool forward>
static __global__ void transposeXyzToYzxKernel(const cufftComplex* __restrict__ gm_arrayIn,
                                               cufftComplex* __restrict__ gm_arrayOut,
                                               const int sizeX,
                                               const int sizeY,
                                               const int sizeZ)
{
    __shared__ cufftComplex sm_temp[c_threads][c_threads];
    int                     x = blockIdx.x * blockDim.x + threadIdx.x;
    int                     y = blockIdx.y;
    int                     z = blockIdx.z * blockDim.z + threadIdx.z;

    // use threads in other order for xyz (works as blockDim.x == blockDim.z)
    int xt = blockIdx.x * blockDim.x + threadIdx.z;
    int zt = blockIdx.z * blockDim.z + threadIdx.x;

    int  xyzIndex = zt + y * sizeZ + xt * sizeY * sizeZ;
    int  yzxIndex = x + z * sizeX + y * sizeX * sizeZ;
    int  inIndex, outIndex;
    bool validIn, validOut;

    if (forward) // xyz to yzx
    {
        inIndex  = xyzIndex;
        outIndex = yzxIndex;
        validIn  = (xt < sizeX && zt < sizeZ);
        validOut = (x < sizeX && z < sizeZ);
    }
    else // yzx to xyz
    {
        inIndex  = yzxIndex;
        outIndex = xyzIndex;
        validIn  = (x < sizeX && z < sizeZ);
        validOut = (xt < sizeX && zt < sizeZ);
    }

    if (validIn)
    {
        sm_temp[threadIdx.x][threadIdx.z] = gm_arrayIn[inIndex];
    }
    __syncthreads();

    if (validOut)
    {
        gm_arrayOut[outIndex] = sm_temp[threadIdx.z][threadIdx.x];
    }
}

/*! \brief
 * A CUDA kernel which merges multiple blocks in YZX layout from different ranks
 *
 * \param[in] gm_arrayIn          Input local grid
 * \param[in] gm_arrayOut         Output local grid in converted layout
 * \param[in] sizeX               Grid size in X-dim.
 * \param[in] sizeY               Grid size in Y-dim.
 * \param[in] sizeZ               Grid size in Z-dim.
 * \param[in] xBlockSizes         Array containing X-block sizes for each rank
 * \param[in] xOffset             Array containing grid offsets for each rank
 */
static __global__ void convertBlockedYzxToYzxKernel(const cufftComplex* __restrict__ gm_arrayIn,
                                                    cufftComplex* __restrict__ gm_arrayOut,
                                                    const int sizeX,
                                                    const int sizeY,
                                                    const int sizeZ,
                                                    const int* __restrict__ xBlockSizes,
                                                    const int* __restrict__ xOffset)
{
    // no need to cache block unless x_block_size is small
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int region = blockIdx.z;
    int xLocal = thread % xBlockSizes[region];
    int z      = thread / xBlockSizes[region];
    int y      = blockIdx.y;
    int x      = xOffset[region] + xLocal;

    int indexIn  = xLocal + xBlockSizes[region] * (z + sizeZ * y) + xOffset[region] * sizeY * sizeZ;
    int indexOut = x + sizeX * (z + sizeZ * y);

    if (x < xOffset[region + 1] && z < sizeZ)
    {
        gm_arrayOut[indexOut] = gm_arrayIn[indexIn];
    }
}

/*! \brief
 * A CUDA kernel which merges multiple blocks in XYZ layout from different ranks
 *
 * \param[in] gm_arrayIn          Input local grid
 * \param[in] gm_arrayOut         Output local grid in converted layout
 * \param[in] sizeX               Grid size in X-dim.
 * \param[in] sizeY               Grid size in Y-dim.
 * \param[in] sizeZ               Grid size in Z-dim.
 * \param[in] yBlockSizes         Array containing Y-block sizes for each rank
 * \param[in] yOffset             Array containing grid offsets for each rank
 */
static __global__ void convertBlockedXyzToXyzKernel(const cufftComplex* __restrict__ gm_arrayIn,
                                                    cufftComplex* __restrict__ gm_arrayOut,
                                                    const int sizeX,
                                                    const int sizeY,
                                                    const int sizeZ,
                                                    const int* __restrict__ yBlockSizes,
                                                    const int* __restrict__ yOffset)
{
    int x      = blockIdx.y;
    int yz     = blockIdx.x * blockDim.x + threadIdx.x;
    int region = blockIdx.z;
    int z      = yz % sizeZ;
    int yLocal = yz / sizeZ;

    int y        = yLocal + yOffset[region];
    int indexIn  = z + sizeZ * (yLocal + yBlockSizes[region] * x + sizeX * yOffset[region]);
    int indexOut = z + sizeZ * (y + sizeY * x);

    if (y < yOffset[region + 1] && z < sizeZ)
    {
        gm_arrayOut[indexOut] = gm_arrayIn[indexIn];
    }
}

/*! \brief
 * A CUDA kernel which adds grid overlap data received from neighboring rank
 *
 * \param[in] gm_realGrid          local grid
 * \param[in] gm_transferGrid      overlapping region from neighboring rank
 * \param[in] size                 size of overlap region
 */
__global__ void pmeGpuAddHalo(float* __restrict__ gm_realGrid, const float* __restrict__ gm_transferGrid, int size)
{
    int val = threadIdx.x + blockIdx.x * blockDim.x;
    if (val < size)
    {
        gm_realGrid[val] += gm_transferGrid[val];
    }
}

void GpuParallel3dFft::pmeGpuHaloExchange()
{
    // Note for here we are assuming that width of the chunks is not so small that we need to
    // transfer to/from multiple ranks i.e. that the distributed grid is at least order-1 points wide.

    auto* kernelParamsPtr = pmeGpu_->kernelParams.get();
    int   sizeY           = kernelParamsPtr->grid.realGridSizePadded[YY];
    int   sizeZ           = kernelParamsPtr->grid.realGridSizePadded[ZZ];

    MPI_Status status;
    int        rank = mpiRank_;
    int        size = mpiSize_;

    // account for periodic boundry conditions
    int send = (rank + 1) % size;
    int recv = (rank + size - 1) % size;
    // For an even split the transfer size should be order -1 (i.e. 4-1=3 for GPUs)
    // For an uneven split the grid is rounded as the ATOMS are split between ranks not the
    // gridlines So we could have a larger halo.

    // Note that s2g0[size] is the grid size (array is allocated to size+1)
    int transferstart = (pmeGpu_->common->s2g0x[rank + 1] - pmeGpu_->common->s2g0x[rank]) * sizeY
                        * sizeZ; // explitly rank+1 here not forward
    int transferSizeSend = (pmeGpu_->common->s2g1x[rank] - pmeGpu_->common->s2g0x[rank + 1]) * sizeY
                           * sizeZ; // explicitly rank+1 here
    int transferSizeRecv = (pmeGpu_->common->s2g1x[recv] - pmeGpu_->common->s2g0x[recv + 1]) * sizeY
                           * sizeZ; // using recv here to account for periodic boundry conditions.
                                    //    float* d_transferGrid;

    int tag = 403; // Arbitrarily chosen
    pme_gpu_synchronize(pmeGpu_);
    MPI_Sendrecv(&realGrid_[transferstart],
                 transferSizeSend,
                 MPI_FLOAT,
                 send,
                 tag,
                 d_transferGrid_,
                 transferSizeRecv,
                 MPI_FLOAT,
                 recv,
                 tag,
                 mpi_comm_,
                 &status);

    const int threadsPerBlock = 64;

    KernelLaunchConfig config;
    config.blockSize[0]     = threadsPerBlock;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (transferSizeRecv + threadsPerBlock - 1) / threadsPerBlock;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;


    auto kernelFn = pmeGpuAddHalo;

    const auto kernelArgs = prepareGpuKernelArguments(
            kernelFn, config, &realGrid_, &d_transferGrid_, &transferSizeRecv);

    launchGpuKernel(kernelFn, config, stream_, nullptr, "PME Domdec GPU Apply Grid Halo Exchange", kernelArgs);
}

void GpuParallel3dFft::pmeGpuHaloExchangeReverse()
{
    auto* kernelParamsPtr = pmeGpu_->kernelParams.get();
    int   sizeY           = kernelParamsPtr->grid.realGridSizePadded[YY];
    int   sizeZ           = kernelParamsPtr->grid.realGridSizePadded[ZZ];

    MPI_Status status;
    int        rank = mpiRank_;
    int        size = mpiSize_;
    int        recv = (rank + 1) % size;
    int        send = (rank + size - 1) % size;

    // see above
    int transferstart = (pmeGpu_->common->s2g0x[rank + 1] - pmeGpu_->common->s2g0x[rank]) * sizeY * sizeZ;
    int transfersize = (pmeGpu_->common->pme_order - 1) * sizeY * sizeZ;

    int tag = 402; // Arbitrarily chosen
    pme_gpu_synchronize(pmeGpu_);
    MPI_Sendrecv(&realGrid_[0],
                 transfersize,
                 MPI_FLOAT,
                 send,
                 tag,
                 &realGrid_[transferstart],
                 transfersize,
                 MPI_FLOAT,
                 recv,
                 tag,
                 mpi_comm_,
                 &status);
}

template<bool forward>
void GpuParallel3dFft::transposeXyzToYzx(cufftComplex* arrayIn, cufftComplex* arrayOut, int sizeX, int sizeY, int sizeZ)
{
    KernelLaunchConfig config;
    config.blockSize[0]     = c_threads;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = c_threads;
    config.gridSize[0]      = (sizeX + c_threads - 1) / c_threads;
    config.gridSize[1]      = sizeY;
    config.gridSize[2]      = (sizeZ + c_threads - 1) / c_threads;
    config.sharedMemorySize = 0;


    auto kernelFn = transposeXyzToYzxKernel<forward>;

    const auto kernelArgs =
            prepareGpuKernelArguments(kernelFn, config, &arrayIn, &arrayOut, &sizeX, &sizeY, &sizeZ);

    launchGpuKernel(kernelFn, config, stream_, nullptr, "PME FFT GPU grid transpose", kernelArgs);
}

void GpuParallel3dFft::convertBlockedYzxToYzx(cufftComplex* arrayIn,
                                              cufftComplex* arrayOut,
                                              int           sizeX,
                                              int           sizeY,
                                              int           sizeZ,
                                              int*          xBlockSizes,
                                              int*          xOffsets,
                                              int           numRegions,
                                              int           maxRegionSize)
{
    int blockDim = c_threads * c_threads;
    int sizexz   = maxRegionSize * sizeZ;

    KernelLaunchConfig config;
    config.blockSize[0]     = blockDim;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (sizexz + blockDim - 1) / blockDim;
    config.gridSize[1]      = sizeY;
    config.gridSize[2]      = numRegions;
    config.sharedMemorySize = 0;


    auto kernelFn = convertBlockedYzxToYzxKernel;

    const auto kernelArgs = prepareGpuKernelArguments(
            kernelFn, config, &arrayIn, &arrayOut, &sizeX, &sizeY, &sizeZ, &xBlockSizes, &xOffsets);

    launchGpuKernel(kernelFn, config, stream_, nullptr, "PME FFT GPU grid rearrange", kernelArgs);
}

void GpuParallel3dFft::convertBlockedXyzToXyz(cufftComplex* arrayIn,
                                              cufftComplex* arrayOut,
                                              int           sizeX,
                                              int           sizeY,
                                              int           sizeZ,
                                              int*          yBlockSizes,
                                              int*          yOffsets,
                                              int           numRegions,
                                              int           maxRegionSize)
{
    int blockDim = c_threads * c_threads;
    int sizexz   = maxRegionSize * sizeZ;

    KernelLaunchConfig config;
    config.blockSize[0]     = blockDim;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (sizexz + blockDim - 1) / blockDim;
    config.gridSize[1]      = sizeX;
    config.gridSize[2]      = numRegions;
    config.sharedMemorySize = 0;


    auto kernelFn = convertBlockedXyzToXyzKernel;

    const auto kernelArgs = prepareGpuKernelArguments(
            kernelFn, config, &arrayIn, &arrayOut, &sizeX, &sizeY, &sizeZ, &yBlockSizes, &yOffsets);

    launchGpuKernel(kernelFn, config, stream_, nullptr, "PME FFT GPU grid rearrange", kernelArgs);
}

#endif // CUDA_AWARE_MPI


GpuParallel3dFft::GpuParallel3dFft(const PmeGpu* pmeGpu, const int gridIndex) :
    stream_(pmeGpu->archSpecific->pmeStream_)
{
    const PmeGpuCudaKernelParams* kernelParamsPtr = pmeGpu->kernelParams.get();
    ivec                          realGridSize, realGridSizePadded, complexGridSizePadded;
#if CUDA_AWARE_MPI
    ivec complexGridSize;
    int  size = 1;
#endif
    for (int i = 0; i < DIM; i++)
    {
        realGridSize[i]          = kernelParamsPtr->grid.realGridSize[i];
        realGridSizePadded[i]    = kernelParamsPtr->grid.realGridSizePadded[i];
        complexGridSizePadded[i] = kernelParamsPtr->grid.complexGridSizePadded[i];
#if CUDA_AWARE_MPI
        complexGridSize[i] = kernelParamsPtr->grid.complexGridSize[i];
        size *= kernelParamsPtr->grid.complexGridSizePadded[i];
#endif
    }

    GMX_RELEASE_ASSERT(CUDA_AWARE_MPI || !pme_gpu_settings(pmeGpu).useDecomposition,
                       "PME decomposition for GPU is supported only with cuda-aware mpi");

    const int complexGridSizePaddedTotal =
            complexGridSizePadded[XX] * complexGridSizePadded[YY] * complexGridSizePadded[ZZ];
    const int realGridSizePaddedTotal =
            realGridSizePadded[XX] * realGridSizePadded[YY] * realGridSizePadded[ZZ];

    realGrid_ = (cufftReal*)kernelParamsPtr->grid.d_realGrid[gridIndex];

    GMX_RELEASE_ASSERT(realGrid_, "Bad (null) input real-space grid");
    complexGrid_ = (cufftComplex*)kernelParamsPtr->grid.d_fourierGrid[gridIndex];
    GMX_RELEASE_ASSERT(complexGrid_, "Bad (null) input complex grid");


    cufftResult_t result;
    /* Commented code for a simple 3D grid with no padding */
    /*
       result = cufftPlan3d(&planR2C_, realGridSize[XX], realGridSize[YY], realGridSize[ZZ],
       CUFFT_R2C); handleCufftError(result, "cufftPlan3d R2C plan failure");

       result = cufftPlan3d(&planC2R_, realGridSize[XX], realGridSize[YY], realGridSize[ZZ],
       CUFFT_C2R); handleCufftError(result, "cufftPlan3d C2R plan failure");
     */

    cudaStream_t stream = stream_.stream();
    GMX_RELEASE_ASSERT(stream, "Using the default CUDA stream for PME cuFFT");

    if (!pmeGpu->settings.useDecomposition)
    {
        int rank  = 3;
        int batch = 1;
        result    = cufftPlanMany(&planR2C_,
                               rank,
                               realGridSize,
                               realGridSizePadded,
                               1,
                               realGridSizePaddedTotal,
                               complexGridSizePadded,
                               1,
                               complexGridSizePaddedTotal,
                               CUFFT_R2C,
                               batch);
        handleCufftError(result, "cufftPlanMany R2C plan failure");
        result = cufftSetStream(planR2C_, stream);
        handleCufftError(result, "cufftSetStream R2C failure");


        result = cufftPlanMany(&planC2R_,
                               rank,
                               realGridSize,
                               complexGridSizePadded,
                               1,
                               complexGridSizePaddedTotal,
                               realGridSizePadded,
                               1,
                               realGridSizePaddedTotal,
                               CUFFT_C2R,
                               batch);
        handleCufftError(result, "cufftPlanMany C2R plan failure");
        result = cufftSetStream(planC2R_, stream);
        handleCufftError(result, "cufftSetStream C2R failure");
    }

    pmeGpu_ = pmeGpu;

#if CUDA_AWARE_MPI
    int mpiSize   = 1;
    int mpiRank   = 0;
    complexGrid2_ = NULL;

    // count and displacement arrays used in MPI_Alltoall call
    sendCount_ = sendDisp_ = recvCount_ = recvDisp_ = NULL;
#    if UCX_MPIALLTOALLV_BUG_HACK
    sendCountTemp_ = recvCountTemp_ = NULL;
#    endif

    // local grid size along decmposed dimension
    d_xBlockSizes_ = d_yBlockSizes_ = NULL;

    // device arrays keeping local grid offsets
    d_s2g0x_ = d_s2g0y_ = NULL;

    // device memory to transfer overlapping regions between ranks
    d_transferGrid_ = NULL;
    if (pmeGpu->settings.useDecomposition)
    {
        changePinningPolicy(&h_xBlockSizes_, gmx::PinningPolicy::PinnedIfSupported);
        changePinningPolicy(&h_yBlockSizes_, gmx::PinningPolicy::PinnedIfSupported);
        changePinningPolicy(&h_s2g0y_, gmx::PinningPolicy::PinnedIfSupported);

        const DeviceContext& devContext = pmeGpu->archSpecific->deviceContext_;

        const int complexGridSizePaddedTotal2D = complexGridSizePadded[YY] * complexGridSizePadded[ZZ];
        const int realGridSizePaddedTotal2D    = realGridSizePadded[YY] * realGridSizePadded[ZZ];

        int localx = realGridSize[XX];
        int localy = realGridSize[YY];

        MPI_Comm_size(pmeGpu->common->mpi_comm, &mpiSize);
        MPI_Comm_rank(pmeGpu->common->mpi_comm, &mpiRank);
        mpi_comm_  = pmeGpu->common->mpi_comm;
        sendCount_ = (int*)malloc(mpiSize * sizeof(int));
        sendDisp_  = (int*)malloc(mpiSize * sizeof(int));
        recvCount_ = (int*)malloc(mpiSize * sizeof(int));
        recvDisp_  = (int*)malloc(mpiSize * sizeof(int));
        h_xBlockSizes_.resize(mpiSize);
        h_yBlockSizes_.resize(mpiSize);
        h_s2g0y_.resize(mpiSize + 1);
        allocateDeviceBuffer(&d_xBlockSizes_, mpiSize, devContext);
        allocateDeviceBuffer(&d_yBlockSizes_, mpiSize, devContext);
        allocateDeviceBuffer(&d_s2g0x_, (mpiSize + 1), devContext);
        allocateDeviceBuffer(&d_s2g0y_, (mpiSize + 1), devContext);

        localx = pmeGpu_->common->s2g0x[mpiRank + 1] - pmeGpu_->common->s2g0x[mpiRank];

        for (int i = 0; i < mpiSize; i++)
        {
            h_s2g0y_[i] = (i * complexGridSizePadded[YY] + 0) / mpiSize;
        }
        h_s2g0y_[mpiSize] = complexGridSizePadded[YY];

        localy        = h_s2g0y_[mpiRank + 1] - h_s2g0y_[mpiRank];
        int totalSend = 0;
        int totalRecv = 0;
        int xmax      = 0;
        int ymax      = 0;
        for (int i = 0; i < mpiSize; i++)
        {
            int ix            = pmeGpu_->common->s2g0x[i + 1] - pmeGpu_->common->s2g0x[i];
            int iy            = h_s2g0y_[i + 1] - h_s2g0y_[i];
            h_xBlockSizes_[i] = ix;
            h_yBlockSizes_[i] = iy;
            if (xmax < ix)
                xmax = ix;
            if (ymax < iy)
                ymax = iy;
            sendCount_[i] = complexGridSize[ZZ] * localx * iy;
            recvCount_[i] = complexGridSize[ZZ] * localy * ix;
            sendDisp_[i]  = totalSend;
            recvDisp_[i]  = totalRecv;
            totalSend += sendCount_[i];
            totalRecv += recvCount_[i];
        }
        xMax_ = xmax;
        yMax_ = ymax;
        copyToDeviceBuffer(
                &d_s2g0x_, pmeGpu_->common->s2g0x.data(), 0, (mpiSize + 1), stream_, GpuApiCallBehavior::Sync, nullptr);
        copyToDeviceBuffer(
                &d_xBlockSizes_, h_xBlockSizes_.data(), 0, mpiSize, stream_, GpuApiCallBehavior::Async, nullptr);
        copyToDeviceBuffer(
                &d_yBlockSizes_, h_yBlockSizes_.data(), 0, mpiSize, stream_, GpuApiCallBehavior::Async, nullptr);
        copyToDeviceBuffer(
                &d_s2g0y_, h_s2g0y_.data(), 0, (mpiSize + 1), stream_, GpuApiCallBehavior::Async, nullptr);

        allocateDeviceBuffer(
                &d_transferGrid_, xmax * realGridSizePadded[YY] * realGridSizePadded[ZZ], devContext);

#    if UCX_MPIALLTOALLV_BUG_HACK
        sendCountTemp_ = (int*)malloc(mpiSize * sizeof(int));
        recvCountTemp_ = (int*)malloc(mpiSize * sizeof(int));

        memcpy(sendCountTemp_, sendCount_, mpiSize * sizeof(int));
        memcpy(recvCountTemp_, recvCount_, mpiSize * sizeof(int));

        // don't make any self copies. UCX has perf issues with self copies
        sendCountTemp_[mpiRank] = 0;
        recvCountTemp_[mpiRank] = 0;
#    endif

        complexGrid2_ = (cufftComplex*)kernelParamsPtr->grid.d_fourierGrid2[gridIndex];
        GMX_RELEASE_ASSERT(complexGrid_, "Bad (null) input complex grid 2");

        int rank  = 2;
        int batch = localx;
        // split 3d fft as 2D fft and 1d fft to implement distributed fft
        result = cufftPlanMany(&planR2C2D_,
                               rank,
                               &realGridSize[YY],
                               &realGridSizePadded[YY],
                               1,
                               realGridSizePaddedTotal2D,
                               &complexGridSizePadded[YY],
                               1,
                               complexGridSizePaddedTotal2D,
                               CUFFT_R2C,
                               batch);
        handleCufftError(result, "cufftPlanMany 2D R2C plan failure");
        result = cufftSetStream(planR2C2D_, stream);
        handleCufftError(result, "cufftSetStream R2C failure");

        result = cufftPlanMany(&planC2R2D_,
                               rank,
                               &realGridSize[YY],
                               &complexGridSizePadded[YY],
                               1,
                               complexGridSizePaddedTotal2D,
                               &realGridSizePadded[YY],
                               1,
                               realGridSizePaddedTotal2D,
                               CUFFT_C2R,
                               batch);
        handleCufftError(result, "cufftPlanMany 2D C2R plan failure");
        result = cufftSetStream(planC2R2D_, stream);
        handleCufftError(result, "cufftSetStream C2R failure");

        rank   = 1;
        batch  = localy * complexGridSize[ZZ];
        result = cufftPlanMany(&planC2C1D_,
                               rank,
                               &complexGridSize[XX], // 1D C2C part of the R2C
                               &complexGridSizePadded[XX],
                               1,
                               complexGridSizePadded[XX],
                               &complexGridSizePadded[XX],
                               1,
                               complexGridSizePadded[XX],
                               CUFFT_C2C,
                               batch);
        handleCufftError(result, "cufftPlanMany  1D C2C plan failure");
        result = cufftSetStream(planC2C1D_, stream);
        handleCufftError(result, "cufftSetStream C2C failure");

        MPI_Type_contiguous(2, MPI_FLOAT, &complexType_);
        MPI_Type_commit(&complexType_);
    }
    mpiSize_ = mpiSize;
    mpiRank_ = mpiRank;

#endif // CUDA_AWARE_MPI
}

GpuParallel3dFft::~GpuParallel3dFft()
{
    cufftResult_t result;
    if (!pme_gpu_settings(pmeGpu_).useDecomposition)
    {
        result = cufftDestroy(planR2C_);
        handleCufftError(result, "cufftDestroy R2C failure");
        result = cufftDestroy(planC2R_);
        handleCufftError(result, "cufftDestroy C2R failure");
    }
#if CUDA_AWARE_MPI
    else
    {
        result = cufftDestroy(planR2C2D_);
        handleCufftError(result, "cufftDestroy R2C failure");
        result = cufftDestroy(planC2R2D_);
        handleCufftError(result, "cufftDestroy C2R failure");
        result = cufftDestroy(planC2C1D_);
        handleCufftError(result, "cufftDestroy C2C failure");

        MPI_Type_free(&complexType_);

        free(sendCount_);
        free(sendDisp_);
        free(recvCount_);
        free(recvDisp_);
        freeDeviceBuffer(&d_xBlockSizes_);
        freeDeviceBuffer(&d_yBlockSizes_);
        freeDeviceBuffer(&d_s2g0x_);
        freeDeviceBuffer(&d_s2g0y_);

#    if UCX_MPIALLTOALLV_BUG_HACK
        free(sendCountTemp_);
        free(recvCountTemp_);

#    endif // UCX_MPIALLTOALLV_BUG_HACK
    }
#endif // CUDA_AWARE_MPI
}

void GpuParallel3dFft::perform3dFft(gmx_fft_direction dir, CommandEvent* /*timingEvent*/)
{
    cufftResult_t result;
    if (!pme_gpu_settings(pmeGpu_).useDecomposition)
    {
        if (dir == GMX_FFT_REAL_TO_COMPLEX)
        {
            result = cufftExecR2C(planR2C_, realGrid_, complexGrid_);
            handleCufftError(result, "cuFFT R2C execution failure");
        }
        else
        {
            result = cufftExecC2R(planC2R_, complexGrid_, realGrid_);
            handleCufftError(result, "cuFFT C2R execution failure");
        }
    }
#if CUDA_AWARE_MPI
    else
    {
        ivec                          complexGridSizePadded;
        int                           localx, localy;
        const PmeGpuCudaKernelParams* kernelParamsPtr = pmeGpu_->kernelParams.get();
        for (int i = 0; i < DIM; i++)
        {
            complexGridSizePadded[i] = kernelParamsPtr->grid.complexGridSizePadded[i];
        }
        localx = pmeGpu_->common->s2g0x[mpiRank_ + 1] - pmeGpu_->common->s2g0x[mpiRank_];
        localy = h_s2g0y_[mpiRank_ + 1] - h_s2g0y_[mpiRank_];

        if (dir == GMX_FFT_REAL_TO_COMPLEX)
        {
            // halo exchange overlapping region in grid
            pmeGpuHaloExchange();

            // 2D FFT
            result = cufftExecR2C(planR2C2D_, realGrid_, complexGrid_);
            handleCufftError(result, "cuFFT R2C 2D execution failure");
            // Transpose and communicate
            transposeXyzToYzx<true>(
                    complexGrid_, complexGrid2_, localx, complexGridSizePadded[YY], complexGridSizePadded[ZZ]);
            pme_gpu_synchronize(pmeGpu_);

#    if UCX_MPIALLTOALLV_BUG_HACK

            // self copy on the same rank
            cudaMemcpyAsync(complexGrid_ + recvDisp_[mpiRank_],
                            complexGrid2_ + sendDisp_[mpiRank_],
                            recvCount_[mpiRank_] * sizeof(cufftComplex),
                            cudaMemcpyDeviceToDevice,
                            stream_.stream());

            // copy to other ranks. UCX has perf issues if self copies are made in MPI_Alltoallv call
            MPI_Alltoallv(complexGrid2_,
                          sendCountTemp_,
                          sendDisp_,
                          complexType_,
                          complexGrid_,
                          recvCountTemp_,
                          recvDisp_,
                          complexType_,
                          mpi_comm_);

#    else
            // MPI_Alltoallv has perf issues where copy to self is too slow. above implementation takes care of that
            MPI_Alltoallv(complexGrid2_,
                          sendCount_,
                          sendDisp_,
                          complexType_,
                          complexGrid_,
                          recvCount_,
                          recvDisp_,
                          complexType_,
                          mpi_comm_);
#    endif

            // make data in proper layout once different blocks are received from different MPI ranks
            convertBlockedYzxToYzx(complexGrid_,
                                   complexGrid2_,
                                   complexGridSizePadded[XX],
                                   localy,
                                   complexGridSizePadded[ZZ],
                                   d_xBlockSizes_,
                                   d_s2g0x_,
                                   mpiSize_,
                                   xMax_);
            // 1D FFT
            result = cufftExecC2C(planC2C1D_, complexGrid2_, complexGrid_, CUFFT_FORWARD);
            handleCufftError(result, "cuFFT C2C 1D execution failure");
        }
        else
        {
            // 1D FFT
            result = cufftExecC2C(planC2C1D_, complexGrid_, complexGrid2_, CUFFT_INVERSE);
            handleCufftError(result, "cuFFT C2C 1D execution failure");
            // transpose and communicate
            transposeXyzToYzx<false>(
                    complexGrid2_, complexGrid_, complexGridSizePadded[XX], localy, complexGridSizePadded[ZZ]);
            pme_gpu_synchronize(pmeGpu_);

#    if UCX_MPIALLTOALLV_BUG_HACK
            // self copy on the same rank
            cudaMemcpyAsync(complexGrid2_ + recvDisp_[mpiRank_],
                            complexGrid_ + sendDisp_[mpiRank_],
                            recvCount_[mpiRank_] * sizeof(cufftComplex),
                            cudaMemcpyDeviceToDevice,
                            stream_.stream());

            // copy to other ranks. UCX has perf issues if self copies are made in MPI_Alltoallv call
            MPI_Alltoallv(complexGrid_,
                          sendCountTemp_,
                          sendDisp_,
                          complexType_,
                          complexGrid2_,
                          recvCountTemp_,
                          recvDisp_,
                          complexType_,
                          mpi_comm_);

#    else
            MPI_Alltoallv(complexGrid_,
                          sendCount_,
                          sendDisp_,
                          complexType_,
                          complexGrid2_,
                          recvCount_,
                          recvDisp_,
                          complexType_,
                          mpi_comm_);
#    endif

            // make data in proper layout once different blocks are received from different MPI ranks
            convertBlockedXyzToXyz(complexGrid2_,
                                   complexGrid_,
                                   localx,
                                   complexGridSizePadded[YY],
                                   complexGridSizePadded[ZZ],
                                   d_yBlockSizes_,
                                   d_s2g0y_,
                                   mpiSize_,
                                   yMax_);
            // 2D
            result = cufftExecC2R(planC2R2D_, complexGrid_, realGrid_);
            handleCufftError(result, "cuFFT C2R 2D execution failure");

            // halo exchange overlapping region in grid
            pmeGpuHaloExchangeReverse();
        }
    }
#endif
}
