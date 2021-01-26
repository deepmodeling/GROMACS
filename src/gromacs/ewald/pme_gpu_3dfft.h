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
 *  \brief Declares the 3D FFT class for PME.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *  \ingroup module_ewald
 */

#ifndef GMX_EWALD_PME_GPU_3DFFT_H
#define GMX_EWALD_PME_GPU_3DFFT_H

#include "config.h"

#include <vector>

#if GMX_GPU_CUDA
#    include <cufft.h>

#    include "gromacs/gpu_utils/gputraits.cuh"
#    include "gromacs/gpu_utils/devicebuffer_datatype.h"
#    include "gromacs/gpu_utils/device_stream.h"
#    include "gromacs/gpu_utils/hostallocator.h"
#    include "gromacs/utility/gmxmpi.h"
#elif GMX_GPU_OPENCL
#    include <clFFT.h>

#    include "gromacs/gpu_utils/gmxopencl.h"
#    include "gromacs/gpu_utils/gputraits_ocl.h"
#endif

#include "gromacs/fft/fft.h" // for the enum gmx_fft_direction

// Workaround for UCX MPI_Alltoallv poor performance issue with self copy
#define UCX_MPIALLTOALLV_BUG_HACK 1

struct PmeGpu;

/*! \internal \brief
 * A 3D FFT class for performing R2C/C2R transforms
 * \todo Make this class actually parallel over multiple GPUs
 */
class GpuParallel3dFft
{
public:
    /*! \brief
     * Constructs CUDA/OpenCL FFT plans for performing 3D FFT on a PME grid.
     *
     * \param[in] pmeGpu                  The PME GPU structure.
     * \param[in] gridIndex               The index of the grid on which to perform the calculations.
     */
    GpuParallel3dFft(const PmeGpu* pmeGpu, int gridIndex);
    /*! \brief Destroys the FFT plans. */
    ~GpuParallel3dFft();
    /*! \brief Performs the FFT transform in given direction
     *
     * \param[in]  dir           FFT transform direction specifier
     * \param[out] timingEvent   pointer to the timing event where timing data is recorded
     */
    void perform3dFft(gmx_fft_direction dir, CommandEvent* timingEvent);

private:
#if GMX_GPU_CUDA
    cufftHandle   planR2C_;
    cufftHandle   planC2R_;
    cufftReal*    realGrid_;
    cufftComplex* complexGrid_;
    const PmeGpu* pmeGpu_;

    /*! \brief
     * CUDA stream used for PME computation
     */
    const DeviceStream& stream_;

#    if CUDA_AWARE_MPI

    /*! \brief
     * 2D and 1D cufft plans used for distributed fft implementation
     */
    cufftHandle planR2C2D_;
    cufftHandle planC2R2D_;
    cufftHandle planC2C1D_;

    /*! \brief
     * temporary grid used for distributed FFT for out-of-place transpose
     */
    cufftComplex* complexGrid2_;

    /*! \brief
     * MPI complex type
     */
    MPI_Datatype complexType_;

    /*! \brief
     * MPI communicator for PME ranks
     */
    MPI_Comm mpi_comm_;

    /*! \brief
     * total ranks within PME group
     */
    int mpiSize_;

    /*! \brief
     * current local mpi rank within PME group
     */
    int mpiRank_;

    /*! \brief
     * Max local grid size in X-dim (used during transposes in forward pass)
     */
    int xMax_;

    /*! \brief
     * Max local grid size in Y-dim (used during transposes in reverse pass)
     */
    int yMax_;

    /*! \brief
     * device array containing 1D decomposition size in X-dim (forwarad pass)
     */
    DeviceBuffer<int> d_xBlockSizes_;

    /*! \brief
     * device array containing 1D decomposition size in Y-dim (reverse pass)
     */
    DeviceBuffer<int> d_yBlockSizes_;

    /*! \brief
     * device arrays for local interpolation grid start values in X-dim
     * (used during transposes in forward pass)
     */
    DeviceBuffer<int> d_s2g0x_;

    /*! \brief
     * device arrays for local interpolation grid start values in Y-dim
     * (used during transposes in reverse pass)
     */
    DeviceBuffer<int> d_s2g0y_;

    /*! \brief
     * host array containing 1D decomposition size in X-dim (forwarad pass)
     */
    gmx::HostVector<int> h_xBlockSizes_;

    /*! \brief
     * host array containing 1D decomposition size in Y-dim (reverse pass)
     */
    gmx::HostVector<int> h_yBlockSizes_;

    /*! \brief
     * host array for local interpolation grid start values in Y-dim
     */
    gmx::HostVector<int> h_s2g0y_;

    /*! \brief
     * device array big enough to hold grid overlapping region
     * used during grid halo exchange
     */
    DeviceBuffer<float> d_transferGrid_;

    /*! \brief
     * count and displacement arrays used in MPI_Alltoall call
     *
     */
    int *sendCount_, *sendDisp_;
    int *recvCount_, *recvDisp_;

#        if UCX_MPIALLTOALLV_BUG_HACK
    /*! \brief
     * count arrays used in MPI_Alltoall call which has no self copies
     *
     */
    int *sendCountTemp_, *recvCountTemp_;
#        endif

    /*! \brief
     * Exchange grid overlap data with neighboring ranks before Forward FFT
     *
     */
    void pmeGpuHaloExchange();

    /*! \brief
     * Exchange grid overlap data with neighboring ranks after reverse FFT
     *
     */
    void pmeGpuHaloExchangeReverse();

    /*! \brief
     * Merge multiple blocks in YZX layout from different ranks
     *
     * \param[in] arrayIn          Input local grid
     * \param[in] arrayOut         Output local grid in converted layout
     * \param[in] sizeX            Grid size in X-dim.
     * \param[in] sizeY            Grid size in Y-dim.
     * \param[in] sizeZ            Grid size in Z-dim.
     * \param[in] xBlockSizes      Array containing X-block sizes for each rank
     * \param[in] xOffset          Array containing grid offsets for each rank
     * \param[in] numRegions       number of regions correspond to number of PME ranks
     * \param[in] maxRegionSize    max X-block size
     */
    void convertBlockedYzxToYzx(cufftComplex* arrayIn,
                                cufftComplex* arrayOut,
                                int           sizeX,
                                int           sizeY,
                                int           sizeZ,
                                int*          xBlockSizes,
                                int*          xOffset,
                                int           numRegions,
                                int           maxRegionSize);

    /*! \brief
     * Merge multiple blocks in XYZ layout from different ranks
     *
     * \param[in] arrayIn          Input local grid
     * \param[in] arrayOut         Output local grid in converted layout
     * \param[in] sizeX            Grid size in X-dim.
     * \param[in] sizeY            Grid size in Y-dim.
     * \param[in] sizeZ            Grid size in Z-dim.
     * \param[in] yBlockSizes      Array containing Y-block sizes for each rank
     * \param[in] yOffsets         Array containing grid offsets for each rank
     * \param[in] numRegions       number of regions correspond to number of PME ranks
     * \param[in] maxRegionSize    max X-block size
     */
    void convertBlockedXyzToXyz(cufftComplex* arrayIn,
                                cufftComplex* arrayOut,
                                int           sizeX,
                                int           sizeY,
                                int           sizeZ,
                                int*          yBlockSizes,
                                int*          yOffsets,
                                int           numRegions,
                                int           maxRegionSize);


    /*! \brief
     * Converts grid from XYZ to YZX layout in case of forward fft
     * and converts from YZX to XYZ layout in case of reverse fft
     *
     * \tparam[in] forward         Forward pass or reverse pass
     *
     * \param[in] arrayIn          Input local grid
     * \param[in] arrayOut         Output local grid in converted layout
     * \param[in] sizeX            Grid size in X-dim.
     * \param[in] sizeY            Grid size in Y-dim.
     * \param[in] sizeZ            Grid size in Z-dim.
     */
    template<bool forward>
    void transposeXyzToYzx(cufftComplex* arrayIn, cufftComplex* arrayOut, int sizeX, int sizeY, int sizeZ);

#    endif // CUDA_AWARE_MPI

#elif GMX_GPU_OPENCL
    clfftPlanHandle               planR2C_;
    clfftPlanHandle               planC2R_;
    std::vector<cl_command_queue> deviceStreams_;
    cl_mem                        realGrid_;
    cl_mem                        complexGrid_;
#endif
};

#endif
