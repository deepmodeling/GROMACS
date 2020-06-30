/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020, by the GROMACS development team, led by
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
 * \brief Implements GPU Force Reduction using CUDA
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_mdlib
 */

#include "gmxpre.h"

#include "gpuforcereduction_impl.cuh"

#include <stdio.h>

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.cuh"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/gpu_utils/vectype_ops.cuh"
#include "gromacs/mdlib/gpuforcereduction.h"
#include "gromacs/utility/gmxassert.h"

// Maximum number of forces to be added
#define MAXFORCES 3

namespace gmx
{

constexpr static int c_threadsPerBlock = 128;

/* \brief force data required on device, to be passed to kernel as a parameter */
struct deviceForceData
{
    float3*     gm_f[MAXFORCES];
    ForceFormat format[MAXFORCES];
    int*        gm_cell[MAXFORCES];
};

typedef struct deviceForceData deviceForceData_t;


template<bool accumulateForce>
static __global__ void reduceKernel(const deviceForceData_t __restrict__ forceToAddData,
                                    float3*   gm_fTotal,
                                    const int numAtoms,
                                    const int numForces)
{

    // map particle-level parallelism to 1D CUDA thread and block index
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // perform addition for each particle
    if (threadIndex < numAtoms)
    {

        float3* gm_fDest = &gm_fTotal[threadIndex];
        float3  temp;

        // Accumulate or set first force
        int i = (forceToAddData.format[0] == ForceFormat::Nbat) ? forceToAddData.gm_cell[0][threadIndex]
                                                                : threadIndex;
        if (accumulateForce)
        {
            temp = *gm_fDest;
            temp += forceToAddData.gm_f[0][i];
        }
        else
        {
            temp = forceToAddData.gm_f[0][i];
        }

        // Accumulate any additional forces
        for (int iForce = 1; iForce < numForces; iForce++)
        {
            i = (forceToAddData.format[iForce] == ForceFormat::Nbat)
                        ? forceToAddData.gm_cell[iForce][threadIndex]
                        : threadIndex;
            temp += forceToAddData.gm_f[iForce][i];
        }

        *gm_fDest = temp;
    }
    return;
}

GpuForceReduction::Impl::Impl(const DeviceContext& deviceContext, const DeviceStream& deviceStream) :
    deviceContext_(deviceContext),
    deviceStream_(deviceStream){};

void GpuForceReduction::Impl::setBase(GpuForceForReduction_t forceReductionBase)
{
    GMX_ASSERT((forceReductionBase.forcePtr != nullptr),
               "Input force reduction object has no data");
    baseForce_ = forceReductionBase;
};

void GpuForceReduction::Impl::setNumAtoms(const int numAtoms)
{
    numAtoms_ = numAtoms;
};

void GpuForceReduction::Impl::setAtomStart(const int atomStart)
{
    atomStart_ = atomStart;
};

void GpuForceReduction::Impl::registerForce(const GpuForceForReduction_t forceToAdd)
{

    GMX_ASSERT((forceToAdd.forcePtr != nullptr), "Input force reduction object has no data");
    GMX_ASSERT((baseForce_.forceFormat == ForceFormat::Rvec),
               "Only Rvec format is supported for base of force reduction");

    forceToAddList_.push_back(forceToAdd);

    GMX_RELEASE_ASSERT((forceToAddList_.size() <= 3),
                       "A maximum of 3 forces are supported in the GPU force reduction");
};

void GpuForceReduction::Impl::addDependency(GpuEventSynchronizer* const dependency)
{
    dependencyList_.push_back(dependency);
}

void GpuForceReduction::Impl::setAccumulate(const bool accumulate)
{
    accumulate_ = accumulate;
}

void GpuForceReduction::Impl::setCell(const int* cell)
{
    cell_ = cell;
    reallocateDeviceBuffer(&d_cell_, atomStart_ + numAtoms_, &cellSize_, &cellSizeAlloc_, deviceContext_);
    copyToDeviceBuffer(&d_cell_, cell_, 0, atomStart_ + numAtoms_, deviceStream_,
                       GpuApiCallBehavior::Async, nullptr);
}

void GpuForceReduction::Impl::setCompletionMarker(GpuEventSynchronizer* completionMarker)
{
    completionMarker_ = completionMarker;
}

void GpuForceReduction::Impl::clearDependencies()
{
    dependencyList_.clear();
}

void GpuForceReduction::Impl::popForce()
{
    forceToAddList_.pop_back();
}

void GpuForceReduction::Impl::apply()
{

    if (numAtoms_ == 0)
        return;

    // Enqueue wait on all dependencies passed
    for (auto const synchronizer : dependencyList_)
    {
        synchronizer->enqueueWaitEvent(deviceStream_);
    }

    // Populate data force data struct to be passed as an argument to kernel
    deviceForceData_t d_forceToAddData;
    int               iForce = 0;
    for (auto const forceToAdd : forceToAddList_)
    {
        int forceOffset = (forceToAdd.forceFormat == ForceFormat::Rvec) ? atomStart_ : 0;
        d_forceToAddData.gm_f[iForce]   = &(static_cast<float3*>(forceToAdd.forcePtr))[forceOffset];
        d_forceToAddData.format[iForce] = forceToAdd.forceFormat;
        if (forceToAdd.forceFormat == ForceFormat::Nbat)
        {
            d_forceToAddData.gm_cell[iForce] = &d_cell_[atomStart_];
        }
        iForce++;
    }

    // Configure and launch kernel
    KernelLaunchConfig config;
    config.blockSize[0]     = c_threadsPerBlock;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = ((numAtoms_ + 1) + c_threadsPerBlock - 1) / c_threadsPerBlock;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    auto      kernelFn       = accumulate_ ? reduceKernel<true> : reduceKernel<false>;
    float3*   d_fTotal       = &(static_cast<float3*>(baseForce_.forcePtr))[atomStart_];
    const int numForcesToAdd = forceToAddList_.size();

    const auto kernelArgs = prepareGpuKernelArguments(kernelFn, config, &d_forceToAddData,
                                                      &d_fTotal, &numAtoms_, &numForcesToAdd);

    launchGpuKernel(kernelFn, config, deviceStream_, nullptr, "Force Reduction", kernelArgs);

    // Mark that kernel has been launched
    if (completionMarker_ != nullptr)
    {
        completionMarker_->markEvent(deviceStream_);
    }
}

GpuForceReduction::Impl::~Impl(){};

GpuForceReduction::GpuForceReduction(const DeviceContext& deviceContext, const DeviceStream& deviceStream) :
    impl_(new Impl(deviceContext, deviceStream))
{
}

void GpuForceReduction::setBase(GpuForceForReduction_t forceReductionBase)
{
    impl_->setBase(forceReductionBase);
}

void GpuForceReduction::setNumAtoms(const int numAtoms)
{
    impl_->setNumAtoms(numAtoms);
}

void GpuForceReduction::setAtomStart(const int atomStart)
{
    impl_->setAtomStart(atomStart);
}

void GpuForceReduction::registerForce(const GpuForceForReduction_t forceToAdd)
{
    impl_->registerForce(forceToAdd);
}

void GpuForceReduction::addDependency(GpuEventSynchronizer* const dependency)
{
    impl_->addDependency(dependency);
}

void GpuForceReduction::setAccumulate(const bool accumulate)
{
    impl_->setAccumulate(accumulate);
}

void GpuForceReduction::setCell(const int* cell)
{
    impl_->setCell(cell);
}

void GpuForceReduction::setCompletionMarker(GpuEventSynchronizer* completionMarker)
{
    impl_->setCompletionMarker(completionMarker);
}

void GpuForceReduction::clearDependencies()
{
    impl_->clearDependencies();
}

void GpuForceReduction::popForce()
{
    impl_->popForce();
}

void GpuForceReduction::apply()
{
    impl_->apply();
}

GpuForceReduction::~GpuForceReduction() = default;

} // namespace gmx
