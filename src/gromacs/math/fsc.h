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
 * \brief FSC routines.
 *
 * \author Christian Blau <blau@kth.se>
 * \ingroup
 */

#ifndef GMX_FSC_H
#define GMX_FSC_H

#include <numeric>
#include <vector>

#include "gromacs/fft/parallel_3dfft.h"
#include "gromacs/math/multidimarray.h"
#include "gromacs/math/functions.h"
#include "gromacs/mdspan/mdspan.h"
#include "gromacs/mdspan/extensions.h"
#include "gromacs/utility/real.h"

namespace gmx
{

struct FourierShellCorrelationCurve
{
    std::vector<real> correlation;
    std::vector<real> count;
};

class FourierTransform
{
public:
    FourierTransform(IVec dataSize)
    {
        MPI_Comm comm[] = { MPI_COMM_NULL, MPI_COMM_NULL };

        gmx_parallel_3dfft_init(&fftSetup_, dataSize, &rdata_, &cdata_, comm, false, 1);

        {
            // parameters only used to satisfy function signature
            ivec localNData;
            ivec offset;
            ivec complexOrder;
            gmx_parallel_3dfft_real_limits(fftSetup_, localNData, offset, rsize_);
            gmx_parallel_3dfft_complex_limits(fftSetup_, complexOrder, localNData, offset, csize_);
        }

        transformInput_.resize(rsize_[0], rsize_[1], rsize_[2]);
        cResult_.resize(csize_[0], csize_[1], csize_[2]);
        cResultTransposed_.resize(csize_[YY], csize_[ZZ], csize_[XX]);
    }

    ~FourierTransform()
    {
        if (fftSetup_)
        {
            gmx_parallel_3dfft_destroy(fftSetup_);
        }
    }

    basic_mdspan<const t_complex, dynamicExtents3D> transform(basic_mdspan<const real, dynamicExtents3D> inputData)
    {

        std::fill(begin(transformInput_), end(transformInput_), 0.);

        for (ptrdiff_t x = 0; x < inputData.extent(XX); ++x)
        {
            for (ptrdiff_t y = 0; y < inputData.extent(YY); ++y)
            {
                std::copy(begin(inputData[x][y]), end(inputData[x][y]),
                          begin(transformInput_.asView()[x][y]));
            }
        }
        std::copy(begin(transformInput_), end(transformInput_), rdata_);

        gmx_parallel_3dfft_execute(fftSetup_, GMX_FFT_REAL_TO_COMPLEX, 0, nullptr);


        std::copy(cdata_, cdata_ + cResultTransposed_.asView().mapping().required_span_size(),
                  begin(cResultTransposed_));

        // cdata has a mangled layout with y,z,x ordering
        for (ptrdiff_t x = 0; x < cResult_.extent(XX); ++x)
        {
            for (ptrdiff_t y = 0; y < cResult_.extent(YY); ++y)
            {
                for (ptrdiff_t z = 0; z < cResult_.extent(ZZ); ++z)
                {
                    cResult_(x, y, z) = cResultTransposed_(y, z, x);
                }
            }
        }


        return cResult_.asConstView();
    }

    RVec complexBasis()
    {
        return { 1._real / rsize_[XX], 1._real / rsize_[YY], 1._real / rsize_[ZZ] };
    }

    IVec realSize() { return rsize_; }

    IVec complexSize() { return csize_; }

private:
    gmx_parallel_3dfft_t                                    fftSetup_ = nullptr;
    ivec                                                    rsize_;
    ivec                                                    csize_;
    real*                                                   rdata_;
    t_complex*                                              cdata_;
    MultiDimArray<std::vector<real>, dynamicExtents3D>      transformInput_;
    MultiDimArray<std::vector<t_complex>, dynamicExtents3D> cResultTransposed_;
    MultiDimArray<std::vector<t_complex>, dynamicExtents3D> cResult_;
};

std::vector<real> fscAverage(const FourierShellCorrelationCurve& curve);

class FourierShellCorrelation
{
public:
    FourierShellCorrelation(basic_mdspan<const real, dynamicExtents3D> referenceDensity,
                            real                                       voxelSize,
                            int                                        numberOfShells) :
        voxelSize_(voxelSize),
        fft_(IVec(2 * referenceDensity.extent(XX),
                  2 * referenceDensity.extent(YY),
                  2 * referenceDensity.extent(ZZ))),
        numberOfShells_(numberOfShells)
    {
        const auto transform = fft_.transform(referenceDensity);
        referenceDensityTransform_.resize(transform.extents());
        std::copy(begin(transform), end(transform), begin(referenceDensityTransform_));

        comparisonTransformNormSquared_.resize(transform.extents());

        referenceTransformNormSquared_.resize(transform.extents());
        std::transform(begin(referenceDensityTransform_), end(referenceDensityTransform_),
                       begin(referenceTransformNormSquared_),
                       [](const t_complex& value) { return cabs2(value); });

        conjugateProduct_.resize(transform.extents());
        fourierSpaceLimits_ =
                BasicVector<ptrdiff_t>(conjugateProduct_.extent(XX), conjugateProduct_.extent(YY),
                                       conjugateProduct_.extent(ZZ));

        maxKDistance_ = std::min({ fft_.complexBasis()[XX] * fourierSpaceLimits_[XX],
                                   fft_.complexBasis()[YY] * fourierSpaceLimits_[YY],
                                   fft_.complexBasis()[ZZ] * fourierSpaceLimits_[ZZ] });

        shellIndex_.resize(transform.extents());

        const RVec fftBasis = fft_.complexBasis();

        // pre-calculate shell index for each point in Fourier-space
        // if number of points in shells is zero for any shell,
        // half the number of shells and redo pre-calculation

        numberOfShells_ *= 2; // avoid if(firstRun) logic
        do
        {
            numberOfShells_ /= 2;

            count_.resize(numberOfShells_);
            std::fill(std::begin(count_), std::end(count_), 0);

            for (ptrdiff_t x = 0; x < fourierSpaceLimits_[XX]; x++)
            {
                for (ptrdiff_t y = 0; y < fourierSpaceLimits_[YY]; y++)
                {
                    /* use symmetry in z-coordinate in Fourier transform
                    values above fourierSpaceLimits_[ZZ] would just be
                    the conjugates. Needs therefore no periodic shift for
                    k-distance */
                    for (ptrdiff_t z = 0; z < fourierSpaceLimits_[ZZ]; z++)
                    {

                        RVec kVector{ fftBasis[XX] * x, fftBasis[YY] * y, fftBasis[ZZ] * z };

                        if (x > fourierSpaceLimits_[XX] / 2)
                        {
                            kVector[XX] = fftBasis[XX] * (fourierSpaceLimits_[XX] - x);
                        }

                        if (y > fourierSpaceLimits_[YY] / 2)
                        {
                            kVector[YY] = fftBasis[YY] * (fourierSpaceLimits_[YY] - y);
                        }

                        const real kDistance = kVector.norm();

                        const int shellIndex =
                                static_cast<int>(std::round(numberOfShells_ * kDistance / maxKDistance_));
                        shellIndex_(x, y, z) = shellIndex;
                        if (shellIndex < numberOfShells_)
                        {
                            ++count_[shellIndex];
                        }
                    }
                }
            }

        } while (std::any_of(std::begin(count_), std::end(count_),
                             [](int value) { return value == 0; }));
    }

    ~FourierShellCorrelation() = default;

    int numberOfShells() const { return numberOfShells_; }

    FourierShellCorrelationCurve fscCurve(basic_mdspan<const real, dynamicExtents3D> comparedDensity)
    {
        const auto comparedDensityTransform = fft_.transform(comparedDensity);
        std::transform(begin(comparedDensityTransform), end(comparedDensityTransform),
                       begin(comparisonTransformNormSquared_),
                       [](const t_complex& value) { return cabs2(value); });

        std::transform(begin(comparedDensityTransform), end(comparedDensityTransform),
                       begin(referenceDensityTransform_), begin(conjugateProduct_),
                       [](const t_complex& comp, const t_complex& ref) {
                           return comp.re * ref.re + comp.im * ref.im;
                       });

        FourierShellCorrelationCurve fscCurve;
        fscCurve.count.resize(numberOfShells_);
        std::copy(std::begin(count_), std::end(count_), std::begin(fscCurve.count));
        fscCurve.correlation.resize(numberOfShells_);

        std::vector<real> fscNominator(numberOfShells_);
        std::vector<real> fscDenominatorReference(numberOfShells_);
        std::vector<real> fscDenominatorComparison(numberOfShells_);


        for (ptrdiff_t x = 0; x < fourierSpaceLimits_[XX]; x++)
        {
            for (ptrdiff_t y = 0; y < fourierSpaceLimits_[YY]; y++)
            {
                /* use symmetry in z-coordinate in Fourier transform
                 values above fourierSpaceLimits_[ZZ] would just be
                 the conjugates. Needs therefore no periodic shift for
                 k-distance */
                for (ptrdiff_t z = 0; z < fourierSpaceLimits_[ZZ]; z++)
                {
                    const int shellIndex = shellIndex_(x, y, z);
                    if (shellIndex < numberOfShells_)
                    {
                        fscNominator[shellIndex] += conjugateProduct_(x, y, z);
                        fscDenominatorReference[shellIndex] += referenceTransformNormSquared_(x, y, z);
                        fscDenominatorComparison[shellIndex] += comparisonTransformNormSquared_(x, y, z);

                        // correct for double accounting of points on
                        // z=0 plane when using the fourier transform symmetry
                        if (z == 0)
                        {
                            fscNominator[shellIndex] -= 0.5 * conjugateProduct_(x, y, z);
                            fscDenominatorReference[shellIndex] -=
                                    0.5 * referenceTransformNormSquared_(x, y, z);
                            fscDenominatorComparison[shellIndex] -=
                                    0.5 * comparisonTransformNormSquared_(x, y, z);
                        }
                    }
                }
            }
        }

        std::vector<real> fscDenominator(numberOfShells_);

        std::transform(std::begin(fscDenominatorComparison), std::end(fscDenominatorComparison),
                       std::begin(fscDenominatorReference), std::begin(fscDenominator),
                       [](real denominatorComparison, real denominatorReference) {
                           return sqrt(denominatorComparison * denominatorReference);
                       });

        std::transform(std::begin(fscNominator), std::end(fscNominator), std::begin(fscDenominator),
                       std::begin(fscCurve.correlation),
                       [](real nominator, real denominator) { return nominator / denominator; });

        return fscCurve;
    }

    real spacing() { return maxKDistance_ / (voxelSize_ * numberOfShells_); }


private:
    MultiDimArray<std::vector<int>, dynamicExtents3D>       shellIndex_;
    MultiDimArray<std::vector<real>, dynamicExtents3D>      referenceTransformNormSquared_;
    MultiDimArray<std::vector<real>, dynamicExtents3D>      comparisonTransformNormSquared_;
    MultiDimArray<std::vector<real>, dynamicExtents3D>      conjugateProduct_;
    MultiDimArray<std::vector<t_complex>, dynamicExtents3D> referenceDensityTransform_;
    real                                                    voxelSize_;
    std::vector<int>                                        count_;
    FourierTransform                                        fft_;
    ptrdiff_t                                               numberOfShells_;
    real                                                    maxKDistance_;
    BasicVector<ptrdiff_t>                                  fourierSpaceLimits_;
};

} // namespace gmx

#endif
