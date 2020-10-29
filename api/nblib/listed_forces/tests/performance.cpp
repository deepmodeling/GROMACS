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
 * \brief
 * This implements basic nblib utility tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include <chrono>
#include <iostream>
#include <numeric>

#include "nblib/listed_forces/calculator.h"
#include "nblib/listed_forces/gmxcalculator.h"
#include "nblib/listed_forces/tests/linear_chain_input.hpp"
#include "nblib/listed_forces/tests/poly_dimethyl_butene_input.hpp"
#include "nblib/listed_forces/traits.h"

namespace nblib
{
namespace test
{
namespace
{

//! \brief compare whether the forces between a pair of gmx/nblib datasets match up
template<class Data>
void compareForces(const Data& nblibData, const Data& gmxData, int numThreads)
{
    ListedForceCalculator nblibCalculator(nblibData.interactions, nblibData.x.size(), numThreads,
                                          *nblibData.box);
    ListedGmxCalculator gmxCalculator(gmxData.interactions, gmxData.x.size(), numThreads, *gmxData.box);

    ListedForceCalculator::EnergyType nblibEnergies;
    std::vector<gmx::RVec>            nblibForces(nblibData.x.size(), Vec3{ 0, 0, 0 });
    nblibCalculator.compute(nblibData.x, nblibForces, nblibEnergies);

    ListedForceCalculator::EnergyType gmxEnergies;
    std::vector<gmx::RVec>            gmxForces(gmxData.x.size(), Vec3{ 0, 0, 0 });
    gmxCalculator.compute(gmxData.x, gmxForces, gmxEnergies);

    for (int i = 0; i < nblibForces.size(); ++i)
    {
        for (int m = 0; m < dimSize; ++m)
        {
            if (std::abs(nblibForces[i][m] - gmxForces[i][m]) > 1e-4)
            {
                std::cout << "Force @" << i << "." << m << " differ: " << nblibForces[i][m] << " "
                          << gmxForces[i][m] << std::endl;
            }
        }
    }

    std::cout << "Force values compared" << std::endl;
}

//! \brief benchmark fixture for bonded calculators
template<class Calculator, class DataObject>
class listedBenchmark
{
public:
    listedBenchmark(const DataObject& data, int reps_, int nThr) :
        data_(data),
        numThreads(nThr),
        reps(reps_),
        forceBuffer_(data_.x.size()),
        lfCalculator(data_.interactions, data_.x.size(), numThreads, *data_.box),
        calculatorEnergies{ 0 }
    {
    }

    void operator()()
    {
        for (int i = 0; i < reps; ++i)
        {
            lfCalculator.compute(data_.x, forceBuffer_, calculatorEnergies);
        }
    }

    double energy()
    {
        return std::accumulate(begin(calculatorEnergies), end(calculatorEnergies), 0.0);
    }

private:
    int numThreads;
    int reps;

    DataObject                        data_;
    Calculator                        lfCalculator;
    std::vector<Vec3>                 forceBuffer_;
    ListedForceCalculator::EnergyType calculatorEnergies;
};

} // namespace
} // namespace test
} // namespace nblib


template<class F>
double timeit(F&& functionToTime)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    functionToTime();
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(t2 - t1).count();
}

template<class Data>
void testNblib(const Data& data, int reps, int nThreads)
{
    nblib::test::listedBenchmark<nblib::ListedForceCalculator, Data> nblibRunner(data, reps, nThreads);

    [[maybe_unused]] double warmup  = timeit(nblibRunner);
    double                  elapsed = timeit(nblibRunner);
    std::cout << "nblib time elapsed " << elapsed << ", energy: " << nblibRunner.energy() << std::endl;
}

template<class Data>
void testGmx(const Data& data, int reps, int nThreads)
{
    nblib::test::listedBenchmark<nblib::ListedGmxCalculator, Data> gmxRunner(data, reps, nThreads);

    [[maybe_unused]] double warmup  = timeit(gmxRunner);
    double                  elapsed = timeit(gmxRunner);
    std::cout << "gmx time elapsed " << elapsed << ", energy: " << gmxRunner.energy() << std::endl;
}


int main()
{
    int nParticles = 50003;

    nblib::LinearChainData nblibData(nParticles), gmxData(nParticles);
    // nblibData.createAggregates();
    nblib::test::compareForces(nblibData, gmxData, 4);

    testNblib(nblibData, 100, 4);
    testGmx(gmxData, 100, 4);

    // Note: this test cases needs double precision for forces to match precisely
    nblib::PolyDimethylButene nbPolyData(100), gmxPolyData(100);
    // nbPolyData.createAggregates();
    nblib::test::compareForces(nbPolyData, gmxPolyData, 4);

    testNblib(nbPolyData, 1, 4);
    testGmx(gmxPolyData, 1, 4);
}
