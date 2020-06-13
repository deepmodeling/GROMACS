/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018,2019,2020, by the GROMACS development team, led by
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
 * \brief SHAKE and LINCS tests header.
 *
 * Contains description and constructor for the test data accumulating object,
 * declares CPU- and GPU-based functions used to apply SHAKE or LINCS on the
 * test data.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 * \ingroup module_mdlib
 */

#ifndef GMX_MDLIB_TESTS_CONSTRTESTRUNNERS_H
#define GMX_MDLIB_TESTS_CONSTRTESTRUNNERS_H

#include "constrtestdata.h"

#include <gtest/gtest.h>

#include "testutils/testhardwarecontext.h"

struct t_pbc;

namespace gmx
{
namespace test
{

enum class ConstraintsAlgorithm : int
{
    Shake = 0,
    Lincs = 1,
    Count
};

/*! \brief Apply SHAKE constraints to the test data.
 */
void applyShake(ConstraintsTestData* testData, t_pbc pbc);
/*! \brief Apply LINCS constraints to the test data.
 */
void applyLincs(ConstraintsTestData* testData, t_pbc pbc);
/*! \brief
 * Initialize and apply LINCS constraints on GPU.
 *
 * \param[in] testData             Test data structure.
 * \param[in] pbc                  Periodic boundary data.
 * \param[in] testHardwareContext  Test herdware environment to get DeviceContext and DeviceStream from.
 */
void applyLincsGpu(ConstraintsTestData* testData, t_pbc pbc, TestHardwareContext* testHardwareContext);


class ConstraintsTestRunner
{
public:
    ConstraintsTestRunner(ConstraintsAlgorithm algorithm, TestHardwareContext* testHardwareContext) :
        algorithm_(algorithm),
        testHardwareContext_(testHardwareContext)
    {
    }
    void applyConstraints(ConstraintsTestData* testData, t_pbc pbc)
    {
        switch (testHardwareContext_->codePath())
        {
            case CodePath::CPU:
                switch (algorithm_)
                {
                    case ConstraintsAlgorithm::Shake: return applyShake(testData, pbc);

                    case ConstraintsAlgorithm::Lincs: return applyLincs(testData, pbc);

                    default:
                        FAIL() << "Only SHAKE and LINCS are supported as CPU runners for the "
                                  "constraints tests.";
                }
            case CodePath::GPU:
                switch (algorithm_)
                {
                    case ConstraintsAlgorithm::Lincs: return applyLincs(testData, pbc);

                    default:
                        FAIL() << "Only LINCS is supported as a GPU runner for the constraints "
                                  "tests.";
                }
            default: FAIL() << "Unknown code path: can only be CPU or GPU.";
        }
    }
    std::string name()
    {
        switch (testHardwareContext_->codePath())
        {
            case CodePath::CPU:
                switch (algorithm_)
                {
                    case ConstraintsAlgorithm::Shake: return "SHAKE";

                    case ConstraintsAlgorithm::Lincs: return "LINCS";

                    default: return "Unsupported";
                }
            case CodePath::GPU:
                switch (algorithm_)
                {
                    case ConstraintsAlgorithm::Lincs: return "LINCS_GPU";

                    default: return "Unsupported";
                }
            default: return "Unsupported";
        }
    }

private:
    ConstraintsAlgorithm algorithm_;
    //! Pointer to the global test hardware context (if on GPU)
    TestHardwareContext* testHardwareContext_;
};


} // namespace test
} // namespace gmx

#endif // GMX_MDLIB_TESTS_CONSTRTESTRUNNERS_H
