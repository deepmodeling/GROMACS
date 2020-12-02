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

/*! \file
 * \brief Test gmxapi::Context
 *
 * Provides additional test coverage of template headers only used by client code.
 */

#include "config.h"

#include <gtest/gtest.h>

#include "gromacs/utility/gmxmpi.h"

#include "gmxapi/context.h"
#include "gmxapi/mpi/gmxapi_mpi.h"


namespace gmxapi
{

namespace testing
{

namespace
{

TEST(GmxApiMpiTest, AllContext)
{
    // Default Implicit COMM_WORLD for MPI builds.
    auto context = createContext();
}

#if GMX_LIB_MPI
TEST(GmxApiMpiTest, NullContext)
{
    // Explicit COMM_NULL is not supported.
    EXPECT_ANY_THROW(assignResource(MPI_COMM_NULL));
}

TEST(GmxApiMpiTest, MpiWorldContext)
{
    // Note that this test is only compiled when GMX_MPI is enabled for the
    // build tree, so we cannot unit test the behavior of non-MPI GROMACS
    // provided with MPI-enabled Context. For that, we defer to the Python
    // package testing.
    // Note also that the code should look the same for tMPI or regular MPI.

    // Explicit COMM_WORLD.
    auto resources = assignResource(MPI_COMM_WORLD);
    EXPECT_TRUE(resources->size() != 0);

    // Store the rank for debugging convenience.
    [[maybe_unused]] auto rank = resources->rank();

    auto context = createContext(*resources);
}

TEST(GmxApiMpiTest, MpiSplitContext)
{
    // Explicit sub-communicator.
    MPI_Comm communicator = MPI_COMM_NULL;
    int      rank{ 0 };
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Run each rank as a separate ensemble member.
    MPI_Comm_split(MPI_COMM_WORLD, rank, rank, &communicator);
    auto context = createContext(*assignResource(communicator));
}
#endif

} // end anonymous namespace

} // end namespace testing

} // end namespace gmxapi
