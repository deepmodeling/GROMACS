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
 * \brief
 * Tests for constant pH parameter objects.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gromacs/applied_forces/constantpHparameters.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/options/options.h"
#include "gromacs/options/treesupport.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreemdpwriter.h"
#include "gromacs/utility/keyvaluetreetransform.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringcompare.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/testasserts.h"
#include "testutils/testmatchers.h"

namespace gmx
{

namespace
{

TEST(LambdaAtomCollectionTest, EqualCheckWorks)
{
    LambdaAtomCollection collection1;
    {
        collection1.name  = "Foo collection";
        collection1.atoms = { 1, 2, 3, 5, 6 };
    }
    LambdaAtomCollection collection2;
    {
        collection2.name  = "Bar collection";
        collection2.atoms = { 2, 3, 4, 5, 6 };
    }
    EXPECT_FALSE(collection1 == collection2);
    collection2.atoms = collection1.atoms;
    EXPECT_FALSE(collection1 == collection2);
    collection2.name = collection1.name;
    EXPECT_TRUE(collection1 == collection2);
}

} // namespace

} // namespace gmx
