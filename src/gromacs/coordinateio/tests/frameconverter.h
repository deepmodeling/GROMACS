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
/*!\file
 * \libinternal
 * \brief
 * Helper classes for frameconverter tests
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \inlibraryapi
 * \ingroup module_coordinateio
 */

#ifndef GMX_COORDINATEIO_TESTS_FRAMECONVERTER_H
#define GMX_COORDINATEIO_TESTS_FRAMECONVERTER_H

#include <gtest/gtest.h>

#include "gromacs/coordinateio/iframeconverter.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/smalloc.h"

#include "testutils/testasserts.h"

namespace gmx
{

namespace test
{

class DummyConverter : public IFrameConverter
{
public:
    DummyConverter(FrameConverterFlags flag) : flag_(flag) {}

    DummyConverter(DummyConverter&& old) noexcept = default;

    ~DummyConverter() override {}

    void convertFrame(t_trxframe* /* input */) override {}

    unsigned long guarantee() const override { return convertFlag(flag_); }

private:
    FrameConverterFlags flag_;
};

//! Convenience typedef for dummy module
using DummyConverterPointer = std::unique_ptr<DummyConverter>;

class FrameconverterTestBase : public ::testing::Test
{
public:
    FrameconverterTestBase()
    {
        clear_trxframe(&frame_, true);
        frame_.natoms = 20;
        frame_.x      = x();
        frame_.v      = v();
        frame_.f      = f();
        frame_.bX     = true;
        frame_.bV     = true;
        frame_.bF     = true;
    }

    //! Access coordinate frame.
    const t_trxframe* frame() const { return &frame_; }
    //! Access coordinate pointer.
    rvec* x() { return as_rvec_array(x_.data()); }
    //! Access velocity pointer.
    rvec* v() { return as_rvec_array(v_.data()); }
    //! Access force pointer.
    rvec* f() { return as_rvec_array(f_.data()); }

private:
    //! Coordinate data to use for tests.
    t_trxframe frame_;
    //! Vector for coordinates.
    std::array<RVec, 20> x_;
    //! Vector for velocities.
    std::array<RVec, 20> v_;
    //! Vector for forces.
    std::array<RVec, 20> f_;
};

} // namespace test

} // namespace gmx

#endif
