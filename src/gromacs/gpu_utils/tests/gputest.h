/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2017,2018,2019,2020, by the GROMACS development team, led by
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
 * Declares test fixture testing GPU utility components.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 */
#ifndef GMX_GPU_UTILS_TESTS_GPUTEST_H
#define GMX_GPU_UTILS_TESTS_GPUTEST_H

#include "gmxpre.h"

#include <gtest/gtest.h>

#include "gromacs/hardware/devices_manager.h"

struct DeviceInformation;
class DevicesManager;

namespace gmx
{
namespace test
{

class GpuTest : public ::testing::Test
{
public:
    //! Information about GPUs that are present.
    DevicesManager deviceManager_;
    //! Contains the IDs of all compatible GPUs
    std::vector<int> compatibleGpuIds_;

    GpuTest();
    ~GpuTest() override;
    //! Return whether compatible GPUs were found
    bool haveCompatibleGpus() const;
    //! Return a vector of handles, each to a device info for a compatible GPU.
    std::vector<const DeviceInformation*> getDeviceInfos() const;
};

} // namespace test
} // namespace gmx

#endif
