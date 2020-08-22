/*
 * This file is part of the GROMACS molecular simulation package.
 *
<<<<<<< HEAD
<<<<<<<< HEAD:src/gromacs/hardware/device_management.cpp
 * Copyright (c) 2012,2013,2014,2015,2017 The GROMACS development team.
 * Copyright (c) 2018,2019,2020, by the GROMACS development team, led by
========
 * Copyright (c) 2019,2020, by the GROMACS development team, led by
>>>>>>>> 358555b33bb23e9f3c1b9b9498bdb1d20a94d840:src/gromacs/gpu_utils/gpu_testutils.cpp
=======
 * Copyright (c) 2012,2013,2014,2015,2017 The GROMACS development team.
 * Copyright (c) 2018,2019,2020, by the GROMACS development team, led by
>>>>>>> 358555b33bb23e9f3c1b9b9498bdb1d20a94d840
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
<<<<<<< HEAD
 *  \brief Defines the CPU stubs for the DevicesManager class.
=======
 *  \brief Defines the CPU stubs for the device management.
>>>>>>> 358555b33bb23e9f3c1b9b9498bdb1d20a94d840
 *
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_hardware
 */
#include "gmxpre.h"

#include "device_management.h"

#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/utility/fatalerror.h"

std::vector<std::unique_ptr<DeviceInformation>> findDevices()
{
    return {};
}

void setDevice(const DeviceInformation& /* deviceInfo */) {}

void freeDevice(DeviceInformation* /* deviceInfo */) {}

std::string getDeviceInformationString(const DeviceInformation& /* deviceInfo */)
{
    gmx_fatal(FARGS, "Device information requested in CPU build.");
}

bool isDeviceDetectionFunctional(std::string* /* errorMessage */)
{
    return false;
}
