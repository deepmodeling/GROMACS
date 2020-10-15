/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2016 by the GROMACS development team.
 * Copyright (c) 2017,2019,2020, by the GROMACS development team, led by
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
#ifndef GMX_HARDWARE_HWINFO_H
#define GMX_HARDWARE_HWINFO_H

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gromacs/hardware/device_management.h"
#include "gromacs/utility/basedefinitions.h"

namespace gmx
{
class CpuInfo;
class HardwareTopology;
enum class SimdType;
} // namespace gmx
struct DeviceInformation;

/*! \brief Contains summary information obtained from the
 * hardware detection.
 *
 * The data is computed by reduction over all ranks of this
 * simulation. */
struct HardwareSummaryInformation
{
    //! Number of physical nodes used in the simulation
    int nphysicalnode;
    //! Sum of number of cores over all ranks of this simulation, can be 0
    int ncore_tot;
    //! Min number of cores over all ranks of this simulation
    int ncore_min;
    //! Max number of cores over all ranks of this simulation
    int ncore_max;
    //! Sum of number of hwthreads over all ranks of this simulation
    int nhwthread_tot;
    //! Min number of hwthreads over all ranks of this simulation
    int nhwthread_min;
    //! Max number of hwthreads over all ranks of this simulation
    int nhwthread_max;
    //! Sum of number of GPUs over all ranks of this simulation
    int ngpu_compatible_tot;
    //! Min number of GPUs over all ranks of this simulation
    int ngpu_compatible_min;
    //! Max number of GPUs over all ranks of this simulation
    int ngpu_compatible_max;
    //! Highest SIMD instruction set supported by all ranks
    gmx::SimdType minimumDetectedSimdSupport;
    //! Highest SIMD instruction set supported by at least one rank
    gmx::SimdType maximumDetectedSimdSupport;
    //! True if all ranks have the same type(s) and order of GPUs
    bool bIdenticalGPUs;
    /*! \brief True when at least one CPU in any of the nodes of
     * ranks of this simulation is AMD Zen of the first
     * generation. */
    bool haveAmdZen1Cpu;
};

/*! \brief Hardware information structure with CPU and GPU information.
 *
 * This structure may only contain data that is
 * valid over the whole process (i.e. must be able to
 * be shared among all threads, particularly the ranks
 * of thread-MPI)
 *
 * \todo Make deviceInfoList something like a
 * std::vector<std::variant<CudaDeviceInformation,
 *   OpenCLDeviceInformation, SyclDeviceInformation>>
 * so that gmx_hw_info_t becomes copyable. Then setup code
 * will have an easier time passing the results of hardware
 * detection to the runner.
 */
struct gmx_hw_info_t
{
    ~gmx_hw_info_t();

    //! Information about CPU capabilities on this physical node
    std::unique_ptr<gmx::CpuInfo> cpuInfo;
    //! Information about hardware topology on this phyiscal node
    std::unique_ptr<gmx::HardwareTopology> hardwareTopology;
    //! Information about GPUs detected on this physical node
    std::vector<std::unique_ptr<DeviceInformation>> deviceInfoList;
    //! Summary information across all ranks of this simulation.
    HardwareSummaryInformation summaryInformation;
    //! Any warnings to log when that is possible.
    std::optional<std::string> hardwareDetectionWarnings_;
};


/* The options for the thread affinity setting, default: auto */
enum class ThreadAffinity
{
    Select,
    Auto,
    On,
    Off,
    Count
};

/*! \internal \brief Threading and GPU options, can be set automatically or by the user
 *
 * \todo During mdrunner(), if the user has left any of these values
 * at their defaults (which tends to mean "choose automatically"),
 * then those values are over-written with the result of such
 * automation. This creates problems for the subsequent code in
 * knowing what was done, why, and reporting correctly to the
 * user. Find a way to improve this.
 */
struct gmx_hw_opt_t
{
    //! Total number of threads requested (thread-MPI + OpenMP).
    int totalThreadsRequested = 0;
    //! Number of thread-MPI threads requested.
    int nthreads_tmpi = 0;
    //! Number of OpenMP threads requested.
    int nthreads_omp = 0;
    //! Number of OpenMP threads to use on PME_only ranks.
    int nthreads_omp_pme = 0;
    //! Thread affinity switch, see enum above.
    ThreadAffinity threadAffinity = ThreadAffinity::Select;
    //! Logical core pinning stride.
    int core_pinning_stride = 0;
    //! Logical core pinning offset.
    int core_pinning_offset = 0;
    //! Empty, or a string provided by the user declaring (unique) GPU IDs available for mdrun to use.
    std::string gpuIdsAvailable = "";
    //! Empty, or a string provided by the user mapping GPU tasks to devices.
    std::string userGpuTaskAssignment = "";
    //! Tells whether mdrun is free to choose the total number of threads (by choosing the number of OpenMP and/or thread-MPI threads).
    bool totNumThreadsIsAuto;
};

#endif
