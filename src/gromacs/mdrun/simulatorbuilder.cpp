/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019-2020, by the GROMACS development team, led by
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

/*! \internal
 * \brief Defines the simulator builder for mdrun
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrun
 */
#include "simulatorbuilder.h"

#include <memory>
#include <gromacs/modularsimulator/modularsimulator.h>

#include "simulatorbuilder_detail.h"
#include "legacysimulator.h"

namespace gmx
{

SimulatorBuilderImplementation::~SimulatorBuilderImplementation() = default;

SimulatorBuilder::SimulatorBuilder(std::unique_ptr<SimulatorBuilderImplementation> impl) :
    impl_(std::move(impl))
{
}

//! Build a Simulator object
std::unique_ptr<ISimulator> SimulatorBuilder::build(FILE*                    fplog,
                                                    t_commrec*               cr,
                                                    const gmx_multisim_t*    ms,
                                                    const MDLogger&          mdlog,
                                                    int                      nfile,
                                                    const t_filenm*          fnm,
                                                    const gmx_output_env_t*  oenv,
                                                    const MdrunOptions&      mdrunOptions,
                                                    StartingBehavior         startingBehavior,
                                                    gmx_vsite_t*             vsite,
                                                    Constraints*             constr,
                                                    gmx_enfrot*              enforcedRotation,
                                                    BoxDeformation*          deform,
                                                    IMDOutputProvider*       outputProvider,
                                                    const MdModulesNotifier& mdModulesNotifier,
                                                    t_inputrec*              inputrec,
                                                    ImdSession*              imdSession,
                                                    pull_t*                  pull_work,
                                                    t_swap*                  swap,
                                                    gmx_mtop_t*              top_global,
                                                    t_fcdata*                fcd,
                                                    t_state*                 state_global,
                                                    ObservablesHistory*      observablesHistory,
                                                    MDAtoms*                 mdAtoms,
                                                    t_nrnb*                  nrnb,
                                                    gmx_wallcycle*           wcycle,
                                                    t_forcerec*              fr,
                                                    gmx_enerdata_t*          enerd,
                                                    gmx_ekindata_t*          ekind,
                                                    MdrunScheduleWorkload*   runScheduleWork,
                                                    const ReplicaExchangeParameters& replExParams,
                                                    gmx_membed_t*                    membed,
                                                    gmx_walltime_accounting* walltime_accounting,
                                                    std::unique_ptr<StopHandlerBuilder> stopHandlerBuilder,
                                                    bool                                doRerun)
{
    GMX_ASSERT(impl_, "Bug: Builder not properly initialized.");
    return impl_->build(fplog, cr, ms, mdlog, nfile, fnm, oenv, mdrunOptions, startingBehavior, vsite,
                        constr, enforcedRotation, deform, outputProvider, mdModulesNotifier, inputrec,
                        imdSession, pull_work, swap, top_global, fcd, state_global, observablesHistory,
                        mdAtoms, nrnb, wcycle, fr, enerd, ekind, runScheduleWork, replExParams,
                        membed, walltime_accounting, std::move(stopHandlerBuilder), doRerun);
}

std::unique_ptr<SimulatorBuilderImplementation> getSimulatorBuilderImpl(SimulatorVersion simulatorChoice);

std::unique_ptr<SimulatorBuilderImplementation> getSimulatorBuilderImpl(SimulatorVersion simulatorChoice)
{
    if (simulatorChoice == SimulatorVersion::LegacySimulator)
    {
        return std::make_unique<LegacySimulatorBuilder>();
    }
    else
    {
        GMX_ASSERT(simulatorChoice == SimulatorVersion::ModularSimulator,
                   "Logic error: This implementation assumes only two possible code paths.");
        return std::make_unique<ModularSimulator::Builder>();
    }
}

SimulatorBuilder getSimulatorBuilder(SimulatorVersion simulatorChoice)
{
    return SimulatorBuilder(getSimulatorBuilderImpl(simulatorChoice));
}

} // end namespace gmx
