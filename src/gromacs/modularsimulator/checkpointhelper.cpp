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
 * \brief Defines the checkpoint helper for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "checkpointhelper.h"

#include "gromacs/domdec/domdec.h"
#include "gromacs/mdlib/mdoutf.h"
#include "gromacs/mdtypes/checkpointdata.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/energyhistory.h"
#include "gromacs/mdtypes/observableshistory.h"
#include "gromacs/mdtypes/pullhistory.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/utility/stringutil.h"

#include "trajectoryelement.h"

namespace gmx
{
CheckpointHelper::CheckpointHelper(std::vector<std::tuple<std::string, ICheckpointHelperClient*>>&& clients,
                                   std::unique_ptr<CheckpointHandler> checkpointHandler,
                                   int                                initStep,
                                   TrajectoryElement*                 trajectoryElement,
                                   int                                globalNumAtoms,
                                   FILE*                              fplog,
                                   t_commrec*                         cr,
                                   ObservablesHistory*                observablesHistory,
                                   gmx_walltime_accounting*           walltime_accounting,
                                   t_state*                           state_global,
                                   bool                               writeFinalCheckpoint) :
    clients_(std::move(clients)),
    checkpointHandler_(std::move(checkpointHandler)),
    initStep_(initStep),
    lastStep_(-1),
    globalNumAtoms_(globalNumAtoms),
    writeFinalCheckpoint_(writeFinalCheckpoint),
    trajectoryElement_(trajectoryElement),
    localState_(nullptr),
    fplog_(fplog),
    cr_(cr),
    observablesHistory_(observablesHistory),
    walltime_accounting_(walltime_accounting),
    state_global_(state_global)
{
    if (DOMAINDECOMP(cr))
    {
        localState_ = std::make_unique<t_state>();
        dd_init_local_state(cr->dd, state_global, localState_.get());
        localStateInstance_ = localState_.get();
    }
    else
    {
        state_change_natoms(state_global, state_global->natoms);
        localStateInstance_ = state_global;
    }

    if (!observablesHistory_->energyHistory)
    {
        observablesHistory_->energyHistory = std::make_unique<energyhistory_t>();
    }
    if (!observablesHistory_->pullHistory)
    {
        observablesHistory_->pullHistory = std::make_unique<PullHistory>();
    }
}

void CheckpointHelper::run(Step step, Time time)
{
    // reads out signal, decides if we should signal checkpoint
    checkpointHandler_->decideIfCheckpointingThisStep(true, step == initStep_, false);
    if (checkpointHandler_->isCheckpointingStep())
    {
        writeCheckpoint(step, time);
    }

    // decides if we should set a checkpointing signal
    checkpointHandler_->setSignal(walltime_accounting_);
}

void CheckpointHelper::scheduleTask(Step step, Time time, const RegisterRunFunction& registerRunFunction)
{
    // Only last step checkpointing is done here
    if (step != lastStep_ || !writeFinalCheckpoint_)
    {
        return;
    }
    registerRunFunction([this, step, time]() { writeCheckpoint(step, time); });
}

void CheckpointHelper::writeCheckpoint(Step step, Time time)
{
    localStateInstance_->flags = 0;

    WriteCheckpointDataHolder checkpointDataHolder;
    for (const auto& [key, client] : clients_)
    {
        client->saveCheckpointState(
                MASTER(cr_) ? std::make_optional(checkpointDataHolder.checkpointData(key)) : std::nullopt,
                cr_);
    }

    mdoutf_write_to_trajectory_files(fplog_, cr_, trajectoryElement_->outf_, MDOF_CPT,
                                     globalNumAtoms_, step, time, localStateInstance_, state_global_,
                                     observablesHistory_, ArrayRef<RVec>(), &checkpointDataHolder);
}

std::optional<SignallerCallback> CheckpointHelper::registerLastStepCallback()
{
    return [this](Step step, Time gmx_unused time) { this->lastStep_ = step; };
}

CheckpointHelperBuilder::CheckpointHelperBuilder(std::unique_ptr<ReadCheckpointDataHolder> checkpointDataHolder,
                                                 StartingBehavior startingBehavior,
                                                 t_commrec*       cr) :
    resetFromCheckpoint_(startingBehavior != StartingBehavior::NewSimulation),
    checkpointDataHolder_(std::move(checkpointDataHolder)),
    checkpointHandler_(nullptr),
    cr_(cr),
    state_(ModularSimulatorBuilderState::AcceptingClientRegistrations)
{
}

void CheckpointHelperBuilder::registerClient(ICheckpointHelperClient* client)
{
    if (!client)
    {
        return;
    }
    if (state_ == ModularSimulatorBuilderState::NotAcceptingClientRegistrations)
    {
        throw SimulationAlgorithmSetupError(
                "Tried to register to CheckpointHelper after it was built.");
    }
    const auto& key = client->clientID();
    if (clientsMap_.count(key) != 0)
    {
        throw SimulationAlgorithmSetupError("CheckpointHelper client key is not unique.");
    }
    clientsMap_[key] = client;
    if (resetFromCheckpoint_)
    {
        if (MASTER(cr_) && !checkpointDataHolder_->keyExists(key))
        {
            throw SimulationAlgorithmSetupError(
                    formatString(
                            "CheckpointHelper client with key %s registered for checkpointing, "
                            "but %s does not exist in the input checkpoint file.",
                            key.c_str(), key.c_str())
                            .c_str());
        }
        client->restoreCheckpointState(
                MASTER(cr_) ? std::make_optional(checkpointDataHolder_->checkpointData(key)) : std::nullopt,
                cr_);
    }
}

void CheckpointHelperBuilder::setCheckpointHandler(std::unique_ptr<CheckpointHandler> checkpointHandler)
{
    checkpointHandler_ = std::move(checkpointHandler);
}

} // namespace gmx
