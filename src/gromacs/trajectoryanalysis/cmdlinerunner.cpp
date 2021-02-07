/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2010-2018, The GROMACS development team.
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
 * Implements gmx::TrajectoryAnalysisCommandLineRunner.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "cmdlinerunner.h"

#include "gromacs/analysisdata/paralleloptions.h"
#include "gromacs/commandline/cmdlinemodulemanager.h"
#include "gromacs/commandline/cmdlineoptionsmodule.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/options/timeunitmanager.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/selectioncollection.h"
#include "gromacs/selection/selectionoptionbehavior.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysismodule.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxomp.h"

#include "runnercommon.h"

namespace gmx
{

namespace
{

/********************************************************************
 * RunnerModule
 */

class RunnerModule : public ICommandLineOptionsModule
{
public:
    explicit RunnerModule(TrajectoryAnalysisModulePointer module) :
        module_(std::move(module)),
        common_(&settings_)
    {
    }

    void init(CommandLineModuleSettings* /*settings*/) override {}
    void initOptions(IOptionsContainer* options, ICommandLineOptionsModuleSettings* settings) override;
    void optionsFinished() override;
    int  run() override;

    TrajectoryAnalysisModulePointer module_;
    TrajectoryAnalysisSettings      settings_;
    TrajectoryAnalysisRunnerCommon  common_;
    SelectionCollection             selections_;

    // Needs to persist past initOptions for .
    std::shared_ptr<SelectionOptionBehavior> selectionOptionBehavior_;
};

void RunnerModule::initOptions(IOptionsContainer* options, ICommandLineOptionsModuleSettings* settings)
{
    std::shared_ptr<TimeUnitBehavior>        timeUnitBehavior(new TimeUnitBehavior());
    selectionOptionBehavior_ = std::make_shared<SelectionOptionBehavior>(&selections_, common_.topologyProvider());
    settings->addOptionsBehavior(timeUnitBehavior);
    settings->addOptionsBehavior(selectionOptionBehavior_);
    IOptionsContainer& commonOptions = options->addGroup();
    IOptionsContainer& moduleOptions = options->addGroup();

    settings_.setOptionsModuleSettings(settings);
    module_->initOptions(&moduleOptions, &settings_);
    settings_.setOptionsModuleSettings(nullptr);
    common_.initOptions(&commonOptions, timeUnitBehavior.get(), module_->supportsMultiThreading());
    selectionOptionBehavior_->initOptions(&commonOptions);

}

void RunnerModule::optionsFinished()
{
    common_.optionsFinished();
    module_->optionsFinished(&settings_);
}

int RunnerModule::run()
{
    common_.initTopology();
    const TopologyInformation& topology = common_.topologyInformation();
    module_->initAnalysis(settings_, topology);

    // Load first frame.
    common_.initFirstFrame();
    common_.initFrameIndexGroup();
    module_->initAfterFirstFrame(settings_, common_.frame());

    t_pbc  pbc;
    const bool hasPbc = settings_.hasPBC();

    int                                 nframes = 0;
    AnalysisDataParallelOptions         dataOptions;

    std::vector<SelectionCollection> frameLocalSelections;
    std::vector<TrajectoryAnalysisModuleDataPointer> frameLocalData;
    for (int i = 0; i < common_.nThreads(); i++)
    {
        SelectionCollection sc2(selections_);
        // frameLocalSelections.emplace_back(selections_);
        // frameLocalData.emplace_back(module_->startFrames(dataOptions, frameLocalSelections.back()));

        frameLocalData.emplace_back(module_->startFrames(dataOptions, selections_));
    }

    int nThreads = 1;
    gmx_omp_set_num_threads(nThreads);
    bool framesRemaining = true;
    t_trxframe frame;
    int thisFrameNum = 0;
    int localDataIndex = 0;

#pragma omp parallel shared(ppbc, topology, frameLocalData, framesRemaining, nframes) private(frame, thisFrameNum, localDataIndex, pbc) default(none)
    do
    {
#pragma ordered
        {
            thisFrameNum = nframes;
            localDataIndex = thisFrameNum % nThreads;
            common_.initFrame();
            frame = common_.frame();
            if (hasPbc)
            {
                set_pbc(&pbc, topology.pbcType(), frame.box);
            }
            selections_.evaluate(&frame, hasPbc ? &pbc: nullptr);
        }

        module_->analyzeFrame(thisFrameNum, frame, hasPbc ? &pbc: nullptr, frameLocalData[localDataIndex].get());
        module_->finishFrameSerial(nframes);

#pragma ordered
        {
            ++nframes;
            framesRemaining = common_.readNextFrame();
        }
    } while (framesRemaining) ;


    for (int i = 0; i < common_.nThreads(); i++) {
        TrajectoryAnalysisModuleData* pdata = frameLocalData[i].get();
        module_->finishFrames(pdata);
        if (pdata != nullptr)
        {
            pdata->finish();
        }
        frameLocalData[i].reset();
    }

    if (common_.hasTrajectory())
    {
        fprintf(stderr, "Analyzed %d frames, last time %.3f\n", nframes, common_.frame().time);
    }
    else
    {
        fprintf(stderr, "Analyzed topology coordinates\n");
    }

    // Restore the maximal groups for dynamic selections.
    selections_.evaluateFinal(nframes);

    module_->finishAnalysis(nframes);
    module_->writeOutput();

    return 0;
}

} // namespace

/********************************************************************
 * TrajectoryAnalysisCommandLineRunner
 */

// static
int TrajectoryAnalysisCommandLineRunner::runAsMain(int argc, char* argv[], const ModuleFactoryMethod& factory)
{
    auto runnerFactory = [factory] { return createModule(factory()); };
    return ICommandLineOptionsModule::runAsMain(argc, argv, nullptr, nullptr, runnerFactory);
}

// static
void TrajectoryAnalysisCommandLineRunner::registerModule(CommandLineModuleManager*  manager,
                                                         const char*                name,
                                                         const char*                description,
                                                         const ModuleFactoryMethod& factory)
{
    auto runnerFactory = [factory] { return createModule(factory()); };
    ICommandLineOptionsModule::registerModuleFactory(manager, name, description, runnerFactory);
}

// static
std::unique_ptr<ICommandLineOptionsModule>
TrajectoryAnalysisCommandLineRunner::createModule(TrajectoryAnalysisModulePointer module)
{
    return ICommandLineOptionsModulePointer(new RunnerModule(std::move(module)));
}

} // namespace gmx
