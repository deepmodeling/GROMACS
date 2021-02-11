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

#include <future>

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
    AnalysisDataParallelOptions dataOptions(common_.nThreads() > 0 ? common_.nThreads() : 1);
    int localDataIndex = 0;



    if (module_->supportsMultiThreading() && common_.nThreads() > 0) {
        std::vector<SelectionCollection> frameLocalSelections;
        std::vector<TrajectoryAnalysisModuleDataPointer> frameLocalData;
        std::vector<std::future<int>> waiters;
        std::vector<t_pbc> pbcs;
        for (int i = 0; i < common_.nThreads(); i++)
        {
            pbcs.emplace_back();
            waiters.emplace_back();
            frameLocalSelections.emplace_back(selections_);
        }
        for (int i = 0; i < common_.nThreads(); i++) {
            frameLocalData.push_back(module_->startFrames(dataOptions, frameLocalSelections[i]));
        }
        const int nThreads = common_.nThreads();
        printf("\n\nParallel execution with %d threads\n\n", nThreads);
        gmx_omp_set_num_threads(nThreads);
        std::vector<t_trxframe> localFrames;
        for (int i = 0; i < nThreads; i ++) {
            t_trxframe& back = localFrames.emplace_back(common_.frame());
            initFrame(&back, common_.frame().natoms, common_.frame().atoms);
        }
        do
        {
            localDataIndex = nframes % nThreads;
            printf("Start loop %d\n", nframes);
            printf("Local data indeex %d\n", localDataIndex);
            if (nframes >= nThreads) {
                printf("Waits for thread %d\n", nframes - nThreads);
                module_->finishFrameSerial(waiters[localDataIndex].get());
            }
            common_.initFrame();
            copyFrame(&common_.frame(), &localFrames[localDataIndex]);
            t_pbc* ppbc = &pbcs[localDataIndex];
            if (hasPbc)
            {
                set_pbc(ppbc, topology.pbcType(), localFrames[localDataIndex].box);
            } else  {
                  ppbc = nullptr;
                }
            frameLocalSelections[localDataIndex].evaluate(&localFrames[localDataIndex], ppbc);
            printf("Dispatches thread frame %d\n", nframes);
            waiters[localDataIndex] = std::async(std::launch::async,[&, ppbc, localDataIndex, nframes]() -> int{
                module_->analyzeFrame(nframes, localFrames[localDataIndex], ppbc, frameLocalData.at(localDataIndex).get());
                return nframes;
            });

            ++nframes;
        } while (common_.readNextFrame()) ;
        printf("Finishes loops\n");
        for (auto& waiter: waiters) {
            if (waiter.valid()) {
                printf("Waits for thread\n");
                waiter.wait();
            }
        }

        for (int i = 0; i < common_.nThreads(); i++) {
            TrajectoryAnalysisModuleData* pdata = frameLocalData[i].get();
            module_->finishFrames(pdata);
        }
        for (int i = 0; i < common_.nThreads(); i++) {
            frameLocalData[i]->finish();
        }
    } else {
        auto pdata =             module_->startFrames(dataOptions, selections_);
        printf("\n\nSerial execution\n\n");
        do
        {
            common_.initFrame();
            t_trxframe& frame = common_.frame();
            if (hasPbc)
            {
                set_pbc(&pbc, topology.pbcType(), frame.box);
            }

            selections_.evaluate(&frame, &pbc);
            module_->analyzeFrame(nframes, frame, &pbc, pdata.get());
            module_->finishFrameSerial(nframes);

            ++nframes;
        } while (common_.readNextFrame());

        module_->finishFrames(pdata.get());
        if (pdata != nullptr)
        {
            pdata->finish();
        }
        pdata.reset();
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
