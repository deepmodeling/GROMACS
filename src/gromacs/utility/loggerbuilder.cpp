/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2018,2019,2020, by the GROMACS development team, led by
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
#include "gmxpre.h"

#include "loggerbuilder.h"

#include <memory>
#include <vector>

#include "gromacs/utility/filestream.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textwriter.h"

namespace gmx
{

class LogTargetCollection : public ILogTarget
{
public:
    void addTarget(ILogTarget* target) { targets_.push_back(target); }

    void writeEntry(const LogEntry& entry) override
    {
        for (ILogTarget* target : targets_)
        {
            target->writeEntry(entry);
        }
    }

private:
    std::vector<ILogTarget*> targets_;
};

class LogTargetFormatter : public ILogTarget
{
public:
    explicit LogTargetFormatter(TextOutputStream* stream) : writer_(stream) {}

    void writeEntry(const LogEntry& entry) override;

private:
    TextWriter writer_;
};


void LogTargetFormatter::writeEntry(const LogEntry& entry)
{
    if (entry.asParagraph)
    {
        writer_.ensureEmptyLine();
    }
    writer_.writeLine(entry.text);
    if (entry.asParagraph)
    {
        writer_.ensureEmptyLine();
    }
}

/********************************************************************
 * LoggerOwner::Impl
 */

class LoggerOwner::Impl
{
public:
    explicit Impl(std::array<std::array<ILogTarget*, VerbosityLevelCount>, MDLogger::LogStreamCount> loggerTargets) :
        logger_(loggerTargets)
    {
    }

    MDLogger                                       logger_;
    std::vector<std::unique_ptr<TextOutputStream>> streams_;
    std::vector<std::unique_ptr<ILogTarget>>       targets_;
};

/********************************************************************
 * LoggerOwner
 */

LoggerOwner::LoggerOwner(std::unique_ptr<Impl> impl) :
    impl_(impl.release()),
    logger_(&impl_->logger_)
{
}

LoggerOwner::LoggerOwner(LoggerOwner&& other) noexcept :
    impl_(std::move(other.impl_)),
    logger_(&impl_->logger_)
{
}

LoggerOwner& LoggerOwner::operator=(LoggerOwner&& other) noexcept
{
    impl_   = std::move(other.impl_);
    logger_ = &impl_->logger_;
    return *this;
}

LoggerOwner::~LoggerOwner() {}

/********************************************************************
 * LoggerBuilder::Impl
 */

class LoggerBuilder::Impl
{
public:
    std::vector<std::unique_ptr<TextOutputStream>> streams_;
    std::vector<std::unique_ptr<ILogTarget>>       targets_;
    std::array<std::array<std::vector<ILogTarget*>, VerbosityLevelCount>, MDLogger::LogStreamCount> loggerTargets_;
    int verbosityLevel_      = 0;
    int errorVerbosityLevel_ = 0;
    int debugVerbosityLevel_ = 0;
};

/********************************************************************
 * LoggerBuilder
 */

LoggerBuilder::LoggerBuilder() : impl_(new Impl) {}

LoggerBuilder::~LoggerBuilder() {}

void LoggerBuilder::addTargetStream(MDLogger::LoggingStreams target, VerbosityLevel level, TextOutputStream* stream)
{
    impl_->targets_.push_back(std::unique_ptr<ILogTarget>(new LogTargetFormatter(stream)));
    ILogTarget* logTarget   = impl_->targets_.back().get();
    const int   targetValue = static_cast<int>(target);

    for (int i = 0; i <= static_cast<int>(level); ++i)
    {
        impl_->loggerTargets_[targetValue][i].push_back(logTarget);
    }
}

void LoggerBuilder::addTargetFile(MDLogger::LoggingStreams target, VerbosityLevel level, FILE* fp)
{
    std::unique_ptr<TextOutputStream> stream(new TextOutputFile(fp));
    addTargetStream(target, level, stream.get());
    impl_->streams_.push_back(std::move(stream));
}

LoggerOwner LoggerBuilder::build()
{
    std::array<std::array<ILogTarget*, VerbosityLevelCount>, MDLogger::LogStreamCount> loggerTargets;
    for (int stream = 0; stream < MDLogger::LogStreamCount; ++stream)
    {
        for (int level = 0; level < VerbosityLevelCount; ++level)
        {
            auto& levelTargets           = impl_->loggerTargets_[stream][level];
            loggerTargets[stream][level] = nullptr;
            if (!levelTargets.empty())
            {
                if (levelTargets.size() == 1)
                {
                    loggerTargets[stream][level] = levelTargets[0];
                }
                else
                {
                    std::unique_ptr<LogTargetCollection> collection(new LogTargetCollection);
                    for (auto& target : levelTargets)
                    {
                        collection->addTarget(target);
                    }
                    loggerTargets[stream][level] = collection.get();
                    impl_->targets_.push_back(std::move(collection));
                }
            }
            levelTargets.clear();
        }
    }
    std::unique_ptr<LoggerOwner::Impl> data(new LoggerOwner::Impl(loggerTargets));
    data->targets_ = std::move(impl_->targets_);
    data->streams_ = std::move(impl_->streams_);
    return LoggerOwner(std::move(data));
}

} // namespace gmx
