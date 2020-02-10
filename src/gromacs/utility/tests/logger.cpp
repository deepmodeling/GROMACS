/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2019,2020, by the GROMACS development team, led by
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

#include "gromacs/utility/logger.h"

#include <gtest/gtest.h>

#include "gromacs/utility/loggerbuilder.h"
#include "gromacs/utility/stringstream.h"

#include "testutils/stringtest.h"
#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{
namespace
{

//! Test fixture for logging tests.
typedef gmx::test::StringTestBase LoggerTest;

TEST_F(LoggerTest, EmptyLoggerWorks)
{
    MDLogger logger;
    GMX_LOG(logger.info).appendText("foobar");
    GMX_LOG(logger.warning).appendText("foobar").asParagraph();
    GMX_LOG(logger.debug).appendText("foobaz");
    GMX_LOG(logger.error).appendText("baz");
}

TEST_F(LoggerTest, EmptyLoggerWorksWithExplicitLevels)
{
    MDLogger logger;
    GMX_LOG_LEVEL(logger.info, VerbosityLevel::NoVerbose).appendText("foobar");
    GMX_LOG_LEVEL(logger.warning, VerbosityLevel::Verbose).appendText("foobar").asParagraph();
    GMX_LOG_LEVEL(logger.debug, VerbosityLevel::NoVerbose).appendText("foobaz");
    GMX_LOG_LEVEL(logger.error, VerbosityLevel::Verbose).appendText("baz");
}

TEST_F(LoggerTest, LogsToStreamNoVerbose)
{
    StringOutputStream stream;
    LoggerBuilder      builder;
    builder.addTargetStream(MDLogger::LoggingStreams::Error, VerbosityLevel::NoVerbose, &stream);
    LoggerOwner     owner  = builder.build();
    const MDLogger& logger = owner.logger();
    GMX_LOG(logger.error).asParagraph().appendText("line that is printed");
    GMX_LOG_LEVEL(logger.error, VerbosityLevel::Verbose)
            .asParagraph()
            .appendText("line that is not printed due to wrong verbosity");
    GMX_LOG(logger.warning)
            .asParagraph()
            .appendText("line that is also not printed, but due to different logging stream");
    checkText(stream.toString(), "Output");
}

TEST_F(LoggerTest, LogsToStreamVerbose)
{
    StringOutputStream stream;
    LoggerBuilder      builder;
    builder.addTargetStream(MDLogger::LoggingStreams::Info, VerbosityLevel::Verbose, &stream);
    LoggerOwner     owner  = builder.build();
    const MDLogger& logger = owner.logger();
    GMX_LOG(logger.info).asParagraph().appendText("line that is printed");
    GMX_LOG_LEVEL(logger.info, VerbosityLevel::Verbose)
            .asParagraph()
            .appendText("line that is printed at different verbosity");
    GMX_LOG(logger.warning)
            .asParagraph()
            .appendText("line that is not printed, due to different logging stream");
    checkText(stream.toString(), "Output");
}

TEST_F(LoggerTest, LogsMultipleLoggingStreamsToSingleTextStream)
{
    StringOutputStream stream;
    LoggerBuilder      builder;
    builder.addTargetStream(MDLogger::LoggingStreams::Info, VerbosityLevel::Verbose, &stream);
    builder.addTargetStream(MDLogger::LoggingStreams::Debug, VerbosityLevel::NoVerbose, &stream);
    LoggerOwner     owner  = builder.build();
    const MDLogger& logger = owner.logger();
    GMX_LOG(logger.info).asParagraph().appendText("line that is printed");
    GMX_LOG_LEVEL(logger.info, VerbosityLevel::Verbose)
            .asParagraph()
            .appendText("line that is printed at different verbosity");
    GMX_LOG(logger.warning)
            .asParagraph()
            .appendText("line that is not printed, due to different logging stream");
    GMX_LOG(logger.debug).asParagraph().appendText("debug line that is printed");
    GMX_LOG_LEVEL(logger.debug, VerbosityLevel::Verbose)
            .asParagraph()
            .appendText("debug line that is not printed due to different verbosity");
    checkText(stream.toString(), "Output");
}

TEST_F(LoggerTest, LogsDifferentVerbosityLevelsToDifferentStreams)
{
    StringOutputStream infoStream;
    StringOutputStream verboseInfoStream;
    LoggerBuilder      builder;
    builder.addTargetStream(MDLogger::LoggingStreams::Info, VerbosityLevel::NoVerbose, &infoStream);
    builder.addTargetStream(MDLogger::LoggingStreams::Info, VerbosityLevel::Verbose, &verboseInfoStream);
    LoggerOwner     owner  = builder.build();
    const MDLogger& logger = owner.logger();
    GMX_LOG(logger.info).asParagraph().appendText("line that is printed to infoStream");
    GMX_LOG_LEVEL(logger.info, VerbosityLevel::Verbose)
            .asParagraph()
            .appendText("line that is printed at different verbosity");
    checkText(infoStream.toString(), "Output of default stream");
    checkText(verboseInfoStream.toString(), "Output of verbose stream");
}

TEST_F(LoggerTest, LogsToFile)
{
    TestFileManager files;
    std::string     filename(files.getTemporaryFilePath("warning.txt"));
    FILE*           fp = fopen(filename.c_str(), "w");
    {
        LoggerBuilder builder;
        builder.addTargetFile(MDLogger::LoggingStreams::Warning, VerbosityLevel::Verbose, fp);
        LoggerOwner     owner  = builder.build();
        const MDLogger& logger = owner.logger();
        GMX_LOG(logger.warning).asParagraph().appendText("Warning that is not verbose");
        GMX_LOG_LEVEL(logger.warning, VerbosityLevel::Verbose)
                .asParagraph()
                .appendText("Warning that is verbose");
    }
    fclose(fp);
    checkFileContents(filename, "Output");
}

TEST_F(LoggerTest, LogsToMultipleFiles)
{
    TestFileManager files;
    std::string     filename1(files.getTemporaryFilePath("warn.txt"));
    std::string     filename2(files.getTemporaryFilePath("error.txt"));
    FILE*           fp1 = fopen(filename1.c_str(), "w");
    FILE*           fp2 = fopen(filename2.c_str(), "w");
    {
        LoggerBuilder builder;
        builder.addTargetFile(MDLogger::LoggingStreams::Warning, VerbosityLevel::NoVerbose, fp1);
        builder.addTargetFile(MDLogger::LoggingStreams::Error, VerbosityLevel::Verbose, fp2);
        gmx::LoggerOwner     owner  = builder.build();
        const gmx::MDLogger& logger = owner.logger();
        GMX_LOG(logger.warning).appendText("Warning that is not verbose is printed").asParagraph();
        GMX_LOG_LEVEL(logger.warning, VerbosityLevel::Verbose)
                .asParagraph()
                .appendText("Verbose warning is not printed");
        GMX_LOG(logger.error).asParagraph().appendText("Default error is printed");
        GMX_LOG_LEVEL(logger.error, VerbosityLevel::Verbose)
                .asParagraph()
                .appendText("Verbose error is printed");
    }
    fclose(fp1);
    fclose(fp2);
    checkFileContents(filename1, "Output1");
    checkFileContents(filename2, "Output2");
}

} // namespace
} // namespace test
} // namespace gmx
