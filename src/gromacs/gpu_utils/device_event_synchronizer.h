/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020,2021, by the GROMACS development team, led by
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
/*! \libinternal \file
 *  \brief Defines DeviceEventSynchronizer class for CUDA, OpenCL, and SYCL.
 *
 *  \author Erik Lindahl <erik.lindahl@gmail.com>
 *  \author Andrey Alekseenko <al42and@gmail.com>
 * \inlibraryapi
 */
#ifndef GMX_GPU_UTILS_DEVICE_EVENT_SYNCHRONIZER_H
#define GMX_GPU_UTILS_DEVICE_EVENT_SYNCHRONIZER_H

#include <optional>

#include "gromacs/gpu_utils/device_event.h"
#include "gromacs/utility/classhelpers.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"

/*! \libinternal \brief
 * Class to allow for CPU thread to mark and wait for a certain GPU stream execution point.
 *
 * The event can be put into the stream with \ref mark and then later waited on with \ref wait.
 * A device-side wait can be enqueued in a stream using \ref enqueueWait.
 *
 * Additionally, this class offers facilities for runtime checking of correctness by counting
 * how many times each marked event is used as a synchronization point.
 *
 * - The class is initialized with 0 consumption tokens left and in unmarked state.
 * - Destructor checks the reset condition. If it fails, no fatal error occurs, but a warning is
 * printed to \c stderr.
 * - Calling \ref reset checks the reset condition and returns object into the initial state.
 * - Calling \ref mark checks the reset condition, enqueues new event, sets \c consumptionLimit
 * consumption tokens, sets \ref isMarked to \c true.
 * - Calling \ref wait and \ref enqueueWait checks the consumption condition, waits on CPU or
 * enqueues barrier into a GPU stream, and removes one consumption token.
 * - Calling \ref consume checks the consumption condition and removes one token.
 *
 * - Reset condition checks whether the event was fully consumed.
 * If there are any consumption tokens left and \c mustBeFullyConsumedOnReset was set to \c true,
 * a fatal error is emitted. If \c mustBeFullyConsumedOnReset was set to \c false, nothing is done.
 * - Consumption condition checks if there's at least one consumption token left. Fatal error
 * if there is none.
 */
class DeviceEventSynchronizer
{
public:
    /*! Default consumption limit.
     *
     * Empirically chosen to be high enough to allow legitimate event reuse, but low enough
     * to prevent outright suspicious cases when a single event is waited upon dozens of times.
     */
    static constexpr int sc_defaultConsumptionLimit = 10;

    DeviceEventSynchronizer(int  consumptionLimit           = sc_defaultConsumptionLimit,
                            bool mustBeFullyConsumedOnReset = false) :
        deviceEvent_(),
        consumptionLimit_(consumptionLimit),
        consumptionLeft_(0),
        mustBeFullyConsumedOnReset_(mustBeFullyConsumedOnReset),
        isMarked_(false)
    {
    }
    ~DeviceEventSynchronizer()
    {
        if (mustBeFullyConsumedOnReset_ && !isFullyConsumed())
        {
            std::cerr << "Destroying an event that should have been fully consumed but has "
                      << consumptionLeft_ << " / " << consumptionLimit_ << " tokens left." << std::endl;
        }
    }

    //! Check whether the event is marked.
    [[nodiscard]] bool isMarked() const { return isMarked_; }
    //! Check whether the event is ready.
    [[nodiscard]] bool isReady() const { return deviceEvent_.isReady(); }
    //! Check whether the event is marked and there are consumption tokens left.
    [[nodiscard]] bool canBeConsumed() const { return isMarked_ && consumptionLeft_ > 0; }
    //! Check if there are no consumption tokens left.
    [[nodiscard]] bool isFullyConsumed() const { return consumptionLeft_ == 0; }

    // SYCL-TODO (#3895): make it possible to no-op ::mark, ::wait, and ::enqueueWait

    //! Mark the event in \p deviceStream. Replenish consumption tokens.
    void mark(const DeviceStream& deviceStream)
    {
        checkBeforeResetting_();
        deviceStream.markEvent(deviceEvent_);
        consumptionLeft_ = consumptionLimit_;
        isMarked_        = true;
    }

    //! Synchronize CPU thread with the event. Consumes one token.
    void wait()
    {
        checkAndConsume_();
        deviceEvent_.wait();
    }

    //! Enqueue a synchronization barrier waiting for the event into GPU stream. Consumes one token.
    void enqueueWait(const DeviceStream& deviceStream)
    {
        checkAndConsume_();
        deviceStream.enqueueWaitForEvent(deviceEvent_);
    }

    //! Reset the \ref DeviceEventSynchronizer state with check of full consumption if enabled.
    void reset()
    {
        checkBeforeResetting_();
        isMarked_        = false;
        consumptionLeft_ = 0;
    }

    //! Consume one token and do nothing.
    void consume() { checkAndConsume_(); }

private:
    DeviceEvent deviceEvent_;
    int         consumptionLimit_;
    int         consumptionLeft_;
    bool        mustBeFullyConsumedOnReset_;
    bool        isMarked_;

    void checkAndConsume_()
    {
        GMX_RELEASE_ASSERT(canBeConsumed(), "Trying to consume event that is already consumed");
        consumptionLeft_--;
    }
    void checkBeforeResetting_() const
    {
        GMX_RELEASE_ASSERT(!mustBeFullyConsumedOnReset_ || isFullyConsumed(),
                           "Trying to reset an event before it was fully consuming");
    }

    GMX_DISALLOW_COPY_MOVE_AND_ASSIGN(DeviceEventSynchronizer);
};

#endif // GMX_GPU_UTILS_DEVICE_EVENT_SYNCHRONIZER_H
