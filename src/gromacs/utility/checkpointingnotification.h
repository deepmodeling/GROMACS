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
/*! \libinternal \file
 * \brief
 * Declares gmx::CheckpointingNotifier.
 *
 * \author Christian Blau <blau@kth.se>
 * \inlibraryapi
 * \ingroup module_utility
 */

#ifndef GMX_UTILITY_CHECKPOINTINGNOTIFICATION_H
#define GMX_UTILITY_CHECKPOINTINGNOTIFICATION_H

#include "gromacs/utility/mdmodulenotification-impl.h"

namespace gmx
{

struct MdModulesCheckpointReadingDataOnMaster;
struct MdModulesCheckpointReadingBroadcast;
struct MdModulesWriteCheckpointData;

/*! \libinternal
 * \brief Collection of callbacks for checkpointing.
 *
 * Use members of this struct to sign up for checkpointing callbacks.
 *
   \msc
   wordwraparcs=true,
   hscale="2";

   runner [label="runner:\nMdrunner"],
   CallParameter [label = "eventA:\nCallParameter"],
   MOD [label = "mdModules_:\nMdModules"],
   ModuleA [label="moduleA"],
   ModuleB [label="moduleB"],
   MdModuleNotification [label="notifier_:\nMdModuleNotification"];

   MOD box MdModuleNotification [label = "mdModules_ owns notifier_ and moduleA/B"];
   MOD =>> ModuleA [label="instantiates(notifier_)"];
   ModuleA =>> MdModuleNotification [label="subscribe(otherfunc)"];
   ModuleA =>> MOD;
   MOD =>> ModuleB [label="instantiates(notifier_)"];
   ModuleB =>> MdModuleNotification [label="subscribe(func)"];
   ModuleB =>> MOD;
   runner =>> CallParameter [label="instantiate"];
   CallParameter =>> runner ;
   runner =>> MOD [label="notify(eventA)"];
   MOD =>> MdModuleNotification [label="notify(eventA)"];
   MdModuleNotification =>> ModuleA [label="notify(eventA)"];
   ModuleA -> ModuleA [label="func(eventA)"];
   MdModuleNotification =>> ModuleB [label="notify(eventA)"];
   ModuleB -> ModuleB [label="otherfunc(eventA)"];

   \endmsc
 *
 * The template arguments to the members of this struct directly reflect
 * the callback function signature. Arguments passed as pointers are always
 * meant to be modified, but never meant to be stored (in line with the policy
 * everywhere else).
 */
struct CheckpointingNotification
{
    /*! \brief Checkpointing callback functions.
     *
     * MdModulesCheckpointReadingDataOnMaster provides modules with their
     *                                        checkpointed data on the master
     *                                        node and checkpoint file version
     * MdModulesCheckpointReadingBroadcast provides modules with a communicator
     *                                     and the checkpoint file version to
     *                                     distribute their data
     * MdModulesWriteCheckpointData provides the modules with a key-value-tree
     *                              builder to store their checkpoint data and
     *                              the checkpoint file version
     */
    registerMdModuleNotification<MdModulesCheckpointReadingDataOnMaster,
                                 MdModulesCheckpointReadingBroadcast,
                                 MdModulesWriteCheckpointData>::type checkpointingNotifications_;
};

} // namespace gmx

#endif
