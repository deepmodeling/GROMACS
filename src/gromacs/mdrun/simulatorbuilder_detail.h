/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020, by the GROMACS development team, led by
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
 * \brief Simulator Builder details.
 *
 * Implementation support for Simulator Builders.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 * \ingroup module_mdrun
 */

#ifndef GMX_MDRUN_SIMULATORBUILDER_DETAIL_H
#define GMX_MDRUN_SIMULATORBUILDER_DETAIL_H

#include <memory>

#include "gromacs/mdrun/simulatorbuilder.h"

namespace gmx
{

/*! \brief Base class for Simulator Builders.
 *
 * Provides mix-in functionality and the reference implementation of the interface specification.
 *
 * Interface:
 *
 * We can use pointer-to-SimulatorBuilderImplementation as a way to ensure a
 * compatible Simulator Builder. Inheritance may not be the best way to establish
 * the interface, though.
 *
 * Note: there are only two concrete implementations. If we want Simulator
 * implementations to be an extension point for GROMACS, then this interface
 * specification makes sense. If we expect to tightly control the possible
 * Simulator code paths, then this abstraction adds unnecessary complexity,
 * and (with C++17) we could use a std::Variant instead of dynamic polymorphism
 * through a pointer-to-implementation. Then the virtual `add` overloads could
 * be replaced with a single template method and some SFINAE logic.
 *
 * Behavioral inheritance:
 *
 * We can keep a functioning implementation of the Builder interface in a single
 * place by maintaining this class.
 *
 * If we want to separate interface and implementation, then the virtual methods
 * implementations could be implemented in terms of a base template that is
 * specialized instead of overridden.
 */
class SimulatorBuilderImplementation
{
public:
    virtual ~SimulatorBuilderImplementation();

    virtual std::unique_ptr<ISimulator> build(FILE*                            fplog,
                                              t_commrec*                       cr,
                                              const gmx_multisim_t*            ms,
                                              const MDLogger&                  mdlog,
                                              int                              nfile,
                                              const t_filenm*                  fnm,
                                              const gmx_output_env_t*          oenv,
                                              const MdrunOptions&              mdrunOptions,
                                              StartingBehavior                 startingBehavior,
                                              gmx_vsite_t*                     vsite,
                                              Constraints*                     constr,
                                              gmx_enfrot*                      enforcedRotation,
                                              BoxDeformation*                  deform,
                                              IMDOutputProvider*               outputProvider,
                                              const MdModulesNotifier&         mdModulesNotifier,
                                              t_inputrec*                      inputrec,
                                              ImdSession*                      imdSession,
                                              pull_t*                          pull_work,
                                              t_swap*                          swap,
                                              gmx_mtop_t*                      top_global,
                                              t_fcdata*                        fcd,
                                              t_state*                         state_global,
                                              ObservablesHistory*              observablesHistory,
                                              MDAtoms*                         mdAtoms,
                                              t_nrnb*                          nrnb,
                                              gmx_wallcycle*                   wcycle,
                                              t_forcerec*                      fr,
                                              gmx_enerdata_t*                  enerd,
                                              gmx_ekindata_t*                  ekind,
                                              MdrunScheduleWorkload*           runScheduleWork,
                                              const ReplicaExchangeParameters& replExParams,
                                              gmx_membed_t*                    membed,
                                              gmx_walltime_accounting*         walltime_accounting,
                                              std::unique_ptr<StopHandlerBuilder> stopHandlerBuilder,
                                              bool                                doRerun) = 0;
};

} // end namespace gmx

#endif // GMX_MDRUN_SIMULATORBUILDER_DETAIL_H
