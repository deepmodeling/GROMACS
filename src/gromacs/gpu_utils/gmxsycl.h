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
 * \brief
 * Wraps the complexity of including SYCL in GROMACS.
 *
 * SYCL headers use symbol DIM as a template parameter, which gets broken by macro DIM defined
 * in gromacs/math/vectypes.h. Here, we include the SYCL header while temporary undefining this macro.
 * The issue for Intel LLVML compiler: https://github.com/intel/llvm/issues/2981
 *
 * Different compilers, at the time of writing, have different names for some of the proposed features
 * of the SYCL2020 standard. For uniformity, they are all aliased in our custom sycl_2020 namespace.
 *
 * \inlibraryapi
 */

#ifndef GMX_GPU_UTILS_GMXSYCL_H
#define GMX_GPU_UTILS_GMXSYCL_H

/* Some versions of Intel ICPX compiler (at least 2021.1.1 and 2021.1.2) fail to unroll a loop
 * in sycl::accessor::__init, and emit -Wpass-failed=transform-warning. This is a useful
 * warning, but mostly noise right now. Probably related to using shared memory accessors.
 * The unroll directive was introduced in https://github.com/intel/llvm/pull/2449. */
#if defined(__INTEL_LLVM_COMPILER)
#    include <CL/sycl/version.hpp>
#    define DISABLE_UNROLL_WARNINGS \
        ((__SYCL_COMPILER_VERSION >= 20201113) && (__SYCL_COMPILER_VERSION <= 20201214))
#else
#    define DISABLE_UNROLL_WARNINGS 0
#endif

// For hipSYCL, we need to activate floating-point atomics
#if defined(__HIP_SYCL__)
#    define HIPSYCL_EXT_FP_ATOMICS
#endif

#if DISABLE_UNROLL_WARNINGS
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wpass-failed"
#endif

#ifdef DIM
#    if DIM != 3
#        error "The workaround here assumes we use DIM=3."
#    else
#        undef DIM
#        include <CL/sycl.hpp>
#        define DIM 3
#    endif
#else
#    include <CL/sycl.hpp>
#endif

#if DISABLE_UNROLL_WARNINGS
#    pragma clang diagnostic pop
#endif

#undef DISABLE_UNROLL_WARNINGS

/* Exposing Intel-specific extensions in a manner compatible with SYCL2020 provisional spec.
 * Despite ICPX (up to 2021.1.2 at the least) having SYCL_LANGUAGE_VERSION=202001,
 * some parts of the spec are still in private namespace (sycl::intel or sycl::ONEAPI, depending
 * on the version), and some functions have different names. To make things easier to upgrade
 * in the future, this thin layer is added.
 * */
namespace sycl_2020
{
namespace detail
{
#if defined(__SYCL_COMPILER_VERSION) // Intel SYCL compiler
// Confirmed to work for 2021.1-beta10 (20201005), 2021.1.1 (20201113), 2021.1.2 (20201214).
namespace origin = cl::sycl::ONEAPI;
#elif defined(__HIPSYCL__)
namespace origin = cl::sycl;
#else
#    error "Unsupported version of Intel SYCL compiler"
#endif
} // namespace detail

using detail::origin::memory_order;
using detail::origin::memory_scope;
using detail::origin::plus;
using detail::origin::sub_group;

#if defined(__SYCL_COMPILER_VERSION)
using detail::origin::atomic_ref;
template<typename... Args>
bool group_any_of(Args&&... args)
{
    return detail::origin::any_of(std::forward<Args>(args)...);
}
template<typename... Args>
auto group_reduce(Args&&... args) -> decltype(detail::origin::reduce(std::forward<Args>(args)...))
{
    return detail::origin::reduce(std::forward<Args>(args)...);
}
#elif defined(__HIPSYCL__)
// No atomic_ref in hipSYCL yet (2021-01-20)
using detail::origin::group_any_of;
using detail::origin::group_reduce;
#else
#    error "Unsupported SYCL compiler"
#endif

} // namespace sycl_2020

#endif
