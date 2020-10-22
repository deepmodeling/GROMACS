/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2015,2016,2017,2018,2019,2020, by the GROMACS development team, led by
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
 * \brief Tests for gmx::ConstexprOptional.
 *
 * \author Andrey Alekseenko <al42and@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include <numeric>
#include <string>
#include <vector>

#include "gromacs/utility/classhelpers.h"

#include <gtest/gtest.h>

namespace gmx
{

namespace
{

template<bool debug_>
std::string codeFromExample()
{
    constexpr bool                        debug = debug_;
    ConstexprOptional<std::string, debug> errorMsg("Have following errors: ");
    if constexpr (debug)
    {
        errorMsg->append("error1");
    }
    return errorMsg.value_or("");
}

TEST(ConstexpOptionalTest, ExampleCheck)
{
    EXPECT_EQ(codeFromExample<false>(), "");
    EXPECT_EQ(codeFromExample<true>(), "Have following errors: error1");
}

TEST(ConstexpOptionalTest, VectorTrueTest)
{
    // Check constructor
    ConstexprOptional<std::vector<int>, true> vec(10, 0);
    // Check has_value
    EXPECT_TRUE(vec.has_value());
    EXPECT_TRUE(bool(vec));
    // Check has_value in compile-time
    static_assert(vec);
    std::iota(vec->begin(), vec->end(), 0);
    // Check various getters
    vec->push_back(11);
    (*vec).push_back(12);
    vec.value().push_back(13);
    // Check value
    EXPECT_EQ(vec.value().size(), 13);
    // Check value_or
    EXPECT_EQ(vec.value_or(std::vector<int>{}).size(), 13);
    // Check assignment
    vec.value() = std::vector<int>(20, 0);
    EXPECT_EQ(vec->size(), 20);
    // Check that copy works
    void*            ptr_old = vec->data();
    std::vector<int> vec_copy(vec.value());
    void*            ptr_copy = vec_copy.data();
    EXPECT_NE(ptr_old, ptr_copy);
    // Check that move works
    std::vector<int> vec_move(std::move(vec.value()));
    void*            ptr_move = vec_move.data();
    EXPECT_EQ(ptr_old, ptr_move); // Should be a new pointer
}

//! Define the types that end up being available as TypeParam in the test cases for ConstexprOptional
typedef ::testing::Types<bool, char, int, std::string, std::vector<int>, const bool, const char, const int, const std::string, const std::vector<int>>
        TypeParams;

template<typename TypeParam>
class ConstexprOptionalFalseTest : public ::testing::Test
{
public:
    void runTests()
    {
        TypeParam default_value{};
        // Check constructor
        ConstexprOptional<TypeParam, false> x("12");
        // Check has_value
        EXPECT_FALSE(x.has_value());
        EXPECT_FALSE(bool(x));
        // Check has_value in compile-time
        static_assert(!x);
        // Check value_or
        EXPECT_EQ(x.value_or(default_value), default_value);
        // Constructor can take any number of parameters
        [[maybe_unused]] ConstexprOptional<TypeParam, false> x_0;
        [[maybe_unused]] ConstexprOptional<TypeParam, false> x_1(1);
        [[maybe_unused]] ConstexprOptional<TypeParam, false> x_2(1, std::vector<int>(3, 0));
        [[maybe_unused]] ConstexprOptional<TypeParam, false> x_3(0, 'a', false);
    }
};

TYPED_TEST_CASE(ConstexprOptionalFalseTest, TypeParams);

TYPED_TEST(ConstexprOptionalFalseTest, SanityCheck)
{
    this->runTests();
}

} // namespace

} // namespace gmx
