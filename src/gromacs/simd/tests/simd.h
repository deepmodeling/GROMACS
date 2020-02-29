/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2014,2015,2016,2017,2018 by the GROMACS development team.
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
#ifndef GMX_SIMD_TESTS_SIMD_H
#define GMX_SIMD_TESTS_SIMD_H

/*! \internal \file
 * \brief
 * Declares fixture for testing of normal SIMD (not SIMD4) functionality.
 *
 * The SIMD tests are both simple and complicated. The actual testing logic
 * is \a very straightforward since we just need to test single values against
 * the math library, and for some math functions we need to do it in a loop.
 * This could have been achieved in minutes with the default Google Test tools,
 * if it wasn't for the problem that we cannot access or compare SIMD contents
 * directly without using lots of other SIMD functionality. For this reason
 * we have separate the basic testing of load/store operations into a separate
 * bootstrapping test. Once this works, we use a set of utility routines to
 * convert SIMD contents to/from std:vector<> and perform the rest of the tests,
 * which then can farmed out to the base class SimdBaseTest that is common
 * to SIMD and SIMD4.
 *
 * Another complication is that the width of the SIMD implementation will
 * depend on the hardware and precision. For some simple operations it is
 * sufficient to set all SIMD elements to the same value, and check that the
 * result is present in all elements. However, for a few more complex
 * instructions that might rely on shuffling under-the-hood it is important
 * that we can test operations with different elements. We achieve this by
 * having test code that can initialize a SIMD variable from an std::vector
 * of arbitrary length; the vector is simply repeated to fill all elements in
 * the SIMD variable. We also have similar routines to compare a SIMD result
 * with values in a vector, which returns true iff all elements match.
 *
 * This way we can write simple tests that use different values for all SIMD
 * elements. Personally I like using vectors of length 3, since this means
 * there are no simple repeated patterns in low/high halves of SIMD variables
 * that are 2,4,8,or 16 elements wide, and we still don't have to care about
 * the exact SIMD width of the underlying implementation.
 *
 * Note that this utility uses a few SIMD load/store instructions internally -
 * those have been tested separately in the bootstrap_loadstore.cpp file.
 *
 * \author Erik Lindahl <erik.lindahl@scilifelab.se>
 * \ingroup module_simd
 */
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/simd/simd.h"

#include "base.h"
#include "data.h"

#if GMX_SIMD

namespace gmx
{
namespace test
{


/*! \cond internal */
/*! \addtogroup module_simd */
/*! \{ */

/* Unfortunately we cannot keep static SIMD constants in the test fixture class.
 * The problem is that SIMD memory need to be aligned, and in particular
 * this applies to automatic storage of variables in classes. For SSE registers
 * this means 16-byte alignment (which seems to work), but AVX requires 32-bit
 * alignment. At least both gcc-4.7.3 and Apple clang-5.0 (OS X 10.9) fail to
 * align these variables when they are stored as data in a class.
 *
 * In theory we could set some of these on-the-fly e.g. with setSimdFrom3R()
 * instead (although that would mean repeating code between tests), but many of
 * the constants depend on the current precision not to mention they
 * occasionally have many digits that need to be exactly right, and keeping
 * them in a single place makes sure they are consistent.
 */
#    if GMX_SIMD_HAVE_REAL
#        if GMX_SIMD_HAVE_REAL_GLOBAL
extern const SimdReal rSimd_c0c1c2; //!< c0,c1,c2 repeated
extern const SimdReal rSimd_c3c4c5; //!< c3,c4,c5 repeated
extern const SimdReal rSimd_c6c7c8; //!< c6,c7,c8 repeated
extern const SimdReal rSimd_c3c0c4; //!< c3,c0,c4 repeated
extern const SimdReal rSimd_c4c6c8; //!< c4,c6,c8 repeated
extern const SimdReal rSimd_c7c2c3; //!< c7,c2,c3 repeated
extern const SimdReal rSimd_m0m1m2; //!< -c0,-c1,-c2 repeated
extern const SimdReal rSimd_m3m0m4; //!< -c3,-c0,-c4 repeated

extern const SimdReal rSimd_2p25;  //!< Value that rounds down.
extern const SimdReal rSimd_3p25;  //!< Value that rounds down.
extern const SimdReal rSimd_3p75;  //!< Value that rounds up.
extern const SimdReal rSimd_m2p25; //!< Negative value that rounds up.
extern const SimdReal rSimd_m3p25; //!< Negative value that rounds up.
extern const SimdReal rSimd_m3p75; //!< Negative value that rounds down.
//! Three large floating-point values whose exponents are >32.
extern const SimdReal rSimd_Exp;

#            if GMX_SIMD_HAVE_LOGICAL
extern const SimdReal rSimd_logicalA;         //!< Bit pattern to test logical ops
extern const SimdReal rSimd_logicalB;         //!< Bit pattern to test logical ops
extern const SimdReal rSimd_logicalResultOr;  //!< Result or bitwise 'or' of A and B
extern const SimdReal rSimd_logicalResultAnd; //!< Result or bitwise 'and' of A and B
#            endif                            // GMX_SIMD_HAVE_LOGICAL

#            if GMX_SIMD_HAVE_DOUBLE && GMX_DOUBLE
// Make sure we also test exponents outside single precision when we use double
extern const SimdReal rSimd_ExpDouble;
#            endif
// Magic FP numbers corresponding to specific bit patterns
extern const SimdReal rSimd_Bits1; //!< Pattern F0 repeated to fill single/double.
extern const SimdReal rSimd_Bits2; //!< Pattern CC repeated to fill single/double.
extern const SimdReal rSimd_Bits3; //!< Pattern C0 repeated to fill single/double.
extern const SimdReal rSimd_Bits4; //!< Pattern 0C repeated to fill single/double.
extern const SimdReal rSimd_Bits5; //!< Pattern FC repeated to fill single/double.
extern const SimdReal rSimd_Bits6; //!< Pattern 3C repeated to fill single/double.
#        else                      // GMX_SIMD_HAVE_REAL_GLOBAL
//!< c0,c1,c2 repeated
#            define rSimd_c0c1c2 setSimdRealFrom3R(c0, c1, c2)
//!< c3,c4,c5 repeated
#            define rSimd_c3c4c5 setSimdRealFrom3R(c3, c4, c5)
//!< c6,c7,c8 repeated
#            define rSimd_c6c7c8 setSimdRealFrom3R(c6, c7, c8)
//!< c3,c0,c4 repeated
#            define rSimd_c3c0c4 setSimdRealFrom3R(c3, c0, c4)
//!< c4,c6,c8 repeated
#            define rSimd_c4c6c8 setSimdRealFrom3R(c4, c6, c8)
//!< c7,c2,c3 repeated
#            define rSimd_c7c2c3 setSimdRealFrom3R(c7, c2, c3)
//!< -c0,-c1,-c2 repeated
#            define rSimd_m0m1m2 setSimdRealFrom3R(-c0, -c1, -c2)
//!< -c3,-c0,-c4 repeated
#            define rSimd_m3m0m4 setSimdRealFrom3R(-c3, -c0, -c4)

//!< Value that rounds down.
#            define rSimd_2p25 setSimdRealFrom1R(2.25)
//!< Value that rounds down.
#            define rSimd_3p25 setSimdRealFrom1R(3.25)
//!< Value that rounds up.
#            define rSimd_3p75 setSimdRealFrom1R(3.75)
//!< Negative value that rounds up.
#            define rSimd_m2p25 setSimdRealFrom1R(-2.25)
//!< Negative value that rounds up.
#            define rSimd_m3p25 setSimdRealFrom1R(-3.25)
//!< Negative value that rounds down.
#            define rSimd_m3p75 setSimdRealFrom1R(-3.75)
//! Three large floating-point values whose exponents are >32.
#            define rSimd_Exp                                                                       \
                setSimdRealFrom3R(1.4055235171027452623914516e+18, 5.3057102734253445623914516e-13, \
                                  -2.1057102745623934534514516e+16)

#            if GMX_SIMD_HAVE_DOUBLE && GMX_DOUBLE
// Make sure we also test exponents outside single precision when we use double
#                define rSimd_ExpDouble                                                                 \
                    setSimdRealFrom3R(6.287393598732017379054414e+176, 8.794495252903116023030553e-140, \
                                      -3.637060701570496477655022e+202)
#            endif // GMX_SIMD_HAVE_DOUBLE && GMX_DOUBLE

#            if GMX_SIMD_HAVE_LOGICAL
#                if GMX_DOUBLE
//!< Bit pattern to test logical ops
#                    define rSimd_logicalA setSimdRealFrom1R(1.3333333332557231188)
//!< Bit pattern to test logical ops
#                    define rSimd_logicalB setSimdRealFrom1R(1.7999999998137354851)
//!< Result or bitwise 'or' of A and B
#                    define rSimd_logicalResultAnd setSimdRealFrom1R(1.266666666604578495)
//!< Result or bitwise 'and' of A and B
#                    define rSimd_logicalResultOr setSimdRealFrom1R(1.8666666664648801088)
#                else // GMX_DOUBLE
//!< Bit pattern to test logical ops
#                    define rSimd_logicalA setSimdRealFrom1R(1.3333282470703125)
//!< Bit pattern to test logical ops
#                    define rSimd_logicalB setSimdRealFrom1R(1.79998779296875)
//!< Result or bitwise 'or' of A and B
#                    define rSimd_logicalResultAnd setSimdRealFrom1R(1.26666259765625)
//!< Result or bitwise 'and' of A and B
#                    define rSimd_logicalResultOr setSimdRealFrom1R(1.8666534423828125)
#                endif // GMX_DOUBLE
#            endif     // GMX_SIMD_HAVE_LOGICAL

#        endif // GMX_SIMD_HAVE_REAL_GLOBAL
#    endif     // GMX_SIMD_HAVE_REAL
#    if GMX_SIMD_HAVE_INT32_ARITHMETICS
#        if GMX_SIMD_HAVE_INT32_GLOBAL
extern const SimdInt32 iSimd_1_2_3;    //!< Three generic ints.
extern const SimdInt32 iSimd_4_5_6;    //!< Three generic ints.
extern const SimdInt32 iSimd_7_8_9;    //!< Three generic ints.
extern const SimdInt32 iSimd_5_7_9;    //!< iSimd_1_2_3 + iSimd_4_5_6.
extern const SimdInt32 iSimd_1M_2M_3M; //!< Term1 for 32bit add/sub.
extern const SimdInt32 iSimd_4M_5M_6M; //!< Term2 for 32bit add/sub.
extern const SimdInt32 iSimd_5M_7M_9M; //!< iSimd_1M_2M_3M + iSimd_4M_5M_6M.
#        else                          // GMX_SIMD_HAVE_INT32_GLOBAL
//!< Three generic ints.
#            define iSimd_1_2_3 setSimdIntFrom3I(1, 2, 3)
//!< Three generic ints.
#            define iSimd_4_5_6 setSimdIntFrom3I(4, 5, 6)
//!< Three generic ints.
#            define iSimd_7_8_9 setSimdIntFrom3I(7, 8, 9)
//!< iSimd_1_2_3 + iSimd_4_5_6.
#            define iSimd_5_7_9 setSimdIntFrom3I(5, 7, 9)
//!< Term1 for 32bit add/sub.
#            define iSimd_1M_2M_3M setSimdIntFrom3I(1000000, 2000000, 3000000)
//!< Term2 for 32bit add/sub.
#            define iSimd_4M_5M_6M setSimdIntFrom3I(4000000, 5000000, 6000000)
//!< iSimd_1M_2M_3M + iSimd_4M_5M_6M.
#            define iSimd_5M_7M_9M setSimdIntFrom3I(5000000, 7000000, 9000000)
#        endif // GMX_SIMD_HAVE_INT32_GLOBAL
#    endif
#    if GMX_SIMD_HAVE_INT32_LOGICAL
#        if GMX_SIMD_HAVE_INT32_GLOBAL
extern const SimdInt32 iSimd_0xF0F0F0F0; //!< Bitpattern to test integer logical operations.
extern const SimdInt32 iSimd_0xCCCCCCCC; //!< Bitpattern to test integer logical operations.
#        else                            // GMX_SIMD_HAVE_INT32_GLOBAL
//!< Bitpattern to test integer logical operations.
#            define iSimd_0xF0F0F0F0 setSimdIntFrom1I(0xF0F0F0F0)
//!< Bitpattern to test integer logical operations.
#            define iSimd_0xCCCCCCCC setSimdIntFrom1I(0xCCCCCCCC)
#        endif // GMX_SIMD_HAVE_INT32_GLOBAL
#    endif

/*! \internal
 * \brief
 * Test fixture for SIMD tests.
 *
 * This is a very simple test fixture that basically just takes the common
 * SIMD/SIMD4 functionality from SimdBaseTest and creates wrapper routines
 * specific for normal SIMD functionality.
 */
class SimdTest : public SimdBaseTest
{
public:
#    if GMX_SIMD_HAVE_REAL
    /*! \brief Compare two real SIMD variables for approximate equality.
     *
     * This is an internal implementation routine. YOu should always use
     * GMX_EXPECT_SIMD_REAL_NEAR() instead.
     *
     * This routine is designed according to the Google test specs, so the char
     * strings will describe the arguments to the macro.
     *
     * The comparison is applied to each element, and it returns true if each element
     * in the SIMD test variable is within the class tolerances of the corresponding
     * reference element.
     */


    ::testing::AssertionResult
    compareSimdRealUlp(const char* refExpr, const char* tstExpr, SimdReal ref, SimdReal tst);

    /*! \brief Compare two real SIMD variables for exact equality.
     *
     * This is an internal implementation routine. YOu should always use
     * GMX_EXPECT_SIMD_REAL_NEAR() instead.
     *
     * This routine is designed according to the Google test specs, so the char
     * strings will describe the arguments to the macro.
     *
     * The comparison is applied to each element, and it returns true if each element
     * in the SIMD test variable is within the class tolerances of the corresponding
     * reference element.
     */
    ::testing::AssertionResult compareSimdEq(const char* refExpr, const char* tstExpr, SimdReal ref, SimdReal tst);

    /*! \brief Compare two 32-bit integer SIMD variables.
     *
     * This is an internal implementation routine. YOu should always use
     * GMX_EXPECT_SIMD_INT_EQ() instead.
     *
     * This routine is designed according to the Google test specs, so the char
     * strings will describe the arguments to the macro, while the SIMD and
     * tolerance arguments are used to decide if the values are approximately equal.
     *
     * The comparison is applied to each element, and it returns true if each element
     * in the SIMD variable tst is identical to the corresponding reference element.
     */
    ::testing::AssertionResult compareSimdEq(const char* refExpr, const char* tstExpr, SimdInt32 ref, SimdInt32 tst);
#    endif
};

#    if GMX_SIMD_HAVE_REAL
/*! \brief Convert SIMD real to std::vector<real>.
 *
 * The returned vector will have the same length as the SIMD width.
 */
std::vector<real> simdReal2Vector(SimdReal simd);

/*! \brief Return floating-point SIMD value from std::vector<real>.
 *
 * If the vector is longer than SIMD width, only the first elements will be used.
 * If it is shorter, the contents will be repeated to fill the SIMD register.
 */
SimdReal vector2SimdReal(const std::vector<real>& v);

/*! \brief Set SIMD register contents from three real values.
 *
 * Our reason for using three values is that 3 is not a factor in any known
 * SIMD width, so this way there will not be any simple repeated patterns e.g.
 * between the low/high 64/128/256 bits in the SIMD register, which could hide bugs.
 */
SimdReal setSimdRealFrom3R(real r0, real r1, real r2);

/*! \brief Set SIMD register contents from single real value.
 *
 * All elements is set from the given value. This is effectively the same
 * operation as simdSet1(), but is implemented using only load/store
 * operations that have been tested separately in the bootstrapping tests.
 */
SimdReal setSimdRealFrom1R(real value);

/*! \brief Test if a SIMD real is bitwise identical to reference SIMD value. */
#        define GMX_EXPECT_SIMD_REAL_EQ(ref, tst) EXPECT_PRED_FORMAT2(compareSimdEq, ref, tst)

/*! \brief Test if a SIMD is bitwise identical to reference SIMD value. */
#        define GMX_EXPECT_SIMD_EQ(ref, tst) EXPECT_PRED_FORMAT2(compareSimdEq, ref, tst)

/*! \brief Test if a SIMD real is within tolerance of reference SIMD value. */
#        define GMX_EXPECT_SIMD_REAL_NEAR(ref, tst) \
            EXPECT_PRED_FORMAT2(compareSimdRealUlp, ref, tst)

/*! \brief Convert SIMD integer to std::vector<int>.
 *
 * The returned vector will have the same length as the SIMD width.
 */
std::vector<std::int32_t> simdInt2Vector(SimdInt32 simd);

/*! \brief Return 32-bit integer SIMD value from std::vector<int>.
 *
 * If the vector is longer than SIMD width, only the first elements will be used.
 * If it is shorter, the contents will be repeated to fill the SIMD register.
 */
SimdInt32 vector2SimdInt(const std::vector<std::int32_t>& v);

/*! \brief Set SIMD register contents from three int values.
 *
 * Our reason for using three values is that 3 is not a factor in any known
 * SIMD width, so this way there will not be any simple repeated patterns e.g.
 * between the low/high 64/128/256 bits in the SIMD register, which could hide bugs.
 */
SimdInt32 setSimdIntFrom3I(int i0, int i1, int i2);

/*! \brief Set SIMD register contents from single integer value.
 *
 * All elements is set from the given value. This is effectively the same
 * operation as simdSet1I(), but is implemented using only load/store
 * operations that have been tested separately in the bootstrapping tests.
 */
SimdInt32 setSimdIntFrom1I(int value);

/*! \brief Macro that checks SIMD integer expression against SIMD or reference int.
 *
 * If the reference argument is a scalar integer it will be expanded into
 * the width of the SIMD register and tested against all elements.
 */
#        define GMX_EXPECT_SIMD_INT_EQ(ref, tst) EXPECT_PRED_FORMAT2(compareSimdEq, ref, tst)

#    endif // GMX_SIMD_HAVE_REAL

/*! \} */
/*! \endcond */

} // namespace test
} // namespace gmx

#endif // GMX_SIMD

#endif // GMX_SIMD_TESTS_SIMD_H
