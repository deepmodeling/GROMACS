/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2015,2016,2018,2019,2020, by the GROMACS development team, led by
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

#include "gromacs/random/seed.h"

#include <time.h>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/fatalerror.h"

namespace gmx
{


/*! \brief Check if the RDRAND random device functioning correctly
 *
 * Due to a bug in AMD Ryzen microcode, RDRAND may always return -1 (0xFFFFFFFF).
 * To avoid that, fall back to using PRNG instead of RDRAND if this happens.
 *
 * \returns The result of the checks.
 */
static bool checkIfRandomDeviceIsFunctional()
{
    std::random_device rd;

    uint64_t randomNumber1 = static_cast<uint64_t>(rd());
    uint64_t randomNumber2 = static_cast<uint64_t>(rd());

    // Due to a bug in AMD Ryzen microcode, RDRAND may always return -1 (0xFFFFFFFF).
    // To avoid that, fall back to using PRNG instead of RDRAND if this happens.
    if (randomNumber1 == 0xFFFFFFFF && randomNumber2 == 0xFFFFFFFF)
    {
        if (debug)
        {
            fprintf(debug,
                    "Hardware random number generator (RDRAND) returned -1 (0xFFFFFFFF) twice in\n"
                    "a row. This may be due to a known bug in AMD Ryzen microcode.");
        }
        return false;
    }
    return true;
}

/*! \brief Get the next pure or pseudo-random number
 *
 * Returns the next random number taken from the hardware generator or from PRNG.
 *
 * \param[in] gen  Pseudo-random/random numbers generator/device to use.
 *
 * \return Random or pseudo-random number.
 */
template<typename GeneratorType>
static uint64_t makeRandomSeedInternal(GeneratorType& gen)
{
    constexpr std::size_t resultBits = std::numeric_limits<uint64_t>::digits;
    constexpr std::size_t numBitsInRandomNumber =
            std::numeric_limits<typename GeneratorType::result_type>::digits;

    uint64_t result = static_cast<uint64_t>(gen());
    // This is needed so that compiler understands that what follows is a dead branch
    // and not complains about shift count larger than number of bits in the result.
    constexpr std::size_t shiftCount = (resultBits < numBitsInRandomNumber) ? numBitsInRandomNumber : 0;
    for (std::size_t bits = numBitsInRandomNumber; bits < resultBits; bits += numBitsInRandomNumber)
    {
        result = (result << shiftCount) | static_cast<uint64_t>(gen());
    }
    return result;
}

uint64_t makeRandomSeed()
{
    bool useRdrand = checkIfRandomDeviceIsFunctional();
    if (useRdrand)
    {
        std::random_device rd;
        return makeRandomSeedInternal(rd);
    }
    else
    {
        std::mt19937_64 prng(time(nullptr));
        return makeRandomSeedInternal(prng);
    }
}


} // namespace gmx
