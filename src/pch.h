#include "gmxpre.h"

#include <cinttypes>
#include <cmath>

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _MSC_VER
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#endif

#include "gromacs/math/vectypes.h"
#include "gromacs/simd/simd_math.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/stringutil.h"