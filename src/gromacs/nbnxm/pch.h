#include "gmxpre.h"

#include "config.h"

#include "gromacs/pbcutil/ishift.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/simd/simd.h"
#include "gromacs/simd/simd_math.h"
#include "gromacs/simd/vector_operations.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/basedefinitions.h"

#include "atomdata.h"
#include "nbnxm_simd.h"
#include "pairlist.h"
