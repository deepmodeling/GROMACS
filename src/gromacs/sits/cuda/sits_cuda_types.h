#include <cassert>
#include <cinttypes>
#include <csignal>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <memory>

#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/math/utilities.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/range.h"
#include "gromacs/utility/real.h"

#include "gromacs/sits/sits.h"

/** \internal
 * \brief Main data structure for CUDA nonbonded force calculations.
 */

struct sits_cuda
{
    int sits_calc_mode = 0;        // sits calculation mode: classical or simple
    int sits_enh_mode = PP_AND_PW; // sits enhancing region: solvate, intramolecular or intermolecular
    bool sits_enh_bias = false;    // whether to enhance the bias

    int natoms;       /**< number of atoms                              */

    int* energrp;

    float3* d_enerd; // stores pp, pw, and ww energies in bonded and nonbonded SR kernels

    float3* d_force_tot;
    float3* d_force_pw;

    float3* d_force_tot_nbat;
    float3* d_force_pw_nbat;
}