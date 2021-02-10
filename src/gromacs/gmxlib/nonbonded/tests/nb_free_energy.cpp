/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018,2019,2020, by the GROMACS development team, led by
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
 * \brief Implements test of nonbonded fep kernel
 *
 * Implements the test logic from the bonded interactions also for the
 * nonbonded fep kernel. This requires setting up some more input
 * structures that in the bonded case.
 *
 * \author Sebastian Kehl <sebastian.kehl@mpcdf.mpg.de>
 * \ingroup module_gmxlib_nonbonded
 */
#include "gmxpre.h"

#include "gromacs/gmxlib/nonbonded/nb_free_energy.h"
#include "gromacs/gmxlib/nonbonded/nonbonded.h"

#include <cmath>

#include <gtest/gtest.h>

#include "gromacs/math/paddedvector.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/nblist.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/tables/forcetable.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/forcefieldparameters.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/textwriter.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/ewald/ewald_utils.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

//@{
/*
 * Utility functions to setup t_forcerec and interaction_const_t structures
 *
 * These functions are similar to some static functions from forcerec.cpp
 * which is reflected in the nameing that is not according to the style guide.
 */
static std::vector<real> mk_nbfp(const gmx_ffparams_t* idef)
{
    std::vector<real> nbfp;
    int               atnr;

    atnr = idef->atnr;
    nbfp.resize(2 * atnr * atnr);
    int k = 0;
    for (int i = 0; (i < atnr); i++)
    {
        for (int j = 0; (j < atnr); j++, k++)
        {
            /* nbfp now includes the 6.0/12.0 derivative prefactors */
            C6(nbfp, atnr, i, j)  = idef->iparams[k].lj.c6 * 6.0;
            C12(nbfp, atnr, i, j) = idef->iparams[k].lj.c12 * 12.0;
        }
    }

    return nbfp;
}

static std::vector<real> make_ljpme_c6grid(const gmx_ffparams_t* idef)
{
    std::vector<real> grid;
    int   i, j, k, atnr;
    real  c6, c6i, c6j;

    atnr = idef->atnr;
    grid.resize(2 * atnr * atnr);
    for (i = k = 0; (i < atnr); i++)
    {
        for (j = 0; (j < atnr); j++, k++)
        {
            c6i  = idef->iparams[i * (atnr + 1)].lj.c6;
            c6j  = idef->iparams[j * (atnr + 1)].lj.c6;
            c6   = std::sqrt(c6i * c6j);
            grid[2 * (atnr * i + j)] = c6 * 6.0;
        }
    }

    return grid;
}

//! Generate Coulomb and/or Van der Waals Ewald long-range correction tables
static void init_ewald_f_table(const interaction_const_t& ic,
                               const real                 tableExtensionLength,
                               EwaldCorrectionTables&     coulombTables,
                               EwaldCorrectionTables&     vdwTables)
{
    const bool useCoulombTable = EEL_PME_EWALD(ic.eeltype);
    const bool useVdwTable     = EVDW_PME(ic.vdwtype);

    const real tableScale = ewald_spline3_table_scale(ic, useCoulombTable, useVdwTable);

    real tableLen = ic.rcoulomb;
    if (useCoulombTable && tableExtensionLength > 0.0)
    {
        tableLen = ic.rcoulomb + tableExtensionLength;
    }
    const int tableSize = static_cast<int>(tableLen * tableScale) + 2;

    if (useCoulombTable)
    {
        coulombTables =
                generateEwaldCorrectionTables(tableSize, tableScale, ic.ewaldcoeff_q, v_q_ewald_lr);
    }

    if (useVdwTable)
    {
        vdwTables = generateEwaldCorrectionTables(tableSize, tableScale, ic.ewaldcoeff_lj, v_lj_ewald_lr);
    }
}
//@}

namespace gmx
{
namespace test
{
namespace
{

//! Number of atoms used in these tests.
constexpr int c_numAtoms = 2;

/*! \brief Output from nonbonded fep kernel
 *
 */
struct OutputQuantities
{
    OutputQuantities() :
        energy(egNR),
        dvdLambda(efptNR, 0.0),
        fShift(N_IVEC, { 0.0, 0.0, 0.0 }),
        f(c_numAtoms, { 0.0, 0.0, 0.0 })
    {
    }

    //! Energies of this interaction (size EgNR)
    gmx_grppairener_t energy;
    //! Derivative with respect to lambda (size efptNR)
    std::vector<real> dvdLambda;
    //! Shift force vectors (size N_IVEC)
    std::vector<RVec> fShift;
    //! Forces (size c_numAtoms)
    PaddedVector<RVec> f;
};

/*! \brief Utility to check the output from nonbonded test
 *
 * \param[in] checker Reference checker
 * \param[in] output  The output from the test to check
 */
void checkOutput(TestReferenceChecker* checker, const OutputQuantities& output)
{
    checker->checkReal(output.energy.ener[egLJSR][0], "EVdw ");
    checker->checkReal(output.energy.ener[egCOULSR][0], "ECoul ");
    checker->checkReal(output.dvdLambda[efptCOUL], "dVdlCoul ");
    checker->checkReal(output.dvdLambda[efptVDW], "dVdlVdw ");

    checker->checkSequence(std::begin(output.f), std::end(output.f), "Forces");

    auto shiftForcesChecker = checker->checkCompound("Shift-Forces", "Shift-forces");
    shiftForcesChecker.checkVector(output.fShift[0], "Central");
}

class InteractionConstHelper
{
public:
    InteractionConstHelper() {}

    //! init data to construct interaction_const
    void initInteractionConst(int coulType, int vdwType, int vdwMod)
    {
        coulType_ = coulType;
        vdwType_  = vdwType;
        vdwMod_   = vdwMod;

        // initialize correction tables
        interaction_const_t tmp;
        tmp.ewaldcoeff_q  = calc_ewaldcoeff_q(1.0, 1.0e-5);
        tmp.ewaldcoeff_lj = calc_ewaldcoeff_lj(1.0, 1.0e-5);
        tmp.eeltype       = coulType;
        tmp.vdwtype       = vdwType;

        init_ewald_f_table(tmp, 1.0, coulombTables_, vdwTables_);
    }

    /*! \brief Setup interaction_const_t
     *
     * \param[in]  fepVals t_lambda struct of fep values
     * \parma[out] ic      interaction_const_t pointer with data
     */
    void getInteractionConst(const t_lambda& fepVals, interaction_const_t* ic)
    {
        ic->softCoreParameters = std::unique_ptr<interaction_const_t::SoftCoreParameters>(
                new interaction_const_t::SoftCoreParameters(fepVals));

        ic->coulombEwaldTables  = std::unique_ptr<EwaldCorrectionTables>(new EwaldCorrectionTables);
        *ic->coulombEwaldTables = coulombTables_;

        ic->vdwEwaldTables  = std::unique_ptr<EwaldCorrectionTables>(new EwaldCorrectionTables);
        *ic->vdwEwaldTables = vdwTables_;

        // set coulomb and vdw types
        ic->eeltype      = coulType_;
        ic->vdwtype      = vdwType_;
        ic->vdw_modifier = vdwMod_;

        // some non default parameters used in this testcase
        ic->epsfac = ONE_4PI_EPS0 * 0.25;
        ic->k_rf    = 0.0;
        ic->c_rf    = 1.0;
        ic->sh_ewald = 1.0e-5;
        ic->sh_lj_ewald = -1.0;
        ic->dispersion_shift.cpot = -1.0;
        ic->repulsion_shift.cpot = -1.0;
    }

private:
    //! correction tables
    EwaldCorrectionTables coulombTables_;
    EwaldCorrectionTables vdwTables_;

    //! coulomb and vdw type specifiers
    int coulType_;
    int vdwType_;
    int vdwMod_;
};


/* \brief Utility class to setup forcerec
 *
 * This helper takes care of handling the neccessary pointers that are kept
 * at various places and in various forms in the forcerec hierarchy such
 * that this class can safely be used.
 *
 * Data is only initialized as necessary for the nonbonded kernel to work!
 */
class ForcerecHelper
{
public:
    ForcerecHelper()
    {
        fepVals_.sc_alpha         = 0.3;
        fepVals_.sc_power         = 1;
        fepVals_.sc_r_power       = 6.0;
        fepVals_.sc_sigma         = 0.3;
        fepVals_.sc_sigma_min     = 0.3;
        fepVals_.bScCoul          = true;
        fepVals_.softcoreFunction = SoftcoreType::None;
    }

    //! initialize data structure to construct forcerec
    void initForcerec(const gmx_ffparams_t* idef, int coulType, int vdwType, int vdwMod)
    {
        icHelper_.initInteractionConst(coulType, vdwType, vdwMod);
        nbfp_        = mk_nbfp(idef);
        ljPmeC6Grid_ = make_ljpme_c6grid(idef);
    }

    void setSoftcore(const SoftcoreType softcoreType)
    {
        fepVals_.softcoreFunction = softcoreType;
    }

    //! get forcerec data as wanted by the nonbonded kernel
    void getForcerec(t_forcerec* fr, interaction_const_t* ic)
    {
        // set data in ic
        icHelper_.getInteractionConst(fepVals_, ic);

        // set data in fr
        fr->ljpme_c6grid = ljPmeC6Grid_.data();
        fr->nbfp         = nbfp_;
        snew(fr->shift_vec, N_IVEC);
        fr->ic = ic;
    }

private:

    InteractionConstHelper icHelper_;
    std::vector<real>      ljPmeC6Grid_;
    std::vector<real>      nbfp_;
    t_lambda               fepVals_;
};

/*! \brief Utility structure to hold atoms data
 *
 * 2 atoms system, with atom 2 being deleted from the system for fep.
 */
struct AtomData
{
    AtomData()
    {
        idef.atnr = 2;
        idef.iparams.resize(4);
        idef.iparams[0].lj = { 0.001458, 1.0062882e-6 };
        idef.iparams[1].lj = { 0.0, 0.0};
        idef.iparams[2].lj = { 0.0, 0.0};
        idef.iparams[3].lj = { 0.0, 0.0};
    }

    // forcefield parameters
    gmx_ffparams_t idef;

    // atom data
    std::vector<real> chargeA = { 1.0, -1.0 };
    std::vector<real> chargeB = { 1.0, 0.0 };
    std::vector<int>  typeA   = { 0, 0 };
    std::vector<int>  typeB   = { 0, 1 };

    // neighbourhood information
    std::vector<int> iAtoms     = { 0, 1 };
    std::vector<int> jAtoms     = { 0, 1, 1 };
    std::vector<int> jIndex     = { 0, 2, 3 };
    std::vector<int> shift      = { 0, 0 };
    std::vector<int> gid        = { 0, 0 };
    char             exclFep[3] = { 0000, 0001, 0000 };

    /*! \brief Setup utility
     *
     * \param[out] mdatoms t_mdatoms
     * \param[out] nbl     t_nblist
     */
    void fillAtoms(t_mdatoms* mdatoms, t_nblist* nbl)
    {
        mdatoms->chargeA = chargeA.data();
        mdatoms->chargeB = chargeB.data();
        mdatoms->typeA   = typeA.data();
        mdatoms->typeB   = typeB.data();

        nbl->nri      = 2;
        nbl->nrj      = 3;
        nbl->iinr     = iAtoms.data();
        nbl->jindex   = jIndex.data();
        nbl->jjnr     = jAtoms.data();
        nbl->shift    = shift.data();
        nbl->gid      = gid.data();
        nbl->excl_fep = exclFep;
    }
};

/*! \brief Input structure for nonbonded fep kernel
 */
struct ListInput
{
public:
    //! Function type
    int fType = F_LJ;
    //! Tolerance for float evaluation
    float floatToler = 1e-6;
    //! Tolerance for double evaluation
    double doubleToler = 1e-8;
    //! atom parameters
    AtomData atoms;
    //! forcerec helper
    ForcerecHelper frHelper;

    friend std::ostream& operator<<(std::ostream& out, const ListInput& input);

    //! Constructor
    ListInput() {}

    /*! \brief Constructor with tolerance
     *
     * \param[in] ftol Single precision tolerance
     * \param[in] dtol Double precision tolerance
     */
    ListInput(float ftol, double dtol)
    {
        floatToler = ftol;
        doubleToler = dtol;
    }

    /*! \brief Set parameters for nonbonded interaction
     *
     * \param[in] coulType coulomb type
     * \param[in] vdwType  vdw type
     * \param[in] vdwMod   vdw potential modifier
     */
    ListInput setInteraction(int coulType, int vdwType, int vdwMod)
    {
        frHelper.initForcerec(&atoms.idef, coulType, vdwType, vdwMod);
        return *this;
    }
};

//! Prints the interaction and parameters to a stream
std::ostream& operator<<(std::ostream& out, const ListInput& input)
{
    using std::endl;
    out << "Function type " << input.fType << " called " << interaction_function[input.fType].name
        << " ie. labelled '" << interaction_function[input.fType].longname << "' in an energy file"
        << endl;

    // Organize to print the legacy C union t_iparams, whose
    // relevant contents vary with fType.
    StringOutputStream stream;
    {
        TextWriter writer(&stream);
        for (auto ip : input.atoms.idef.iparams)
        {
            printInteractionParameters(&writer, input.fType, ip);
        }
    }
    out << "Function parameters " << stream.toString();
    return out;
}


class NonbondedFepTest :
    public ::testing::TestWithParam<std::tuple<ListInput, PaddedVector<RVec>>>
{
protected:
    PaddedVector<RVec>     x_;
    ListInput              input_;
    TestReferenceData      refData_;
    TestReferenceChecker   checker_;

    NonbondedFepTest() : checker_(refData_.rootChecker())
    {
        input_   = std::get<0>(GetParam());
        x_       = std::get<1>(GetParam());

        test::FloatingPointTolerance tolerance(input_.floatToler, input_.doubleToler, 1.0e-6,
                                               1.0e-12, 10000, 100, false);
        checker_.setDefaultTolerance(tolerance);
    }

    void testOneIfunc(TestReferenceChecker* checker, const real lambda, SoftcoreType softcoreType)
    {
        input_.frHelper.setSoftcore(softcoreType);

        // get forcerec and interaction_const
        t_forcerec fr;
        interaction_const_t ic;
        input_.frHelper.getForcerec(&fr, &ic);

        // atom data
        t_mdatoms mdatoms;
        t_nblist  nbl;
        input_.atoms.fillAtoms(&mdatoms, &nbl);

        // force buffers and kernel data get pointed here:
        OutputQuantities output;
        std::vector<real> lambdas(efptNR, lambda);

        // fep kernel data
        int doNBFlags = 0;
        doNBFlags |= GMX_NONBONDED_DO_FORCE;
        doNBFlags |= GMX_NONBONDED_DO_SHIFTFORCE;
        doNBFlags |= GMX_NONBONDED_DO_POTENTIAL;

        nb_kernel_data_t kernel_data;
        kernel_data.flags          = doNBFlags;
        kernel_data.lambda         = lambdas.data();
        kernel_data.dvdl           = output.dvdLambda.data();
        kernel_data.energygrp_elec = output.energy.ener[egCOULSR].data();
        kernel_data.energygrp_vdw  = output.energy.ener[egLJSR].data();

        // force buffers
        bool unusedBool = true; // this bool has no effect in the kernel
        gmx::ForceWithShiftForces forces(output.f.arrayRefWithPadding(), unusedBool, output.fShift);

        // dummy counter
        t_nrnb nrnb;

        // run fep kernel
        gmx_nb_free_energy_kernel(&nbl, x_.rvec_array(), &forces, &fr, &mdatoms, &kernel_data, &nrnb);

        checkOutput(checker, output);
    }

    void testIfunc()
    {
        const int numLambdas = 3;
        for (int i = 0; i < numLambdas; ++i)
        {
            const real lambda       = i / (numLambdas - 1.0);
            for (SoftcoreType c : EnumerationWrapper<SoftcoreType>{})
            {
                auto lambdaChecker = checker_.checkCompound("Lambda", toString(lambda));
                auto softcoreChecker = lambdaChecker.checkCompound("Softcore", c_softcoreTypeNames[c]);
                testOneIfunc(&softcoreChecker, lambda, c);
            }
        }
    }
};

TEST_P(NonbondedFepTest, Ifunc)
{
    testIfunc();
}

//! configurations to test
std::vector<ListInput> c_interaction = {
    { ListInput(1e-6, 1e-8).setInteraction(eelCUT, evdwCUT, eintmodNONE) },
    { ListInput(1e-6, 1e-8).setInteraction(eelCUT, evdwCUT, eintmodPOTSWITCH) },
    { ListInput(1e-6, 1e-8).setInteraction(eelPME, evdwPME, eintmodNONE) }
};

//! Coordinates for testing
std::vector<PaddedVector<RVec>> c_coordinates = {
    { { 1.0, 1.0, 1.0 }, { 1.1, 1.15, 1.2 } }
};

INSTANTIATE_TEST_CASE_P(NBInteraction,
                        NonbondedFepTest,
                        ::testing::Combine(::testing::ValuesIn(c_interaction),
                                           ::testing::ValuesIn(c_coordinates)));

} // namespace

} // namespace test

} // namespace gmx
