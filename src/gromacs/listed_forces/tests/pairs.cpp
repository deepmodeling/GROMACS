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
 * \brief Implements test of 1-4 interactions
 *
 * This test is copied from the bonded interactions test and slightly
 * modified since 'do_pairs' takes a different set of arguments than
 * 'calculateSimpleBond'. To keep the test setup uncluttered this test is
 * therefore not merged into the bonded test but implemented standalone.
 *
 * The test setup consists of 2 atom pairs that are tested in an fep setting
 * (vanishing charge and lennard-jones parameters of one atom) and without
 * fep. Placement of the atoms in the box is such that shift-forces and pbc
 * paths in do_pairs are covered.
 *
 * The reference values were generated with Gromacs 2022-dev using the
 * following configuration:
 *
 * Precision:          double
 * Memory model:       64 bit
 * MPI library:        thread_mpi
 * OpenMP support:     enabled (GMX_OPENMP_MAX_THREADS = 64)
 * GPU support:        disabled
 * SIMD instructions:  AVX2_256
 * FFT library:        fftpack (built-in)
 * RDTSCP usage:       enabled
 * TNG support:        enabled
 * Hwloc support:      disabled
 * Tracing support:    disabled
 * C compiler:         GNU 10.2.1
 * C compiler flags:   -mavx2 -mfma -fno-inline -g
 * C++ compiler:       GNU 10.2.1
 * C++ compiler flags: -mavx2 -mfma -fno-inline -fopenmp -g
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_listed_forces
 */
#include "gmxpre.h"

#include "gromacs/listed_forces/bonded.h"

#include <cmath>

#include <memory>
#include <unordered_map>

#include <gtest/gtest.h>

#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/listed_forces/pairs.h"
#include "gromacs/math/paddedvector.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/nblist.h"
#include "gromacs/tables/forcetable.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/idef.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

//! Number of atoms used in these tests.
constexpr int c_numAtoms = 2;

/*! \brief Output from pairs kernels
 *
 */
struct OutputQuantities
{
    OutputQuantities(int energyGroup) :
        energy(energyGroup), dvdLambda(efptNR, 0.0) {}

    //! Energy of this interaction
    gmx_grppairener_t energy;
    //! Derivative with respect to lambda
    std::vector<real> dvdLambda;
    //! Shift vectors
    rvec fShift[N_IVEC] = { { 0 } };
    //! Forces
    alignas(GMX_REAL_MAX_SIMD_WIDTH * sizeof(real)) rvec4 f[c_numAtoms] = { { 0 } };
};

/*! \brief Utility to check the output from pairs tests
 *
 * \param[in] checker Reference checker
 * \param[in] output  The output from the test to check
 * \param[in] bondedKernelFlavor  Flavor for determining what output to check
 */
void checkOutput(TestReferenceChecker*    checker,
                 const OutputQuantities&  output,
                 const BondedKernelFlavor bondedKernelFlavor)
{
    if (computeEnergy(bondedKernelFlavor))
    {
        checker->checkReal(output.energy.ener[egLJ14][0], "Epot ");
        checker->checkReal(output.dvdLambda[efptCOUL], "dVdlCoul ");
        checker->checkReal(output.dvdLambda[efptVDW], "dVdlVdw ");
    }
    checker->checkSequence(std::begin(output.f), std::end(output.f), "Forces");
}

/* \brief Utility class to setup forcerec and interaction parameters
 *
 * Data is only initialized as necessary for the 1-4 interactions to work!
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
    }

    //! initialize data structure to construct forcerec
    void initForcerec(bool haveFep)
    {
        haveFep_ = haveFep;
    }

    //! set use of simd if available
    void setUseSimd(const bool useSimd)
    {
        useSimd_ = useSimd;
    }

    //! set use mol pbc
    void setMolPBC(const bool haveMolPBC)
    {
        haveMolPBC_ = haveMolPBC;
    }

    //! get forcerec data as wanted by the 1-4 interactions
    void getForcerec(t_forcerec* fr, interaction_const_t* ic)
    {
        // set data in ic
        ic->softCoreParameters = std::make_unique<interaction_const_t::SoftCoreParameters>(fepVals_);

        // set data in fr
        fr->pairsTable       = make_tables(nullptr, ic, nullptr, 2.9, GMX_MAKETABLES_14ONLY);
        fr->efep             = haveFep_ ? efepYES : efepNO;
        fr->fudgeQQ          = 0.5;
        fr->ic               = ic;
        fr->use_simd_kernels = useSimd_;
        fr->bMolPBC          = haveMolPBC_;
    }

private:
    bool         haveFep_;
    bool         useSimd_     = false;
    bool         haveMolPBC_  = false;
    t_lambda     fepVals_;
};

/*! \brief Input structure for listed forces tests
 */
struct ListInput
{
public:
    //! Function type
    int fType = -1;
    //! do fep
    bool fep = false;
    //! Tolerance for float evaluation
    float floatToler = 1e-6;
    //! Tolerance for double evaluation
    double doubleToler = 1e-8;
    //! Interaction parameters
    t_iparams iparams = { { 0 } };
    //! forcerec setup helper
    ForcerecHelper frHelper;

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

    /*! \brief Set parameters for 1-4 interaction
     *
     * Fep is used if either c6A != c6B or c12A != c12B.
     *
     * \param[in] c6 state A
     * \param[in] c12 state A
     * \param[in] c6 state B
     * \param[in] c12 state B
     */
    ListInput set14Interaction(real c6A, real c12A, real c6B, real c12B)
    {
        fType             = F_LJ14;
        fep               = (c6A != c6B || c12A != c12B);
        iparams.lj14.c6A  = c6A;
        iparams.lj14.c12A = c12A;
        iparams.lj14.c6B  = c6B;
        iparams.lj14.c12B = c12B;

        frHelper.initForcerec(fep);

        return *this;
    }
};

class ListedForcesPairsTest :
    public ::testing::TestWithParam<std::tuple<ListInput, PaddedVector<RVec>, PbcType>>
{
protected:
    matrix                 box_;
    t_pbc                  pbc_;
    PaddedVector<RVec>     x_;
    PbcType                pbcType_;
    ListInput              input_;
    TestReferenceData      refData_;
    TestReferenceChecker   checker_;

    ListedForcesPairsTest() : checker_(refData_.rootChecker())
    {
        input_   = std::get<0>(GetParam());
        x_       = std::get<1>(GetParam());
        pbcType_ = std::get<2>(GetParam());
        clear_mat(box_);
        box_[0][0] = box_[1][1] = box_[2][2] = 1.0;
        set_pbc(&pbc_, pbcType_, box_);

        FloatingPointTolerance tolerance = relativeToleranceAsPrecisionDependentFloatingPoint(1.0, input_.floatToler, input_.doubleToler);

        checker_.setDefaultTolerance(tolerance);
    }

    void testOneIfunc(TestReferenceChecker* checker, const real lambda)
    {
        SCOPED_TRACE(std::string("Testing PBC type: ") + c_pbcTypeNames[pbcType_]);

        // 'definition of pairs' is a concatenation of #npairs (here 2)
        // 'nAtomsPerPair+1'-tuples (fType a_0 a_i ... a_nAtomsPerPair)
        std::vector<t_iatom >       iatoms     = { 0, 1, 2, 0, 0, 2 };

        std::vector<int>            ddgatindex = { 0, 1, 2 };
        std::vector<real>           chargeA    = { 1.0, -0.5, -0.5 };
        std::vector<real>           chargeB    = { 0.0, 0.0 , 0.0};
        std::vector<unsigned short> egrp       = { 0, 0, 0};
        t_mdatoms                   mdatoms    = { 0 };

        mdatoms.chargeA                        = chargeA.data();
        mdatoms.chargeB                        = chargeB.data();
        mdatoms.cENER                          = egrp.data();
        // nPerturbed is not decisive for fep to be used; it is overruled by
        // other conditions in do_pairs_general; just here to not segfault
        // upon query
        mdatoms.nPerturbed = 0;

        if (pbcType_ != PbcType::No)
        {
            input_.frHelper.setMolPBC(true);
        }

        std::vector<BondedKernelFlavor> flavors = { BondedKernelFlavor::ForcesAndVirialAndEnergy };

        if (!input_.fep || lambda == 0)
        {
            input_.frHelper.setUseSimd(true);
            flavors.push_back(BondedKernelFlavor::ForcesSimdWhenAvailable);
        }

        for (const auto flavor : flavors)
        {
            SCOPED_TRACE("Testing bonded kernel flavor: " + c_bondedKernelFlavorStrings[flavor]);

            StepWorkload stepWork;
            stepWork.computeVirial = computeVirial(flavor);
            stepWork.computeEnergy = computeEnergy(flavor);

            bool havePerturbedInteractions = input_.fep;
            if (flavor == BondedKernelFlavor::ForcesSimdWhenAvailable)
            {
                havePerturbedInteractions = false;
            }

            t_forcerec fr;
            interaction_const_t ic;
            input_.frHelper.getForcerec(&fr, &ic);

            OutputQuantities output(egLJ14);
            std::vector<real> lambdas(efptNR, lambda);

            do_pairs(input_.fType, iatoms.size(), iatoms.data(), &input_.iparams,
                     as_rvec_array(x_.data()), output.f, output.fShift, &pbc_, lambdas.data(),
                     output.dvdLambda.data(), &mdatoms, &fr, havePerturbedInteractions,
                     stepWork, &output.energy, ddgatindex.data());

            checkOutput(checker, output, flavor);
            auto shiftForcesChecker = checker->checkCompound("Shift-Forces", "Shift-forces");

            if (computeVirial(flavor))
            {
                shiftForcesChecker.checkVector(output.fShift[CENTRAL], "Central");
            }
            else
            {
                // Permit omitting to compare shift forces with
                // reference data when that is useless.
                shiftForcesChecker.disableUnusedEntriesCheck();
            }
        }
    }

    void testIfunc()
    {
        TestReferenceChecker thisChecker =
                checker_.checkCompound("FunctionType", interaction_function[input_.fType].name)
                        .checkCompound("FEP", (input_.fep ? "Yes" : "No"));

        if (input_.fep)
        {
            const int numLambdas = 3;
            for (int i = 0; i < numLambdas; ++i)
            {
                const real lambda       = i / (numLambdas - 1.0);
                auto       lambdaChecker = thisChecker.checkCompound("Lambda", toString(lambda));
                testOneIfunc(&lambdaChecker, lambda);
            }
        }
        else
        {
            testOneIfunc(&thisChecker, 0.0);
        }
    }
};

TEST_P(ListedForcesPairsTest, Ifunc)
{
    testIfunc();
}

//! Function types for testing 1-4 interaction. Add new terms at the end.
std::vector<ListInput> c_14Interaction = {
    { ListInput(1e-5, 1e-7).set14Interaction(0.001458, 1.0062882e-6, 0.0, 0.0) },
    { ListInput(1e-5, 1e-7).set14Interaction(0.001458, 1.0062882e-6, 0.001458, 1.0062882e-6) }
};

//! PBC values for testing
std::vector<PbcType> c_pbcForTests = { PbcType::No, PbcType::XY, PbcType::Xyz };

/*! \brief Coordinates for testing 1-4 interaction
 *
 * Define coordinates for 3 atoms here, which will be used in 2 interactions.
 */
std::vector<PaddedVector<RVec>> c_coordinatesFor14Interaction = {
    { { 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 }, { 1.1, 1.2, 1.3 } }
};

INSTANTIATE_TEST_CASE_P(14Interaction,
                        ListedForcesPairsTest,
                        ::testing::Combine(::testing::ValuesIn(c_14Interaction),
                                           ::testing::ValuesIn(c_coordinatesFor14Interaction),
                                           ::testing::ValuesIn(c_pbcForTests)));

} // namespace

} // namespace test

} // namespace gmx
