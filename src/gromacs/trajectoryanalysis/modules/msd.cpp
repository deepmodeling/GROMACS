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
/*! \internal \file
 * \brief
 * Defines the trajectory analysis module for mean squared displacement calculations.
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include "gmxpre.h"

#include "msd.h"

#include <numeric>

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/modules/average.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/analysisdata/paralleloptions.h"
#include "gromacs/fileio/oenv.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/statistics/statistics.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility.h"

namespace gmx::analysismodules
{

namespace
{

// Convert nm^2/ps to 10e-5 cm^2/s
constexpr double c_diffusionConversionFactor = 1000.0;
// Used in diffusion coefficient calculations
constexpr double c_3DdiffusionDimensionFactor = 6.0;
constexpr double c_2DdiffusionDimensionFactor = 4.0;
constexpr double c_1DdiffusionDimensionFactor = 2.0;


//! Proxy to a MsdData tau column. Supports only push_back.
class MsdColumnProxy
{
public:
    MsdColumnProxy(std::vector<double>& column) : column_(column) {}

    void push_back(double value) { column_.push_back(value); }

private:
    std::vector<double>& column_;
};


//! \brief Mean Squared Displacement data accumulator
//!
//! This class is used to accumulate individual MSD data points
//! and emit tau-averaged results once data is finished collecting.
//!
//! Data columns per tau are accessed via operator[], which always guarantees
//! a column is initialized and returns a proxy to the column that can push data.
class MsdData
{
public:
    //! Returns a proxy to the column for the given tau index. Guarantees that the column
    //! is initialized.
    MsdColumnProxy operator[](size_t index)
    {
        if (msds_.size() <= index)
        {
            msds_.resize(index + 1);
        }
        return MsdColumnProxy(msds_[index]);
    }
    //! \brief Compute per-tau MSDs averaged over all added points.
    //!
    //! The resulting vector is size(max tau index). Any indices
    //! that have no data points have MSD set to 0.
    //!
    //! \return Average MSD per tau
    [[nodiscard]] std::vector<real> averageMsds() const;

private:
    //! Results - first indexed by tau, then data points
    std::vector<std::vector<double>> msds_;
};


std::vector<real> MsdData::averageMsds() const
{
    std::vector<real> msdSums;
    msdSums.reserve(msds_.size());
    for (gmx::ArrayRef<const double> msdValues : msds_)
    {
        if (msdValues.empty())
        {
            msdSums.push_back(0.0);
            continue;
        }
        msdSums.push_back(std::accumulate(msdValues.begin(), msdValues.end(), 0.0, std::plus<>())
                          / msdValues.size());
    }
    return msdSums;
}

//! \brief Calculates 1,2, or 3D distance for two vectors.
//!
//! \todo Remove NOLINTs once clang-tidy is updated to v11, it should be able to handle constexpr.
//!
//! \tparam x If true, calculate x dimension of displacement
//! \tparam y If true, calculate y dimension of displacement
//! \tparam z If true, calculate z dimension of displacement
//! \param c1 First point
//! \param c2 Second point
//! \return Euclidian distance for the given dimension.
template<bool x, bool y, bool z>
inline double calcSingleSquaredDistance(const RVec c1, const RVec c2)
{
    const DVec firstCoords  = c1.toDVec();
    const DVec secondCoords = c2.toDVec();
    double     result       = 0;
    if constexpr (x)
    {
        result += (firstCoords[XX] - secondCoords[XX]) * (firstCoords[XX] - secondCoords[XX]);
    }
    if constexpr (y) // NOLINT: clang-tidy-9 can't handle if constexpr (https://bugs.llvm.org/show_bug.cgi?id=32203)
    {
        result += (firstCoords[YY] - secondCoords[YY]) * (firstCoords[YY] - secondCoords[YY]);
    }
    if constexpr (z) // NOLINT
    {
        result += (firstCoords[ZZ] - secondCoords[ZZ]) * (firstCoords[ZZ] - secondCoords[ZZ]);
    }
    return result; // NOLINT
}

//! \brief Calculate average displacement between sets of points
//!
//! Each displacement c1[i] - c2[i] is calculated and the distances
//! are averaged.
//!
//! \tparam x If true, calculate x dimension of displacement
//! \tparam y If true, calculate y dimension of displacement
//! \tparam z If true, calculate z dimension of displacement
//! \param c1 First vector
//! \param c2 Second vector
//! \param num_vals
//! \return Per-particle averaged distance
template<bool x, bool y, bool z>
double calcAverageDisplacement(const RVec* c1, const RVec* c2, const int num_vals)
{
    double result = 0;
    for (int i = 0; i < num_vals; i++)
    {
        result += calcSingleSquaredDistance<x, y, z>(c1[i], c2[i]);
    }
    return result / num_vals;
}


//! Describes 1D MSDs, in the given dimension.
enum class SingleDimDiffType : int
{
    X = 0,
    Y,
    Z,
    Unused,
    Count,
};

//! Describes 2D MSDs, in the plane normal to the given dimension.
enum class TwoDimDiffType : int
{
    NormalToX = 0,
    NormalToY,
    NormalToZ,
    Unused,
    Count,
};


//! Holds per-group coordinates, analysis, and results.
struct MsdGroupData
{
    explicit MsdGroupData(const Selection& inputSel) : sel(inputSel) {}

    //! Selection associated with this group.
    const Selection& sel;

    // Coordinate storage
    //! Stored coordinates, indexed by frame then atom number.
    std::vector<std::vector<RVec>> frames;
    //! Frame n-1 - used for removing PBC jumps.
    std::vector<RVec> previousFrame;

    // Result accumulation and calculation
    //! MSD result accumulator
    MsdData msds;
    //! Collector for processed MSD averages per tau
    std::vector<real> msdSums;
    //! Fitted diffusion coefficient
    real diffusionCoefficient = 0.0;
    //! Uncertainty of diffusion coefficient
    double sigma = 0.0;
};

//! Holds data needed for MSD calculations for a single molecule, if requested.
struct MoleculeData
{
    int atomCount = 0;
    //! Total mass.
    double mass = 0;
    //! MSD accumulator and calculator for the molecule
    MsdData msdData;
    //! Calculated diffusion coefficient
    real diffusionCoefficient = 0;
};

} // namespace

//! \brief Implements the gmx msd module
//!
//! \todo Implement -(no)mw. Right now, all calculations are mass-weighted with -mol, and not otherwise
//! \todo Implement -tensor for full MSD tensor calculation
//! \todo Implement -rmcomm for total-frame COM removal
//! \todo Implement -pdb for molecule B factors
//! \todo Implement -maxtau option proposed at https://gitlab.com/gromacs/gromacs/-/issues/3870
//! \todo Update help text as options are added and clarifications decided on at https://gitlab.com/gromacs/gromacs/-/issues/3869
class Msd : public TrajectoryAnalysisModule
{
public:
    Msd();
    ~Msd() override;

    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void initAfterFirstFrame(const TrajectoryAnalysisSettings& settings, const t_trxframe& fr) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;
    void analyzeFrame(int                           frameNumber,
                      const t_trxframe&             frame,
                      t_pbc*                        pbc,
                      TrajectoryAnalysisModuleData* pdata) override;
    void finishAnalysis(int nframes) override;
    void writeOutput() override;

private:
    //! Selections for MSD output
    SelectionList selections_;

    // MSD dimensionality related quantities
    //! MSD type information, for -type {x,y,z}
    SingleDimDiffType singleDimType_ = SingleDimDiffType::Unused;
    //! MSD type information, for -lateral {x,y,z}
    TwoDimDiffType twoDimType_ = TwoDimDiffType::Unused;
    //! Diffusion coefficient conversion factor
    double diffusionCoefficientDimensionFactor_ = c_3DdiffusionDimensionFactor;
    //! Method used to calculate MSD - changes based on dimensonality.
    std::function<double(const RVec*, const RVec*, int)> calcMsd_ =
            calcAverageDisplacement<true, true, true>;

    //! Picoseconds between restarts
    double trestart_ = 10.0;
    //! Initial time
    double t0_ = 0;
    //! Inter-frame delta-t
    std::optional<double> dt_ = std::nullopt;

    //! First tau value to fit from for diffusion coefficient, defaults to 0.1 * max tau
    real beginFit_ = -1.0;
    //! Final tau value to fit to for diffusion coefficient, defaults to 0.9 * max tau
    real endFit_ = -1.0;

    //! All selection group-specific data stored here.
    std::vector<MsdGroupData> groupData_;

    //! Time of stored frames only.
    std::vector<double> frameTimes_;
    //! Time of all frames.
    std::vector<double> times_;
    //! Taus for output - won't know the size until the end.
    std::vector<double> taus_;
    //! Tau indices for fitting.
    size_t beginFitIndex_ = 0;
    size_t endFitIndex_   = 0;

    // MSD per-molecule stuff
    //! Are we doing molecule COM-based MSDs?
    bool molSelected_ = false;
    //! Per molecule topology information and MSD accumulators.
    std::vector<MoleculeData> molecules_;
    //! Atom index -> mol index map
    std::vector<int> moleculeIndexMappings_;

    // Output stuff
    AnalysisData msdPlotData_;
    AnalysisData msdMoleculePlotData_;

    AnalysisDataPlotSettings plotSettings_;
    //! Per-tau MSDs for each selected group
    std::string output_;
    //! Per molecule diffusion coefficients if -mol is selected.
    std::string       moleculeOutput_;
    gmx_output_env_t* oenv_ = nullptr;
};


Msd::Msd() = default;
Msd::~Msd()
{
    output_env_done(oenv_);
}


void Msd::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] computes the mean square displacement (MSD) of atoms from",
        "a set of initial positions. This provides an easy way to compute",
        "the diffusion constant using the Einstein relation.",
        "The time between the reference points for the MSD calculation",
        "is set with [TT]-trestart[tt].",
        "The diffusion constant is calculated by least squares fitting a",
        "straight line (D*t + c) through the MSD(t) from [TT]-beginfit[tt] to",
        "[TT]-endfit[tt] (note that t is time from the reference positions,",
        "not simulation time). An error estimate given, which is the difference",
        "of the diffusion coefficients obtained from fits over the two halves",
        "of the fit interval.[PAR]",
        "There are three, mutually exclusive, options to determine different",
        "types of mean square displacement: [TT]-type[tt], [TT]-lateral[tt]",
        "and [TT]-ten[tt]. Option [TT]-ten[tt] writes the full MSD tensor for",
        "each group, the order in the output is: trace xx yy zz yx zx zy.[PAR]",
        "If [TT]-mol[tt] is set, [THISMODULE] plots the MSD for individual molecules",
        "(including making molecules whole across periodic boundaries): ",
        "for each individual molecule a diffusion constant is computed for ",
        "its center of mass. The chosen index group will be split into ",
        "molecules. With -mol, only one index group can be selected.[PAR]",
        "The diffusion coefficient is determined by linear regression of the MSD.",
        "When [TT]-beginfit[tt] is -1, fitting starts at 10%",
        "and when [TT]-endfit[tt] is -1, fitting goes to 90%.",
        "Using this option one also gets an accurate error estimate",
        "based on the statistics between individual molecules.",
        "Note that this diffusion coefficient and error estimate are only",
        "accurate when the MSD is completely linear between",
        "[TT]-beginfit[tt] and [TT]-endfit[tt].[PAR]",
    };
    settings->setHelpText(desc);

    // Selections
    options->addOption(SelectionOption("sel")
                               .storeVector(&selections_)
                               .required()
                               .onlyStatic()
                               .multiValue()
                               .description("Selections to compute MSDs for from the reference"));

    // Select MSD type - defaults to 3D if neither option is selected.
    EnumerationArray<SingleDimDiffType, const char*> enumTypeNames = { "x", "y", "z", "unselected" };
    EnumerationArray<TwoDimDiffType, const char*> enumLateralNames = { "x", "y", "z", "unselected" };
    options->addOption(EnumOption<SingleDimDiffType>("type")
                               .enumValue(enumTypeNames)
                               .store(&singleDimType_)
                               .defaultValue(SingleDimDiffType::Unused));
    options->addOption(EnumOption<TwoDimDiffType>("lateral")
                               .enumValue(enumLateralNames)
                               .store(&twoDimType_)
                               .defaultValue(TwoDimDiffType::Unused));

    options->addOption(DoubleOption("trestart")
                               .description("Time between restarting points in trajectory (ps)")
                               .defaultValue(10.0)
                               .store(&trestart_));
    options->addOption(RealOption("beginfit").description("").store(&beginFit_));
    options->addOption(RealOption("endfit").description("").store(&endFit_));

    // Output options
    options->addOption(FileNameOption("o")
                               .filetype(eftPlot)
                               .outputFile()
                               .store(&output_)
                               .defaultBasename("msdout")
                               .description("MSD output"));
    options->addOption(
            FileNameOption("mol")
                    .filetype(eftPlot)
                    .outputFile()
                    .store(&moleculeOutput_)
                    .storeIsSet(&molSelected_)
                    .defaultBasename("diff_mol")
                    .description("Report diffusion coefficients for each molecule in selection"));
}

void Msd::initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top)
{
    plotSettings_ = settings.plotSettings();
    // Initial parameter consistency checks.
    if (singleDimType_ != SingleDimDiffType::Unused && twoDimType_ != TwoDimDiffType::Unused)
    {
        std::string errorMessage =
                "Options -type and -lateral are mutually exclusive. Choose one or neither (for 3D "
                "MSDs).";
        GMX_THROW(InconsistentInputError(errorMessage.c_str()));
    }
    if (selections_.size() > 1 && molSelected_)
    {
        std::string errorMessage =
                "Cannot have multiple groups selected with -sel when using -mol.";
        GMX_THROW(InconsistentInputError(errorMessage.c_str()));
    }

    output_env_init(
            &oenv_, getProgramContext(), settings.timeUnit(), FALSE, settings.plotSettings().plotFormat(), 0);

    const int numSelections = selections_.size();
    // Accumulated frames and results
    for (int i = 0; i < numSelections; i++)
    {
        groupData_.emplace_back(selections_[i]);
    }

    // Enumeration helpers for dispatching the right MSD calculation type.
    const EnumerationArray<SingleDimDiffType, std::function<double(const RVec*, const RVec*, int)>>
            oneDimensionalMsdFunctions = { calcAverageDisplacement<true, false, false>,
                                           calcAverageDisplacement<false, true, false>,
                                           calcAverageDisplacement<false, false, true> };
    const EnumerationArray<TwoDimDiffType, std::function<double(const RVec*, const RVec*, int)>>
            twoDimensionalMsdFunctions = { calcAverageDisplacement<false, true, true>,
                                           calcAverageDisplacement<true, false, true>,
                                           calcAverageDisplacement<true, true, false> };

    // Parse dimensionality and assign the MSD calculating function.
    // Note if we don't hit either of these cases, we're computing 3D MSDs.
    if (singleDimType_ != SingleDimDiffType::Unused)
    {
        calcMsd_                             = oneDimensionalMsdFunctions[singleDimType_];
        diffusionCoefficientDimensionFactor_ = c_1DdiffusionDimensionFactor;
    }
    else if (twoDimType_ != TwoDimDiffType::Unused)
    {
        calcMsd_                             = twoDimensionalMsdFunctions[twoDimType_];
        diffusionCoefficientDimensionFactor_ = c_2DdiffusionDimensionFactor;
    }

    // TODO validate that we have mol info and not atom only - and masses, and topology.
    if (molSelected_)
    {
        Selection& sel  = selections_[0];
        const int  nMol = sel.initOriginalIdsToGroup(top.mtop(), INDEX_MOL);

        gmx::ArrayRef<const int> mappedIds = selections_[0].mappedIds();
        moleculeIndexMappings_.resize(selections_[0].posCount());
        std::copy(mappedIds.begin(), mappedIds.end(), moleculeIndexMappings_.begin());

        // Precalculate each molecules mass for speeding up COM calculations.
        ArrayRef<const real> masses = sel.masses();

        molecules_.resize(nMol);
        for (int i = 0; i < sel.posCount(); i++)
        {
            molecules_[mappedIds[i]].atomCount++;
            molecules_[mappedIds[i]].mass += masses[i];
        }
    }
}

void Msd::initAfterFirstFrame(const TrajectoryAnalysisSettings gmx_unused& settings, const t_trxframe& fr)
{
    t0_ = std::round(fr.time);
    for (MsdGroupData& msdData : groupData_)
    {
        msdData.previousFrame.resize(molSelected_ ? molecules_.size() : msdData.sel.posCount());
    }
}

//! Constructs the coordinates to calculate MSDs for a given selection. If individual molecules
//! are requested, molecular center-of-masses are returned.
static std::vector<RVec> buildCoordinates(const Selection&             sel,
                                          ArrayRef<const MoleculeData> molecules,
                                          ArrayRef<const int>          moleculeIndexMapping)
{
    // If not molecule based, we work on the individual coordinates of the selection.
    if (molecules.empty())
    {
        return { sel.coordinates().begin(), sel.coordinates().end() };
    }
    // Do COM gathering for group 0 to get mol stuff. Note that per-molecule PBC removal is
    // already done. First create a clear buffer.
    std::vector<RVec> moleculePositions(molecules.size(), { 0.0, 0.0, 0.0 });

    // Sum up all positions
    gmx::ArrayRef<const real> masses = sel.masses();
    for (int i = 0; i < sel.posCount(); i++)
    {
        const int moleculeIndex = moleculeIndexMapping[i];
        // accumulate ri * mi, and do division at the end to minimize number of divisions.
        moleculePositions[moleculeIndex] += RVec(sel.position(i).x()) * masses[i];
    }
    // Divide accumulated mass * positions to get COM, reaccumulate in mol_masses.
    std::transform(moleculePositions.begin(),
                   moleculePositions.end(),
                   molecules.begin(),
                   moleculePositions.begin(),
                   [](const RVec& position, const MoleculeData& molecule) -> RVec {
                       return position / molecule.mass;
                   });
    return moleculePositions;
}

//! Removes jumps across periodic boundaries for currentFrame, based on the positions in
//! previousFrame. Updates currentCoords in place.
static void removePbcJumps(ArrayRef<RVec> currentCoords, ArrayRef<const RVec> previousCoords, t_pbc* pbc)
{
    // There are two types of "pbc removal" in gmx msd. The first happens in the trajectoryanalysis
    // framework, which makes molecules whole across periodic boundaries and is done
    // automatically where the inputs support it. This lambda performs the second PBC correction, where
    // any "jump" across periodic boundaries BETWEEN FRAMES is put back. The order of these
    // operations is important - since the first transformation may only apply to part of a
    // molecule (e.g., one half in/out of the box is put on one side of the box), the
    // subsequent step needs to be applied to the molecule COM rather than individual atoms, or
    // we'd have a clash where the per-mol PBC removal moves an atom that gets put back into
    // it's original position by the second transformation. Therefore, this second transformation
    // is applied *after* per molecule coordinates have been consolidated into COMs.
    auto pbcRemover = [pbc](RVec in, RVec prev) {
        rvec dx;
        pbc_dx(pbc, in, prev, dx);
        return prev + dx;
    };
    std::transform(
            currentCoords.begin(), currentCoords.end(), previousCoords.begin(), currentCoords.begin(), pbcRemover);
}


void Msd::analyzeFrame(int frameNumber, const t_trxframe& frame, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata)
{
    const real time = std::round(frame.time);
    // Need to populate dt on frame 2;
    if (!dt_.has_value() && !times_.empty())
    {
        dt_ = time - times_[0];
    }

    // Each frame gets an entry in times, but frameTimes only updates if we're at a restart.
    times_.push_back(time);
    if (bRmod(time, t0_, trestart_))
    {
        frameTimes_.push_back(time);
    }

    // Each frame will get a tau between it and frame 0, and all other frame combos should be
    // covered by this.
    // \todo this will no longer hold exactly when maxtau is added
    taus_.push_back(time - times_[0]);

    for (MsdGroupData& msdData : groupData_)
    {
        //NOLINTNEXTLINE(readability-static-accessed-through-instance)
        const Selection& sel = pdata->parallelSelection(msdData.sel);

        std::vector<RVec> coords = buildCoordinates(sel, molecules_, moleculeIndexMappings_);

        if (frameNumber > 0)
        {
            removePbcJumps(coords, msdData.previousFrame, pbc);
        }

        // Update "previous frame" for next rounds pbc handling. Note that for msd mol these coords
        // are actually the molecule COM.
        std::copy(coords.begin(), coords.end(), msdData.previousFrame.begin());

        // For each preceding frame, calculate tau and do comparison.
        for (size_t i = 0; i < msdData.frames.size(); i++)
        {
            double  tau      = time - frameTimes_[i];
            int64_t tauIndex = gmx::roundToInt64(tau / *dt_);
            msdData.msds[tauIndex].push_back(
                    calcMsd_(coords.data(), msdData.frames[i].data(), coords.size()));

            for (size_t molInd = 0; molInd < molecules_.size(); molInd++)
            {
                molecules_[molInd].msdData[tauIndex].push_back(
                        calcMsd_(&coords[molInd], &msdData.frames[i][molInd], 1));
            }
        }


        // We only store the frame for the future if it's a restart per -trestart.
        if (bRmod(time, t0_, trestart_))
        {
            msdData.frames.push_back(std::move(coords));
        }
    }
}

//! Calculate the tau index for fitting. If userFitTau < 0, uses the default fraction of max tau.
static size_t calculateFitIndex(const int    userFitTau,
                                const double defaultTauFraction,
                                const int    numTaus,
                                const double dt)
{
    if (userFitTau < 0)
    {
        return gmx::roundToInt((numTaus - 1) * defaultTauFraction);
    }
    return std::min<size_t>(numTaus - 1, gmx::roundToInt(static_cast<double>(userFitTau) / dt));
}

static constexpr double c_defaultStartFitIndexFraction = 0.1;
static constexpr double c_defaultEndFitIndexFraction   = 0.9;

void Msd::finishAnalysis(int gmx_unused nframes)
{
    beginFitIndex_ = calculateFitIndex(beginFit_, c_defaultStartFitIndexFraction, taus_.size(), *dt_);
    endFitIndex_   = calculateFitIndex(endFit_, c_defaultEndFitIndexFraction, taus_.size(), *dt_);
    const int numTausForFit = 1 + endFitIndex_ - beginFitIndex_;

    // These aren't used, except for correlationCoefficient, which is used to estimate error if
    // enough points are available.
    real b = 0.0, correlationCoefficient = 0.0, chiSquared = 0.0;

    for (MsdGroupData& msdData : groupData_)
    {
        msdData.msdSums = msdData.msds.averageMsds();

        if (numTausForFit >= 4)
        {
            const int halfNumTaus         = numTausForFit / 2;
            const int secondaryStartIndex = beginFitIndex_ + halfNumTaus;
            // Split the fit in 2, and compare the results of each fit;
            real a = 0.0, a2 = 0.0;
            lsq_y_ax_b_xdouble(halfNumTaus,
                               &taus_[beginFitIndex_],
                               &msdData.msdSums[beginFitIndex_],
                               &a,
                               &b,
                               &correlationCoefficient,
                               &chiSquared);
            lsq_y_ax_b_xdouble(halfNumTaus,
                               &taus_[secondaryStartIndex],
                               &msdData.msdSums[secondaryStartIndex],
                               &a2,
                               &b,
                               &correlationCoefficient,
                               &chiSquared);
            msdData.sigma = std::abs(a - a2);
        }
        lsq_y_ax_b_xdouble(numTausForFit,
                           &taus_[beginFitIndex_],
                           &msdData.msdSums[beginFitIndex_],
                           &msdData.diffusionCoefficient,
                           &b,
                           &correlationCoefficient,
                           &chiSquared);
        msdData.diffusionCoefficient *= c_diffusionConversionFactor / diffusionCoefficientDimensionFactor_;
        msdData.sigma *= c_diffusionConversionFactor / diffusionCoefficientDimensionFactor_;
    }

    for (MoleculeData& molecule : molecules_)
    {
        std::vector<real> msds = molecule.msdData.averageMsds();
        lsq_y_ax_b_xdouble(numTausForFit,
                           &taus_[beginFitIndex_],
                           &msds[beginFitIndex_],
                           &molecule.diffusionCoefficient,
                           &b,
                           &correlationCoefficient,
                           &chiSquared);
        molecule.diffusionCoefficient *= c_diffusionConversionFactor / diffusionCoefficientDimensionFactor_;
    }
}

void Msd::writeOutput()
{
    // AnalysisData currently doesn't support changing column counts after analysis has started.
    // We can't determine the number of tau values until the trajectory is fully read, so analysis
    // data construction and plotting are done here.
    AnalysisDataPlotModulePointer msdPlotModule(new AnalysisDataPlotModule(plotSettings_));
    msdPlotModule->setFileName(output_);
    msdPlotModule->setTitle("Mean Squared Displacement");
    msdPlotModule->setXLabel("tau (ps)");
    msdPlotModule->setYLabel(R"(MSD (nm\\S2\\N))");
    msdPlotModule->setYFormat(10, 6, 'g');
    for (const auto& group : groupData_)
    {
        const real D = group.diffusionCoefficient;
        if (D > 0.01 && D < 1e4)
        {
            msdPlotModule->appendLegend(formatString(
                    "D[%10s] = %.4f (+/- %.4f) (1e-5 cm^2/s)", group.sel.name(), D, group.sigma));
        }
        else
        {
            msdPlotModule->appendLegend(formatString(
                    "D[%10s] = %.4g (+/- %.4f) (1e-5 cm^2/s)", group.sel.name(), D, group.sigma));
        }
    }
    msdPlotData_.addModule(msdPlotModule);
    msdPlotData_.setDataSetCount(groupData_.size());
    for (size_t i = 0; i < groupData_.size(); i++)
    {
        msdPlotData_.setColumnCount(i, 1);
    }
    AnalysisDataHandle dh = msdPlotData_.startData({});
    for (size_t tauIndex = 0; tauIndex < taus_.size(); tauIndex++)
    {
        dh.startFrame(tauIndex, taus_[tauIndex]);
        for (size_t dataSetIndex = 0; dataSetIndex < groupData_.size(); dataSetIndex++)
        {
            dh.selectDataSet(dataSetIndex);
            dh.setPoint(0, groupData_[dataSetIndex].msdSums[tauIndex]);
        }
        dh.finishFrame();
    }
    dh.finishData();

    if (molSelected_)
    {
        AnalysisDataPlotModulePointer molPlotModule(new AnalysisDataPlotModule(plotSettings_));
        molPlotModule->setFileName(moleculeOutput_);
        molPlotModule->setTitle("Mean Squared Displacement / Molecule");
        molPlotModule->setXLabel("Molecule");
        molPlotModule->setYLabel("D(1e-5 cm^2/s)");
        molPlotModule->setYFormat(10, 0, 'g');
        msdMoleculePlotData_.addModule(molPlotModule);
        msdMoleculePlotData_.setDataSetCount(1);
        msdMoleculePlotData_.setColumnCount(0, 1);
        AnalysisDataHandle molDh = msdMoleculePlotData_.startData({});
        for (size_t moleculeIndex = 0; moleculeIndex < molecules_.size(); moleculeIndex++)
        {
            molDh.startFrame(moleculeIndex, moleculeIndex);
            molDh.setPoint(0, molecules_[moleculeIndex].diffusionCoefficient);
            molDh.finishFrame();
        }
        molDh.finishData();
    }
}


const char                      MsdInfo::name[]             = "msd";
const char                      MsdInfo::shortDescription[] = "Compute mean squared displacements";
TrajectoryAnalysisModulePointer MsdInfo::create()
{
    return TrajectoryAnalysisModulePointer(new Msd);
}


} // namespace gmx::analysismodules
