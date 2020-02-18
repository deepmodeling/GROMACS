/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2011-2018, The GROMACS development team.
 * Copyright (c) 2019, by the GROMACS development team, led by
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
 * Implements gmx::analysismodules::FSC.
 *
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "fscaverage.h"

#include <numeric>
// #include <memory>
// #include <string>
#include <vector>

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/trajectory/trajectoryframe.h"

#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"

#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"

#include "gromacs/selection/selectionoption.h"
#include "gromacs/math/coordinatetransformation.h"
#include "gromacs/math/gausstransform.h"
#include "gromacs/math/exponentialmovingaverage.h"
#include "gromacs/math/fsc.h"
#include "gromacs/math/densityfit.h"

#include "gromacs/fileio/mrcdensitymap.h"

#include "gromacs/pbcutil/pbc.h"
#include "gromacs/compat/optional.h"

#include "gromacs/topology/atoms.h"

#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/applied_forces/densityfittingamplitudelookup.h"


namespace gmx
{

namespace analysismodules
{

namespace
{

GaussianSpreadKernelParameters::Shape makeSpreadKernel(real sigma, real nSigma, const ScaleCoordinates& scaleToLattice)
{
    RVec sigmaInLatticeCoordinates{ sigma, sigma, sigma };
    scaleToLattice({ &sigmaInLatticeCoordinates, &sigmaInLatticeCoordinates + 1 });
    return { DVec{ sigmaInLatticeCoordinates[XX], sigmaInLatticeCoordinates[YY],
                   sigmaInLatticeCoordinates[ZZ] },
             nSigma };
}

class FSCAvg : public TrajectoryAnalysisModule
{
public:
    FSCAvg() : amplitudeLookup_(amplitudeLookupMethod_)
    {
        registerAnalysisDataset(&fscCurve_, "fsc");
        registerAnalysisDataset(&fscAverage_, "fscAverage");
        registerAnalysisDataset(&fscMoveCurve_, "fscOfMovingAvgMap");
        registerAnalysisDataset(&fscMoveAverage_, "fscAverageOfMovingAvgMap");
        registerAnalysisDataset(&similarityScore_, "similarity");
    }

    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void optionsFinished(TrajectoryAnalysisSettings* settings) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;

    void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata) override;

    void finishAnalysis(int nframes) override;
    void writeOutput() override;

private:
    std::string fnDensity_;
    std::string fnFSC_;
    std::string fnFSCmove_;
    std::string fnFSCmoveavg_;
    std::string fnFSCAvg_;
    std::string fnSimilarity_;
    std::string fnAxis_;

    int numFscShells_;

    AnalysisData fscCurve_;
    AnalysisData fscAverage_;
    AnalysisData fscMoveCurve_;
    AnalysisData fscMoveAverage_;
    AnalysisData similarityScore_;

    // mdAtoms are needed for amplitude lookup
    t_mdatoms mdAtoms_;

    std::vector<ExponentialMovingAverage>               movingMapAverager_;
    MultiDimArray<std::vector<float>, dynamicExtents3D> movingMapData_;

    //! Indices of the atoms that shall be fit to the density
    std::vector<index> indices_;
    //! Determines with what weight atoms are spread
    DensityFittingAmplitudeMethod amplitudeLookupMethod_ = DensityFittingAmplitudeMethod::Unity;
    DensityFittingAmplitudeLookup amplitudeLookup_;
    //! The spreading width used for the gauss transform of atoms onto the density grid
    real gaussianTransformSpreadingWidth_ = 0.2;
    //! The spreading range for spreading atoms onto the grid in multiples of the spreading width
    real              gaussianTransformSpreadingRangeInMultiplesOfWidth_ = 4.0;
    std::vector<RVec> transformedCoordinates_;
    //! Normalize reference and simulated densities
    bool                                                normalizeDensities_ = true;
    Selection                                           refSel_;
    basic_mdspan<const float, dynamicExtents3D>         referenceDensity_;
    MultiDimArray<std::vector<float>, dynamicExtents3D> referenceDensityData_;
    compat::optional<TranslateAndScale>                 transformationToDensityLattice_;
    RVec                                                referenceDensityCenter_;
    compat::optional<GaussTransform3D>                  gaussTransform_;
    compat::optional<FourierShellCorrelation>           fsc_;
    std::vector<DensitySimilarityMeasure>               measure_;
    // Copy and assign disallowed by base.
};

void FSCAvg::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{

    static const char* const desc[] = { "[THISMODULE] calculates the average FSC." };

    settings->setHelpText(desc);

    options->addOption(FileNameOption("o")
                               .filetype(eftPlot)
                               .outputFile()
                               .required()
                               .store(&fnFSC_)
                               .defaultBasename("fsc")
                               .description("Fourier shell correlation as function of time"));

    options->addOption(
            FileNameOption("fscmove")
                    .filetype(eftPlot)
                    .outputFile()
                    .required()
                    .store(&fnFSCmove_)
                    .defaultBasename("fsc-moving-average-map")
                    .description("Fourier shell correlation of moving map as function of time"));

    options->addOption(
            FileNameOption("fscmoveavg")
                    .filetype(eftPlot)
                    .outputFile()
                    .required()
                    .store(&fnFSCmoveavg_)
                    .defaultBasename("fsc-average-moving-average-map")
                    .description(
                            "Fourier shell correlation average of moving map as function of time"));

    options->addOption(FileNameOption("avg")
                               .filetype(eftPlot)
                               .outputFile()
                               .required()
                               .store(&fnFSCAvg_)
                               .defaultBasename("fscavg")
                               .description("FSC average as function of time"));

    options->addOption(FileNameOption("similarity")
                               .filetype(eftPlot)
                               .outputFile()
                               .required()
                               .store(&fnSimilarity_)
                               .defaultBasename("similarity")
                               .description("Similarity to reference as function of time"));

    options->addOption(FileNameOption("ordinate-axis")
                               .filetype(eftPlot)
                               .outputFile()
                               .required()
                               .store(&fnAxis_)
                               .defaultBasename("axis")
                               .description("FSC x-axis"));

    options->addOption(EnumOption<DensityFittingAmplitudeMethod>("amplitude")
                               .enumValue(c_densityFittingAmplitudeMethodNames.m_elements)
                               .store(&amplitudeLookupMethod_));

    options->addOption(RealOption("gausswidth")
                               .store(&gaussianTransformSpreadingWidth_)
                               .description("Spreading Gaussian width to generate density in nm."));

    options->addOption(
            IntegerOption("shells").store(&numFscShells_).defaultValue(61).description("Number of FSC shells."));

    options->addOption(
            StringOption("refmap").store(&fnDensity_).description("The name of the reference density."));

    options->addOption(SelectionOption("sel").store(&refSel_).required().description(
            "Reference selection for FSC computation"));

    settings->setFlag(TrajectoryAnalysisSettings::efRequireTop);
}


void FSCAvg::optionsFinished(TrajectoryAnalysisSettings* /* settings */)
{
    MrcDensityMapOfFloatFromFileReader reader(fnDensity_);
    transformationToDensityLattice_.emplace(reader.transformationToDensityLattice());
    referenceDensityData_ = reader.densityDataCopy();
    const double norm = std::accumulate(begin(referenceDensityData_), end(referenceDensityData_), 0.);

    std::transform(begin(referenceDensityData_), end(referenceDensityData_),
                   begin(referenceDensityData_), [norm](real value) { return value / norm; });

    RVec unitVector = { 1, 0, 0 };
    transformationToDensityLattice_->scaleOperationOnly().inverseIgnoringZeroScale(
            { &unitVector, &unitVector + 1 });

    fsc_.emplace(referenceDensityData_.asConstView(), unitVector[0], numFscShells_);
    numFscShells_ = fsc_->numberOfShells();

    const auto axisfile = fopen(fnAxis_.c_str(), "w");
    for (ptrdiff_t i = 0; i < numFscShells_; i++)
    {
        fprintf(axisfile, "%10.10g\n", i * fsc_->spacing());
    }
    fclose(axisfile);

    // normalize the reference
    const real sumOfDensityData = std::accumulate(begin(referenceDensityData_.asView()),
                                                  end(referenceDensityData_.asView()), 0.);
    for (float& referenceDensityVoxel : referenceDensityData_.asView())
    {
        referenceDensityVoxel /= sumOfDensityData;
    }

    referenceDensity_ = referenceDensityData_.asConstView();

    for (const auto& method : EnumerationWrapper<DensitySimilarityMeasureMethod>{})
    {
        measure_.emplace_back(method, referenceDensity_);
    }

    referenceDensityCenter_ = { real(referenceDensity_.extent(XX)) / 2,
                                real(referenceDensity_.extent(YY)) / 2,
                                real(referenceDensity_.extent(ZZ)) / 2 };
    transformationToDensityLattice_->scaleOperationOnly().inverseIgnoringZeroScale(
            { &referenceDensityCenter_, &referenceDensityCenter_ + 1 });
    // correct the reference density center for a shift
    // if the reference density does not have its origin at (0,0,0)
    RVec referenceDensityOriginShift(0, 0, 0);
    (*transformationToDensityLattice_)({ &referenceDensityOriginShift, &referenceDensityOriginShift + 1 });
    transformationToDensityLattice_->scaleOperationOnly().inverseIgnoringZeroScale(
            { &referenceDensityOriginShift, &referenceDensityOriginShift + 1 });
    referenceDensityCenter_ -= referenceDensityOriginShift;

    GaussianSpreadKernelParameters::Shape spreadKernel = makeSpreadKernel(
            gaussianTransformSpreadingWidth_, gaussianTransformSpreadingRangeInMultiplesOfWidth_,
            transformationToDensityLattice_->scaleOperationOnly());

    gaussTransform_.emplace(GaussTransform3D(referenceDensity_.extents(), spreadKernel));

    movingMapData_.resize(referenceDensity_.extents());
    movingMapAverager_.resize(movingMapData_.asConstView().mapping().required_span_size(), { 10 });
    amplitudeLookup_ = DensityFittingAmplitudeLookup(amplitudeLookupMethod_);
}

void addPlotModule(AnalysisData*                   analysisData,
                   const AnalysisDataPlotSettings& plotSettings,
                   const std::string&              fn,
                   const std::string&              title,
                   const std::string&              ylabel,
                   int                             numberOfColumns)
{
    analysisData->setColumnCount(0, numberOfColumns);
    AnalysisDataPlotModulePointer plotModule = std::make_shared<AnalysisDataPlotModule>(plotSettings);
    plotModule->setFileName(fn);
    plotModule->setXAxisIsTime();
    plotModule->setTitle(title);
    plotModule->setYLabel(ylabel.c_str());
    plotModule->setYFormat(12, 6, 'g');
    analysisData->addModule(plotModule);
}

t_mdatoms mdatomsFromtAtoms(const t_atoms& atoms)
{
    t_mdatoms mdAtoms;

    mdAtoms.nr = atoms.nr;
    snew(mdAtoms.massT, mdAtoms.nr);
    snew(mdAtoms.chargeA, mdAtoms.nr);

    for (int i = 0; i < mdAtoms.nr; i++)
    {
        mdAtoms.massT[i]   = atoms.atom[i].m;
        mdAtoms.chargeA[i] = atoms.atom[i].q;
    }
    return mdAtoms;
}

void FSCAvg::initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top)
{
    addPlotModule(&fscCurve_, settings.plotSettings(), fnFSC_, "Fourier Shell Correlation", "FSC",
                  numFscShells_);
    addPlotModule(&fscMoveCurve_, settings.plotSettings(), fnFSCmove_,
                  "Fourier Shell Correlation Moving Map Average", "FSC", numFscShells_);
    addPlotModule(&fscAverage_, settings.plotSettings(), fnFSCAvg_,
                  "Fourier Shell Correlation Average", "FSC avg", numFscShells_);
    addPlotModule(&fscMoveAverage_, settings.plotSettings(), fnFSCmoveavg_,
                  "Fourier Shell Correlation Average Moving Map Average", "FSC avg", numFscShells_);
    addPlotModule(&similarityScore_, settings.plotSettings(), fnSimilarity_,
                  "Similarity to reference density", "similarity",
                  c_densitySimilarityMeasureMethodNames.size());
    mdAtoms_ = mdatomsFromtAtoms(*(top.atoms()));
}

void addData(int                           nr,
             real                          time,
             const AnalysisData&           analysisData,
             const std::vector<real>&      frameData,
             TrajectoryAnalysisModuleData* pdata)
{
    AnalysisDataHandle dh = pdata->dataHandle(analysisData);
    dh.startFrame(nr, time);
    dh.setPoints(0, frameData.size(), frameData.data(), true);
    dh.finishFrame();
}

void FSCAvg::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata)
{

    transformedCoordinates_.resize(refSel_.atomIndices().size());
    // pick and copy atom coordinates
    std::transform(std::cbegin(refSel_.atomIndices()), std::cend(refSel_.atomIndices()),
                   std::begin(transformedCoordinates_), [&fr](int index) { return fr.x[index]; });

    // pick periodic image that is closest to the center of the reference density
    for (RVec& x : transformedCoordinates_)
    {
        rvec dx;
        pbc_dx(pbc, x, referenceDensityCenter_, dx);
        x = referenceDensityCenter_ + dx;
    }

    // transform local atom coordinates to density grid coordinates
    (*transformationToDensityLattice_)(transformedCoordinates_);

    // spread atoms on grid
    gaussTransform_->setZero();
    std::vector<real> amplitudes = amplitudeLookup_(mdAtoms_, refSel_.atomIndices());

    if (normalizeDensities_)
    {
        real sum = std::accumulate(std::begin(amplitudes), std::end(amplitudes), 0.);
        for (real& amplitude : amplitudes)
        {
            amplitude /= sum;
        }
    }

    auto amplitudeIterator = amplitudes.cbegin();

    for (const auto& r : transformedCoordinates_)
    {
        gaussTransform_->add({ r, *amplitudeIterator });
        ++amplitudeIterator;
    }

    /* Calculate the moving map average */
    auto movingMapAveragerIterator = std::begin(movingMapAverager_);
    for (const auto voxelValue : gaussTransform_->constView())
    {
        movingMapAveragerIterator->updateWithDataPoint(voxelValue);
        ++movingMapAveragerIterator;
    }

    movingMapAveragerIterator = std::begin(movingMapAverager_);
    for (auto& voxelValue : movingMapData_)
    {
        voxelValue = movingMapAveragerIterator->biasCorrectedAverage();
        ++movingMapAveragerIterator;
    }

    /* Caluclate FSC curves from maps */
    const FourierShellCorrelationCurve& curve     = fsc_->fscCurve(gaussTransform_->constView());
    const FourierShellCorrelationCurve& moveCurve = fsc_->fscCurve(movingMapData_.asConstView());

    /* Store the results in analysis data */
    addData(frnr, fr.time, fscCurve_, curve.correlation, pdata);
    addData(frnr, fr.time, fscAverage_, fscAverage(curve), pdata);
    addData(frnr, fr.time, fscMoveCurve_, moveCurve.correlation, pdata);
    addData(frnr, fr.time, fscMoveAverage_, fscAverage(moveCurve), pdata);

    std::vector<real> similarityScores;
    for (const auto& measure : measure_)
    {
        similarityScores.push_back(measure.similarity(gaussTransform_->constView()));
    }
    addData(frnr, fr.time, similarityScore_, similarityScores, pdata);
}


void FSCAvg::finishAnalysis(int /*nframes*/) {}


void FSCAvg::writeOutput() {}

} // namespace

const char FSCAvgInfo::name[]             = "fsc";
const char FSCAvgInfo::shortDescription[] = "Calculate fourier shell correlation";

TrajectoryAnalysisModulePointer FSCAvgInfo::create()
{
    return TrajectoryAnalysisModulePointer(new FSCAvg);
}

} // namespace analysismodules

} // namespace gmx
