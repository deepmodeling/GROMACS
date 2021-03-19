#ifndef GMX_SITS_H
#define GMX_SITS_H

#include <cassert>
#include <cinttypes>
#include <csignal>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <memory>

#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/math/utilities.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/range.h"
#include "gromacs/utility/real.h"

#include "sits_gpu.h"
// #include "gromacs/sits/cuda/sits_cuda_types.h"
struct gmx_sits_cuda_t;

struct gmx_device_info_t;
struct gmx_domdec_zones_t;
struct gmx_enerdata_t;
struct gmx_hw_info_t;
struct gmx_mtop_t;
struct gmx_wallcycle;
struct t_commrec;
struct t_lambda;
struct t_mdatoms;
struct t_nrnb;
struct t_forcerec;
struct t_inputrec;

// enum class sits_cal_mode
// {
//     CLASSICAL_SITS; SIMPLE_SITS
// };

// enum class SITS_ENH_MODE
// {
//     PP_AND_PW; INTRA_MOL; INTER_MOL; ALL
// };

struct sits_atomdata_t
{
public:
    int   sits_cal_mode = 0;     // SITS Calculation mode: Classical / Simple
    int   sits_enh_mode = 0;     //
    bool  sits_enh_bias = false; //
    float pw_enh_factor = 0.5;

    //暂时变量
    int record_count = 0; //记录次数
    int reset = 1; // record的时候，第一次和后面公式不一样，这个变量是拿来控制这个的

    //控制变量
    int   record_interval = 1;   //每隔1步记录一次能量
    int   update_interval = 100; //每隔100步更新一次nk
    int   output_interval = 100;
    int   niter           = 50;
    int   k_numbers;             //划分多少个格子
    float beta0;                 //本身温度对应的beta
    bool  constant_nk = false;   // sits是否迭代更新nk
    //文件
    FILE*       nk_traj_file;   //记录nk变化的文件
    std::string nk_rest_file;   //记录最后一帧nk的文件
    FILE*       norm_traj_file; //记录log_norm变化的文件
    std::string norm_rest_file; //记录最后一帧log_norm的文件
    FILE*       sits_enerd_out;

    //计算时，可以对fc_ball直接修正，+ fb_shift进行调节，
    float fb_shift;
    //也可以对进行修正，使加强计算能量时值为 energy_multiple * 原始能量 + energy_shift;
    float energy_multiple;
    float energy_shift;

    int                  natoms;
    gmx::HostVector<int> energrp;

    // Derivations and physical quantities see:
    // \ref A selective integrated tempering method
    // \ref Self-adaptive enhanced sampling in the energy and trajectory spaces : Accelerated thermodynamics and kinetic calculations

    gmx::HostVector<real> beta_k;
    gmx::HostVector<real> wt_beta_k;
    real sum_beta_factor;
    real factor[2] = {1.0, 1.0};

    // Details of $n_k$ iteration see:
    // \ref An integrate-over-temperature approach for enhanced sampling

    // |   .cpp var    |  ylj .F90 var  |  Ref var
    // | ene_recorded  | vshift         | U
    // | gf            | gf             | log( n_k * exp(-beta_k * U) )
    // | gfsum         | gfsum          | log( Sum_(k=1)^N ( n_k * exp(-beta_k * U) ) )
    // | log_weight    | rb             | log of the weighting function
    // | log_mk_inv    | ratio          | log(m_k^-1)
    // | log_norm_old  | normlold       | W(j-1)
    // | log_norm      | norml          | W(j)
    // | log_pk        | rbfb           | log(p_k)
    // | log_nk        | fb             | log(n_k)

    real enerd[3] = {0.0, 0.0, 0.0};
    real ene_recorded;
    gmx::HostVector<real> gf;
    real gfsum;
    gmx::HostVector<real> log_weight;
    gmx::HostVector<real> log_mk_inv;
    gmx::HostVector<real> log_norm_old;
    gmx::HostVector<real> log_norm;
    gmx::HostVector<real> log_pk;
    gmx::HostVector<real> log_nk;

    sits_atomdata_t();
};

struct sits_t
{

private:
    // struct FC_BALL_INFORMATION
    // {
    //     float move_length = 0.01; // simpleSITS中fcball随机游走的最大步长
    //     float fc_max      = 1.2;  //游走的上限，对应最低的温度T正比于1/factor
    //     float fc_min      = 0.5;  //游走的下限，对应最高的温度

    //     int random_seed = 0; //随机游走的初始种子，可能和其他程序的种子冲突
    //     float* fc_pdf = NULL; //离散的fc概率密度记录列表，用于控制fc的概率分布，cpu上存储
    //     int grid_numbers = 1000; //离散列表的格子数目

    //     float current_fc_probability = 0.01; //初始的概率密度,要非0
    //     float get_fc_probability(float pos); //获得此时fc_ball的值pos对应的概率密度


    //     int is_constant_fc_ball = 0; //记录是否是固定fcball值进行模拟（用于体系初测）
    //     float constant_fc_ball = 1.0; //固定的fcball值
    // } simple_info;
    // void fc_ball_random_walk(); // simple mode里根据上面几个参数进行fc_ball的一次随机移动
    // vsits_cal_modeical_Update_Info(int steps); // classical info中需要迭代nkNk
public:
    std::unique_ptr<sits_atomdata_t> sits_at;

    // gmx::ArrayRefWithPadding<gmx::RVec> force_tot = NULL; //用于记录AB两类原子交叉项作用力
    // gmx::ArrayRefWithPadding<gmx::RVec> force_pw = NULL; //用于记录AB两类原子交叉项作用力

    gmx_sits_cuda_t* gpu_sits;

    void sits_atomdata_set_energygroups(std::vector<int> cginfo);

    void print_sitsvals(bool bFirstTime = true, int step = 0);

    void sits_update_effectiveU(float* Epot) {*Epot -= (sits_at->ene_recorded + sits_at->gfsum / sits_at->beta0);}

    // Interactions enhanced: (bond, angle), dihedral, LJ-SR, PME_Direct-SR, LJ-14, Coul-14;
    // Not enhanced: LJ-Recip, Coul-Recip, Disp. Corr., (bond, angle)

    void clear_sits_energy_force();

    //改变frc，使得需要增强的frc被选择性增强，由于共用的frc，因此这步frc增强需要放到刚好计算完所有要增强的frc的函数下方，而避免增强不应该增强的子模块frc
    //因此，体系的所有可能的frc需要先算待增强的frc，插入此函数，再算不应增强的frc
    void sits_enhance_force(int step);

    void sits_update_params(int step);

    //! Constructs an object from its components
    sits_t(std::unique_ptr<sits_atomdata_t> sits_at_in, gmx_sits_cuda_t* gpu_sits_ptr);

    ~sits_t();
};

namespace Sits
{

/*! \brief Creates an Sits object */
std::unique_ptr<sits_t> init_sits(
        //   const gmx::MDLogger&     mdlog,
        gmx_bool                 bFEP_SITS,
        const t_inputrec*        ir,
        const t_forcerec*        fr,
        const t_commrec*         cr,
        const gmx_hw_info_t&     hardwareInfo,
        const gmx_device_info_t* deviceInfo,
        const gmx_mtop_t*        mtop,
        gmx_wallcycle*           wcycle);
} // namespace Sits

#endif // GMX_SITS_H