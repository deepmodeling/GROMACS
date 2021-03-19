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

    //��ʱ����
    int record_count = 0; //��¼����
    int reset = 1; // record��ʱ�򣬵�һ�κͺ��湫ʽ��һ��������������������������

    //���Ʊ���
    int   record_interval = 1;   //ÿ��1����¼һ������
    int   update_interval = 100; //ÿ��100������һ��nk
    int   output_interval = 100;
    int   niter           = 50;
    int   k_numbers;             //���ֶ��ٸ�����
    float beta0;                 //�����¶ȶ�Ӧ��beta
    bool  constant_nk = false;   // sits�Ƿ��������nk
    //�ļ�
    FILE*       nk_traj_file;   //��¼nk�仯���ļ�
    std::string nk_rest_file;   //��¼���һ֡nk���ļ�
    FILE*       norm_traj_file; //��¼log_norm�仯���ļ�
    std::string norm_rest_file; //��¼���һ֡log_norm���ļ�
    FILE*       sits_enerd_out;

    //����ʱ�����Զ�fc_ballֱ��������+ fb_shift���е��ڣ�
    float fb_shift;
    //Ҳ���ԶԽ���������ʹ��ǿ��������ʱֵΪ energy_multiple * ԭʼ���� + energy_shift;
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
    //     float move_length = 0.01; // simpleSITS��fcball������ߵ���󲽳�
    //     float fc_max      = 1.2;  //���ߵ����ޣ���Ӧ��͵��¶�T������1/factor
    //     float fc_min      = 0.5;  //���ߵ����ޣ���Ӧ��ߵ��¶�

    //     int random_seed = 0; //������ߵĳ�ʼ���ӣ����ܺ�������������ӳ�ͻ
    //     float* fc_pdf = NULL; //��ɢ��fc�����ܶȼ�¼�б����ڿ���fc�ĸ��ʷֲ���cpu�ϴ洢
    //     int grid_numbers = 1000; //��ɢ�б�ĸ�����Ŀ

    //     float current_fc_probability = 0.01; //��ʼ�ĸ����ܶ�,Ҫ��0
    //     float get_fc_probability(float pos); //��ô�ʱfc_ball��ֵpos��Ӧ�ĸ����ܶ�


    //     int is_constant_fc_ball = 0; //��¼�Ƿ��ǹ̶�fcballֵ����ģ�⣨������ϵ���⣩
    //     float constant_fc_ball = 1.0; //�̶���fcballֵ
    // } simple_info;
    // void fc_ball_random_walk(); // simple mode��������漸����������fc_ball��һ������ƶ�
    // vsits_cal_modeical_Update_Info(int steps); // classical info����Ҫ����nkNk
public:
    std::unique_ptr<sits_atomdata_t> sits_at;

    // gmx::ArrayRefWithPadding<gmx::RVec> force_tot = NULL; //���ڼ�¼AB����ԭ�ӽ�����������
    // gmx::ArrayRefWithPadding<gmx::RVec> force_pw = NULL; //���ڼ�¼AB����ԭ�ӽ�����������

    gmx_sits_cuda_t* gpu_sits;

    void sits_atomdata_set_energygroups(std::vector<int> cginfo);

    void print_sitsvals(bool bFirstTime = true, int step = 0);

    void sits_update_effectiveU(float* Epot) {*Epot -= (sits_at->ene_recorded + sits_at->gfsum / sits_at->beta0);}

    // Interactions enhanced: (bond, angle), dihedral, LJ-SR, PME_Direct-SR, LJ-14, Coul-14;
    // Not enhanced: LJ-Recip, Coul-Recip, Disp. Corr., (bond, angle)

    void clear_sits_energy_force();

    //�ı�frc��ʹ����Ҫ��ǿ��frc��ѡ������ǿ�����ڹ��õ�frc������ⲽfrc��ǿ��Ҫ�ŵ��պü���������Ҫ��ǿ��frc�ĺ����·�����������ǿ��Ӧ����ǿ����ģ��frc
    //��ˣ���ϵ�����п��ܵ�frc��Ҫ�������ǿ��frc������˺��������㲻Ӧ��ǿ��frc
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