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

#include "gromacs/sits/cuda/sits_cuda_types.h"

struct gmx_device_info_t;
struct gmx_domdec_zones_t;
struct gmx_enerdata_t;
struct gmx_hw_info_t;
struct gmx_mtop_t;
struct gmx_wallcycle;
struct interaction_const_t;
struct sits_t;
struct t_blocka;
struct t_commrec;
struct t_lambda;
struct t_mdatoms;
struct t_nrnb;
struct t_forcerec;
struct t_inputrec;

enum SITS_CALC_MODE
{
    SIMPLE_SITS    = 0x00000001,
    CLASSICAL_SITS = 0x00000002,
};

enum SITS_ENH_MODE
{
    PP_AND_PW;
    INTRA_MOL;
    INTER_MOL;
};

struct sits_info
{
    int sits_calc_mode = 0; //ѡ��sitsģʽ
    int sits_enh_mode = PP_AND_PW; //
    int sits_enh_bias = false; //
}

struct sits_t
{
public:
    //! Constructs an object from its components
    sits_t(std::unique_ptr<PairlistSets>     pairlistSets,
                       std::unique_ptr<PairSearch>       pairSearch,
                       std::unique_ptr<nbnxn_atomdata_t> nbat,
                       const Nbnxm::KernelSetup&         kernelSetup,
                       sits_cuda*                  gpu_sits,
                       gmx_wallcycle*                    wcycle);

    ~sits_t();

private:
    FILE* sits_enerd_log = NULL;

    struct FC_BALL_INFORMATION
    {
        float fc_ball     = 1.;   // simplesits enhancing factor
        float move_length = 0.01; // simplesits��fcball������ߵ���󲽳�
        float fc_max      = 1.2;  //���ߵ����ޣ���Ӧ��͵��¶�T������1/fc_ball
        float fc_min      = 0.5;  //���ߵ����ޣ���Ӧ��ߵ��¶�

        int random_seed = 0; //������ߵĳ�ʼ���ӣ����ܺ�������������ӳ�ͻ
        float* fc_pdf = NULL; //��ɢ��fc�����ܶȼ�¼�б����ڿ���fc�ĸ��ʷֲ���cpu�ϴ洢
        int grid_numbers = 1000; //��ɢ�б�ĸ�����Ŀ

        float current_fc_probability = 0.01; //��ʼ�ĸ����ܶ�,Ҫ��0
        float get_fc_probability(float pos); //��ô�ʱfc_ball��ֵpos��Ӧ�ĸ����ܶ�
        void  fc_ball_random_walk(); //�������漸����������fc_ball��һ������ƶ�

        int is_constant_fc_ball = 0; //��¼�Ƿ��ǹ̶�fcballֵ����ģ�⣨������ϵ���⣩
        float constant_fc_ball = 1.0; //�̶���fcballֵ
    } simple_info;

public:
    sits_info info;

    gmx::ArrayRefWithPadding<gmx::RVec> force_tot = NULL; //���ڼ�¼AB����ԭ�ӽ�����������
    gmx::ArrayRefWithPadding<gmx::RVec> force_pw = NULL; //���ڼ�¼AB����ԭ�ӽ�����������

    sits_cuda* gpu_sits;

    int init_sits(int na);

    void gpu_init_sits();

    //����ͳ������ѧԭ����ǿ����ֻ��Ҫ������ƥ��ͺã���˲���Ҫ�����е��໥���ý�����ǿ
    //Ŀǰ��Ҫ��ǿ������bond,angle,dihedral,LJ��PME_Direct,nb_14���������ֲ���ǿ����Ӱ������ȷ�ԣ�
    //�ڼ������sits��ص�����֮ǰ�����һ��

    void Clear_sits_Energy();
    //���������������м�Ϳ�������������Ҫ��ǿ������������ģ���ԭ���������㺯��
    //�ڼ������sits������������������
    void Calculate_Total_sits_Energy(int is_fprintf);

    //�ı�frc��ʹ����Ҫ��ǿ��frc��ѡ������ǿ�����ڹ��õ�frc������ⲽfrc��ǿ��Ҫ�ŵ��պü���������Ҫ��ǿ��frc�ĺ����·�����������ǿ��Ӧ����ǿ����ģ��frc
    //��ˣ���ϵ�����п��ܵ�frc��Ҫ�������ǿ��frc������˺��������㲻Ӧ��ǿ��frc
    void sits_Enhanced_Force(VECTOR* frc);

    //����������ͷ�sits�ڵĸ���ָ��
    void Clear_sits();
};

namespace Sits
{

/*! \brief Creates an Nbnxm object */
std::unique_ptr<sits_t> init_sits(const gmx::MDLogger&     mdlog,
                                  gmx_bool                 bFEP_SITS,
                                  const t_inputrec*        ir,
                                  const t_forcerec*        fr,
                                  const t_commrec*         cr,
                                  const gmx_hw_info_t&     hardwareInfo,
                                  const gmx_device_info_t* deviceInfo,
                                  const gmx_mtop_t*        mtop,
                                  matrix                   box,
                                  gmx_wallcycle*           wcycle);

} // namespace Sits