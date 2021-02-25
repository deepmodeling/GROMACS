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
		float move_length = 0.01;//simpleSITS��fcball������ߵ���󲽳�
		float fc_max = 1.2;//���ߵ����ޣ���Ӧ��͵��¶�T������1/fc_ball
		float fc_min = 0.5;//���ߵ����ޣ���Ӧ��ߵ��¶�

		int random_seed = 0;//������ߵĳ�ʼ���ӣ����ܺ�������������ӳ�ͻ
		float *fc_pdf = NULL;//��ɢ��fc�����ܶȼ�¼�б����ڿ���fc�ĸ��ʷֲ���cpu�ϴ洢
		int grid_numbers = 1000;//��ɢ�б�ĸ�����Ŀ

		float current_fc_probability=0.01;//��ʼ�ĸ����ܶ�,Ҫ��0
		float get_fc_probability(float pos);//��ô�ʱfc_ball��ֵpos��Ӧ�ĸ����ܶ�
		

		int is_constant_fc_ball=0;//��¼�Ƿ��ǹ̶�fcballֵ����ģ�⣨������ϵ���⣩
		float constant_fc_ball = 1.0;//�̶���fcballֵ
	}simple_info;
    void fc_ball_random_walk();//simple mode��������漸����������fc_ball��һ������ƶ�
	void SITS_Classical_Update_Info(int steps);//classical info����Ҫ��������Nk
public:
	struct CLASSICAL_SITS_INFORMATION
	{
	public:
		//��ʱ����
		int record_count = 0;     //��¼����
		int reset = 1;  //record��ʱ�򣬵�һ�κͺ��湫ʽ��һ��������������������������
	
		//���Ʊ���
		int record_interval = 1;  //ÿ��1����¼һ������
		int update_interval = 100; //ÿ��100������һ��nk
		int constant_nk = 0;         //sits�Ƿ��������nk
		int k_numbers;           //���ֶ��ٸ�����
		float beta0;             //�����¶ȶ�Ӧ��beta
		//�ļ�
		FILE *nk_traj_file; //��¼nk�仯���ļ�
		char nk_rest_file[256]; //��¼���һ֡nk���ļ�
		FILE *norm_traj_file; //��¼log_norm�仯���ļ�
		char norm_rest_file[256]; //��¼���һ֡log_norm���ļ�
		float *log_nk_recorded_cpu;  //��cpu�˵ļ�¼ֵ
		float *log_norm_recorded_cpu; //��cpuu�˵ļ�¼ֵ

		//����ʱ�����Զ�fc_ballֱ��������+ fb_shift���е��ڣ�
		float fb_shift;
		//Ҳ���ԶԽ���������ʹ��ǿ��������ʱֵΪ energy_multiple * ԭʼ���� + energy_shift;
		float energy_multiple;
		float energy_shift;

		////ԭ�����������������ƪ����
		////A selective integrated tempering method
		////Self-adaptive enhanced sampling in the energy and trajectory spaces : Accelerated thermodynamics and kinetic calculations

		float *beta_k;           
		float *NkExpBetakU;      
		float *Nk;            
		float *sum_a;
		float *sum_b;
		float *d_fc_ball;
		//xyj��cpp������-ylj��F90������-���׶�Ӧ
		//ene_recorded - vshift - ene
		//gf - gf - log( n_k * exp(-beta_k * ene) )
		//gfsum - gfsum - log( Sum_(k=1)^N ( log( n_k * exp(-beta_k * ene) ) ) )
		//log_weight - rb - log of the weighting function
		//log_mk_inverse - ratio - log(m_k^-1)
		//log_norm_old - normlold - W(j-1)
		//log_norm - norml - W(j)
		//log_pk - rbfb - log(p_k)
		//log_nk_inverse - pratio - log(n_k^-1)
		//log_nk - fb - log(n_k)
		float *ene_recorded;
		float *gf;
		float *gfsum;
		float *log_weight;
		float *log_mk_inverse;
		float *log_norm_old;
		float *log_norm;
		float *log_pk;
		float *log_nk_inverse;
		float *log_nk;

		void Export_Restart_Information_To_File();
	}classical_info;

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