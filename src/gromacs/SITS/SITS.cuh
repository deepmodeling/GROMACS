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

struct SITS
{
private:
    FILE* sits_enerd_log = NULL;

    struct FC_BALL_INFORMATION
    {
        float fc_ball     = 1.;   // simpleSITS enhancing factor
        float move_length = 0.01; // simpleSITS��fcball������ߵ���󲽳�
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
    struct SITS_INFORMATION
    {
        int sits_calc_mode = 0; //ѡ��SITSģʽ
        int sits_enh_mode = PP_AND_PW; //
    } info;

    gmx::ArrayRefWithPadding<gmx::RVec> force_tot = NULL; //���ڼ�¼AB����ԭ�ӽ�����������
    gmx::ArrayRefWithPadding<gmx::RVec> force_pw = NULL; //���ڼ�¼AB����ԭ�ӽ�����������

    struct SITS_cuda
    {
        int sits_calc_mode = 0; // SITS calculation mode: classical or simple
        int sits_enh_mode = PP_AND_PW; // SITS enhancing region: solvate, intramolecular or intermolecular

        float* d_Ener_pp; // AA������
        float* d_Ener_ww; // BB������
        float* d_Ener_pw; // AB������

        float3* d_force_tot;
        float3* d_force_pw;

        float3* d_force_nbat_tot;
        float3* d_force_nbat_pw;
    } gpu_sits;

    int init_SITS(int na);

    void gpu_init_SITS();

    //����ͳ������ѧԭ����ǿ����ֻ��Ҫ������ƥ��ͺã���˲���Ҫ�����е��໥���ý�����ǿ
    //Ŀǰ��Ҫ��ǿ������bond,angle,dihedral,LJ��PME_Direct,nb_14���������ֲ���ǿ����Ӱ������ȷ�ԣ�
    //�ڼ������SITS��ص�����֮ǰ�����һ��

    void Clear_SITS_Energy();
    //���������������м�Ϳ�������������Ҫ��ǿ������������ģ���ԭ���������㺯��
    //�ڼ������SITS������������������
    void Calculate_Total_SITS_Energy(int is_fprintf);

    //�ı�frc��ʹ����Ҫ��ǿ��frc��ѡ������ǿ�����ڹ��õ�frc������ⲽfrc��ǿ��Ҫ�ŵ��պü���������Ҫ��ǿ��frc�ĺ����·�����������ǿ��Ӧ����ǿ����ģ��frc
    //��ˣ���ϵ�����п��ܵ�frc��Ҫ�������ǿ��frc������˺��������㲻Ӧ��ǿ��frc
    void SITS_Enhanced_Force(VECTOR* frc);

    //����������ͷ�SITS�ڵĸ���ָ��
    void Clear_SITS();
};