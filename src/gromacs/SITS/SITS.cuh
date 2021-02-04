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
#include "gromacs/utility/arrayref.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/range.h"
#include "gromacs/utility/real.h"

enum SITS_MODE
{
    SIMPLE_SITS_MODE    = 0x00000001,
    CLASSICAL_SITS_MODE = 0x00000002,
};

struct SITS
{
private:
    FILE* sits_enerd_log = NULL;

    struct FC_BALL_INFORMATION
    {
        float fc_ball     = 1.;   // simpleSITS��ǿ������
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
        int sits_mode = 0; //ѡ��SITSģʽ
        int atom_numbers = 0; //��ϵ������ԭ����������һ��Ҫ��MD_INFORMATIONһ����������Ҫ
        int protein_atom_numbers =
                0; //����SITS��ǿ��ԭ������ԭ������һ��Ҫ��[0-protein_atom_numbers)��ģ������ҪԤ������
        int max_neighbor_numbers = 800; //�����һ����neighbor_list�����һ��
    } info;

    float*         d_total_pp_atom_energy      = NULL; // AA������
    float*         d_total_ww_atom_energy      = NULL; // BB������
    float*         d_total_protein_water_atom_energy = NULL; // AB������
    gmx::ArrayRefWithPadding<gmx::RVec> force_pw = NULL; //���ڼ�¼AB����ԭ�ӽ�����������

    int init_SITS(CONTROLLER* controller, int atom_numbers);

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