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
        float fc_ball     = 1.;   // simpleSITS的强化因子
        float move_length = 0.01; // simpleSITS中fcball随机游走的最大步长
        float fc_max      = 1.2;  //游走的上限，对应最低的温度T正比于1/fc_ball
        float fc_min      = 0.5;  //游走的下限，对应最高的温度

        int random_seed = 0; //随机游走的初始种子，可能和其他程序的种子冲突
        float* fc_pdf = NULL; //离散的fc概率密度记录列表，用于控制fc的概率分布，cpu上存储
        int grid_numbers = 1000; //离散列表的格子数目

        float current_fc_probability = 0.01; //初始的概率密度,要非0
        float get_fc_probability(float pos); //获得此时fc_ball的值pos对应的概率密度
        void  fc_ball_random_walk(); //根据上面几个参数进行fc_ball的一次随机移动

        int is_constant_fc_ball = 0; //记录是否是固定fcball值进行模拟（用于体系初测）
        float constant_fc_ball = 1.0; //固定的fcball值
    } simple_info;

public:
    struct SITS_INFORMATION
    {
        int sits_mode = 0; //选择SITS模式
        int atom_numbers = 0; //体系的所有原子数，但不一定要和MD_INFORMATION一样，根据需要
        int protein_atom_numbers =
                0; //进行SITS增强的原子数，原子序数一定要是[0-protein_atom_numbers)里的，因此需要预先排序
        int max_neighbor_numbers = 800; //这个数一般与neighbor_list里面的一样
    } info;

    float*         d_total_pp_atom_energy      = NULL; // AA总能量
    float*         d_total_ww_atom_energy      = NULL; // BB总能量
    float*         d_total_protein_water_atom_energy = NULL; // AB总能量
    gmx::ArrayRefWithPadding<gmx::RVec> force_pw = NULL; //用于记录AB两类原子交叉项作用力

    int init_SITS(CONTROLLER* controller, int atom_numbers);

    //根据统计热力学原理，增强的力只需要和能量匹配就好，因此不需要对所有的相互作用进行增强
    //目前主要增强的力是bond,angle,dihedral,LJ，PME_Direct,nb_14。其他部分不增强（不影响结果正确性）
    //在计算各种SITS相关的能量之前先清空一次

    void Clear_SITS_Energy();
    //在上下两个函数中间就可以填入所有需要增强的能量计算子模块的原子能量计算函数
    //在计算完各SITS分能量后进行能量求和
    void Calculate_Total_SITS_Energy(int is_fprintf);

    //改变frc，使得需要增强的frc被选择性增强，由于共用的frc，因此这步frc增强需要放到刚好计算完所有要增强的frc的函数下方，而避免增强不应该增强的子模块frc
    //因此，体系的所有可能的frc需要先算待增强的frc，插入此函数，再算不应增强的frc
    void SITS_Enhanced_Force(VECTOR* frc);

    //程序结束后，释放SITS内的各个指针
    void Clear_SITS();
};