#include "SITS.cuh"

int SITS::init_SITS(CONTROLLER* controller, int atom_numbers)
{
    printf("\nInitial Selective Integrated Tempering Sampling:\n");
    if (controller[0].Command_Exist("sits_mode"))
    {
        if (strcmp(controller[0].Command("sits_mode"), "simple") == 0)
        {
            info.sits_mode = SIMPLE_SITS_MODE;
            printf("	Choosing SimpleSITS Mode\n");
        }
        else if (strcmp(controller[0].Command("sits_mode"), "classical") == 0)
        {
            info.sits_mode = CLASSICAL_SITS_MODE;
            printf("	Choosing ClassicalSITS Mode\n");
        }
        else
        {
            printf("Error: Please Choose a Correct SITS Mode!\n");
            getchar();
            return 0;
        }
    }
    else
    {
        printf("Error: Please Choose a SITS Mode!\n");
        getchar();
        return 0;
    }


    if (controller[0].Command_Exist("sits_atom_numbers"))
    {
        info.protein_atom_numbers = atoi(controller[0].Command("sits_atom_numbers"));
        printf("	sits_atom_numbers is %d\n", info.protein_atom_numbers);
    }
    else
    {
        printf("Error: Please Give a sits_atom_numbers!\n");
        getchar();
        return 0;
    }
    info.atom_numbers = atom_numbers;

    //初始化一些数组
    info.max_neighbor_numbers = 800;
    if (controller[0].Command_Exist("max_neighbor_numbers"))
        atoi(controller[0].Command("max_neighbor_numbers"));
    Cuda_Malloc_Safely((void**)&d_atom_energy, sizeof(float) * info.atom_numbers);
    Cuda_Malloc_Safely((void**)&d_protein_water_atom_energy, sizeof(float) * info.atom_numbers);
    Cuda_Malloc_Safely((void**)&d_total_pp_atom_energy, sizeof(float) * 1);
    Cuda_Malloc_Safely((void**)&d_total_ww_atom_energy, sizeof(float) * 1);
    Cuda_Malloc_Safely((void**)&d_total_protein_water_atom_energy, sizeof(float) * 1);
    Cuda_Malloc_Safely((void**)&protein_water_frc, sizeof(VECTOR) * info.atom_numbers);
    Reset_List<<<ceilf((float)3. * info.atom_numbers / 128), 128>>>(3 * info.atom_numbers,
                                                                    (float*)protein_water_frc, 0.);

    //记录分能量的文件
    if (controller[0].Command_Exist("sits_energy_record"))
    {
        Open_File_Safely(&sits_enerd_log, controller[0].Command("sits_energy_record"), "w");
    }
    else
    {
        Open_File_Safely(&sits_enerd_log, "SITS_Energy_Record.txt", "w");
        return 0;
    }

    // simple SITS模式的额外初始化
    //暂时不区分模式
    if (true || (info.sits_mode & 0x0000000F) == SIMPLE_SITS_MODE)
    {
        //初始化随机种子，用于fc随机运动
        srand(simple_info.random_seed);
        //初始化fc_pdf
        if (controller[0].Command_Exist("sits_fcball_pdf_grid_numbers"))
        {
            simple_info.grid_numbers = atoi(controller[0].Command("sits_fcball_pdf_grid_numbers"));
        }
        Malloc_Safely((void**)&simple_info.fc_pdf, sizeof(float) * simple_info.grid_numbers);
        for (int i = 0; i < simple_info.grid_numbers; i = i + 1)
        {
            simple_info.fc_pdf[i] = 0.01; //默认概率都为相同的非0值
        }
        if (controller[0].Command_Exist("sits_fcball_pdf"))
        {
            FILE* flin = NULL; //打开读入的临时文件指针
            Open_File_Safely(&flin, controller[0].Command("sits_fcball_pdf"), "r");
            for (int i = 0; i < simple_info.grid_numbers; i = i + 1)
            {
                fscanf(flin, "%f\n", &simple_info.fc_pdf[i]);
            }
            fclose(flin);
        }

        //如果采用固定的fc_ball
        if (controller[0].Command_Exist("sits_constant_fcball"))
        {
            simple_info.is_constant_fc_ball = 1;
            sscanf(controller[0].Command("sits_constant_fcball"), "%f", &simple_info.constant_fc_ball);
        }

        //随机游走的fc上下限和步长
        if (controller[0].Command_Exist("sits_fcball_max"))
        {
            sscanf(controller[0].Command("sits_fcball_max"), "%f", &simple_info.fc_max);
        }
        if (controller[0].Command_Exist("sits_fcball_min"))
        {
            sscanf(controller[0].Command("sits_fcball_min"), "%f", &simple_info.fc_min);
        }
        if (controller[0].Command_Exist("sits_fcball_move_length"))
        {
            sscanf(controller[0].Command("sits_fcball_move_length"), "%f", &simple_info.move_length);
        }

        //随机种子
        if (controller[0].Command_Exist("sits_fcball_random_seed"))
        {
            sscanf(controller[0].Command("sits_fcball_random_seed"), "%f", &simple_info.random_seed);
        }
    }

    printf("End (Selective Integrated Tempering Sampling)\n\n");
    return 1;
}

void SITS::Clear_SITS_Energy()
{
    Reset_List<<<ceilf((float)info.atom_numbers / 32), 32>>>(info.atom_numbers, d_atom_energy,
                                                             0.); //清空用于记录AA作用和BB作用的列表
    Reset_List<<<ceilf((float)info.atom_numbers / 32), 32>>>(
            info.protein_atom_numbers, d_protein_water_atom_energy, 0.); //清空用于记录AB相互作用的列表
}

void SITS::Calculate_Total_SITS_Energy(int is_fprintf)
{
    Sum_Of_List<<<1, 1024>>>(0, info.protein_atom_numbers, d_atom_energy, d_total_pp_atom_energy);
    Sum_Of_List<<<1, 1024>>>(info.protein_atom_numbers, info.atom_numbers, d_atom_energy,
                             d_total_ww_atom_energy);
    Sum_Of_List<<<1, 1024>>>(0, info.protein_atom_numbers, d_protein_water_atom_energy,
                             d_total_protein_water_atom_energy);


    if (is_fprintf)
    {
        float lin1, lin2, lin3;
        cudaMemcpy(&lin1, d_total_pp_atom_energy, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lin2, d_total_ww_atom_energy, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lin3, d_total_protein_water_atom_energy, sizeof(float), cudaMemcpyDeviceToHost);
        printf("SITS: ______AA______ ______BB______ ______AB______ ___Total______ fc_ball\n", lin1,
               lin2, lin3, lin1 + lin2 + lin3, simple_info.fc_ball);
        printf("      %14.4f %14.4f %14.4f %14.4f %7.4f\n", lin1, lin2, lin3, lin1 + lin2 + lin3,
               simple_info.fc_ball);
        fprintf(sits_enerd_log, "%f %f %f %f %f\n", lin1, lin2, lin3, lin1 + lin2 + lin3,
                simple_info.fc_ball);
    }
}

__global__ void SITS_For_Enhanced_Force_Protein(const int     protein_numbers,
                                                VECTOR*       md_frc,
                                                const VECTOR* pw_frc,
                                                const float   fc_ball,
                                                const float   factor)
{
    for (int i = threadIdx.x; i < protein_numbers; i = i + blockDim.x)
    {
        md_frc[i].x = fc_ball * (md_frc[i].x) + factor * pw_frc[i].x;
        md_frc[i].y = fc_ball * (md_frc[i].y) + factor * pw_frc[i].y;
        md_frc[i].z = fc_ball * (md_frc[i].z) + factor * pw_frc[i].z;
    }
}

__global__ void SITS_For_Enhanced_Force_Water(const int     protein_numbers,
                                              const int     atom_numbers,
                                              VECTOR*       md_frc,
                                              const VECTOR* pw_frc,
                                              const float   factor)
{
    for (int i = threadIdx.x + protein_numbers; i < atom_numbers; i = i + blockDim.x)
    {
        md_frc[i].x = md_frc[i].x + factor * pw_frc[i].x;
        md_frc[i].y = md_frc[i].y + factor * pw_frc[i].y;
        md_frc[i].z = md_frc[i].z + factor * pw_frc[i].z;
    }
}

void SITS::SITS_Enhanced_Force(VECTOR* frc)
{
    if ((info.sits_mode & 0x0000000F) == SIMPLE_SITS_MODE)
    {
        //确定fc_ball数值（通过在给定势场中随机游走刷新，以保证fc_ball的制定分布）
        if (!simple_info.is_constant_fc_ball)
        {
            simple_info.fc_ball_random_walk();
        }
        else
        {
            simple_info.fc_ball = simple_info.constant_fc_ball;
        }
        SITS_For_Enhanced_Force_Protein<<<1, 128>>>(info.protein_atom_numbers, frc,
                                                    protein_water_frc, simple_info.fc_ball,
                                                    0.5 * (simple_info.fc_ball + 1.));
        SITS_For_Enhanced_Force_Water<<<1, 128>>>(info.protein_atom_numbers, info.atom_numbers, frc,
                                                  protein_water_frc, 0.5 * (simple_info.fc_ball + 1.));
    }
    else if ((info.sits_mode & 0x0000000F) == CLASSICAL_SITS_MODE)
    {
        //暂时设为跟随simple SITS，检查正确性
        if (!simple_info.is_constant_fc_ball)
        {
            simple_info.fc_ball_random_walk();
        }
        else
        {
            simple_info.fc_ball = simple_info.constant_fc_ball;
        }
        SITS_For_Enhanced_Force_Protein<<<1, 128>>>(info.protein_atom_numbers, frc,
                                                    protein_water_frc, simple_info.fc_ball,
                                                    0.5 * (simple_info.fc_ball + 1.));
        SITS_For_Enhanced_Force_Water<<<1, 128>>>(info.protein_atom_numbers, info.atom_numbers, frc,
                                                  protein_water_frc, 0.5 * (simple_info.fc_ball + 1.));
    }
    Reset_List<<<ceilf((float)3. * info.atom_numbers / 128), 128>>>(3 * info.atom_numbers,
                                                                    (float*)protein_water_frc, 0.);
}

float SITS::FC_BALL_INFORMATION::get_fc_probability(float pos)
{

    // float lin = (float)grid_numbers - (pos - fc_min)*grid_numbers / (fc_max - fc_min);//如果fcball推荐分布的列表是反向放置的，那么就按这条代码计算
    float lin = (float)(pos - fc_min) * grid_numbers / (fc_max - fc_min);

    int a = (int)lin;
    if (a >= 0 && a < grid_numbers)
    {
        return fc_pdf[a];
    }
    else
    {
        return 0.;
    }
}

void SITS::FC_BALL_INFORMATION::fc_ball_random_walk()
{
    float fc_test = fc_ball + move_length * ((float)rand() / RAND_MAX - 0.5); //尝试更新的fc_ball数值
    float p2 = get_fc_probability(fc_test);                                   //新位置的势能
    float p  = p2 / current_fc_probability; //新旧位置的势能比,（向势场高处走，最后的分布就正比于势能曲线）
    if (p > ((float)rand() / RAND_MAX))
    {
        fc_ball                = fc_test;
        current_fc_probability = p2;
        if (fc_ball >= fc_max)
        {
            fc_ball = fc_max;
        }
        else if (fc_ball <= fc_min)
        {
            fc_ball = fc_min;
        }
    }
}

void SITS::Clear_SITS()
{
    if (sits_enerd_log != NULL)
    {
        fclose(sits_enerd_log);
    }

    if ((info.sits_mode & 0x0000000F) == SIMPLE_SITS_MODE)
    {
        if (simple_info.fc_pdf != NULL)
        {
            free(simple_info.fc_pdf);
        }
    }

    if (d_atom_energy != NULL)
    {
        cudaFree(d_atom_energy);
    }
    if (d_total_pp_atom_energy != NULL)
    {
        cudaFree(d_total_pp_atom_energy);
    }
    if (d_total_ww_atom_energy != NULL)
    {
        cudaFree(d_total_ww_atom_energy);
    }
    if (d_protein_water_atom_energy != NULL)
    {
        cudaFree(d_protein_water_atom_energy);
    }
    if (d_total_protein_water_atom_energy != NULL)
    {
        cudaFree(d_total_protein_water_atom_energy);
    }

    if (protein_water_frc != NULL)
    {
        cudaFree(protein_water_frc);
    }

    NEIGHBOR_LIST* h_nl = NULL; //临时的近邻表
    Malloc_Safely((void**)&h_nl, sizeof(NEIGHBOR_LIST) * info.atom_numbers);
    if (d_nl_ppww != NULL)
    {
        cudaMemcpy(h_nl, d_nl_ppww, sizeof(NEIGHBOR_LIST) * info.atom_numbers, cudaMemcpyDeviceToHost);
        for (int i = 0; i < info.atom_numbers; i = i + 1)
        {
            cudaFree(h_nl[i].atom_serial);
        }
        cudaFree(d_nl_ppww);
    }
    if (d_nl_pwwp != NULL)
    {
        cudaMemcpy(h_nl, d_nl_ppww, sizeof(NEIGHBOR_LIST) * info.atom_numbers, cudaMemcpyDeviceToHost);
        for (int i = 0; i < info.atom_numbers; i = i + 1)
        {
            cudaFree(h_nl[i].atom_serial);
        }
        cudaFree(d_nl_pwwp);
    }
    free(h_nl);
}