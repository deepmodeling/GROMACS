#include "SITS.cuh"

static __global__ void SITS_For_Enhanced_Force_Protein(const int     protein_numbers,
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
static __global__ void SITS_For_Enhanced_Force_Water(const int     protein_numbers,
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

static __device__ __host__ float log_add_log(float a, float b)
{
    return fmaxf(a, b) + logf(1.0 + expf(-fabsf(a - b)));
}

static __global__ void SITS_Record_Ene(float*       ene_record,
                                       const float* pw_ene,
                                       const float* pp_ene,
                                       const float  pe_a,
                                       const float  pe_b,
                                       const float  pwwp_factor)
{
    float temp = *pw_ene * pwwp_factor + *pp_ene;
    temp       = pe_a * temp + pe_b;

    *ene_record = temp;
    // printf("DEBUG ene_record: %f\n", ene_record[0]);
}

static __global__ void
SITS_Update_gf(const int kn, float* gf, const float* ene_record, const float* log_nk, const float* beta_k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
    {
        gf[i] = -beta_k[i] * ene_record[0] + log_nk[i];
        // printf("DEBUG gf: %d %f\n", i, gf[i]);
    }
}

static __global__ void SITS_Update_gfsum(const int kn, float* gfsum, const float* gf)
{
    if (threadIdx.x == 0)
    {
        gfsum[0] = -FLT_MAX;
    }
    for (int i = 0; i < kn; i = i + 1)
    {
        gfsum[0] = log_add_log(gfsum[0], gf[i]);
        // printf("DEBUG gfsum: %d %f %f\n", i, gfsum[0], gf[i]);
    }
}

static __global__ void
SITS_Update_log_pk(const int kn, float* log_pk, const float* gf, const float* gfsum, const int reset)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
    {
        float gfi = gf[i];
        log_pk[i] = ((float)reset) * gfi + ((float)(1 - reset)) * log_add_log(log_pk[i], gfi - gfsum[0]);
        // printf("DEBUG log_pk: %d %f %f\n", i, log_pk[i], gfsum[0]);
    }
}


static __global__ void SITS_Update_log_mk_inv(const int    kn,
                                              float*       log_weight,
                                              float*       log_mk_inv,
                                              float*       log_norm_old,
                                              float*       log_norm,
                                              const float* log_pk,
                                              const float* log_nk)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn - 1)
    {
        log_weight[i] = (log_pk[i] + log_pk[i + 1]) * 0.5;
        // printf("DEBUG log_weight: %d %f %f\n", i, log_pk[i], log_pk[i + 1]);
        log_mk_inv[i]   = log_nk[i] - log_nk[i + 1];
        log_norm_old[i] = log_norm[i];
        log_norm[i]     = log_add_log(log_norm[i], log_weight[i]);
        log_mk_inv[i] =
                log_add_log(log_mk_inv[i] + log_norm_old[i] - log_norm[i],
                            log_pk[i + 1] - log_pk[i] + log_mk_inv[i] + log_weight[i] - log_norm[i]);
        // printf("DEBUG log_norm: %d %f %f\n", i, log_norm[i], log_weight[i]);
    }
}

static __global__ void SITS_Update_log_nk_inv(const int kn, float* log_nk_inv, const float* log_mk_inv)
{
    for (int i = 0; i < kn - 1; i++)
    {
        log_nk_inv[i + 1] = log_nk_inv[i] + log_mk_inv[i];
    }
}

static __global__ void SITS_Update_nk(const int kn, float* log_nk, float* nk, const float* log_nk_inv)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kn)
    {
        log_nk[i] = -log_nk_inv[i];
        nk[i]     = exp(log_nk[i]);
    }
}

__global__ void SITS_For_Enhanced_Force_Calculate_nkExpBetakU_1(const int    k_numbers,
                                                                const float* beta_k,
                                                                const float* nk,
                                                                float*       nkexpbetaku,
                                                                const float  ene)
{
    float lin = beta_k[k_numbers - 1];
    for (int i = threadIdx.x; i < k_numbers; i = i + blockDim.x)
    {
        nkexpbetaku[i] = nk[i] * expf(-(beta_k[i] - lin) * ene);
        // printf("%f %f\n", beta_k[i], nkexpbetaku[i]);
    }
}

__global__ void SITS_For_Enhanced_Force_Calculate_nkExpBetakU_2(const int    k_numbers,
                                                                const float* beta_k,
                                                                const float* nk,
                                                                float*       nkexpbetaku,
                                                                const float  ene)
{
    float lin = beta_k[0];
    for (int i = threadIdx.x; i < k_numbers; i = i + blockDim.x)
    {
        nkexpbetaku[i] = nk[i] * expf(-(beta_k[i] - lin) * ene);
        // printf("%f %f\n", beta_k[i], nkexpbetaku[i]);
    }
}

__global__ void SITS_For_Enhanced_Force_Sum_Of_Above(const int    k_numbers,
                                                     const float* nkexpbetaku,
                                                     const float* beta_k,
                                                     float*       sum_of_above)
{
    if (threadIdx.x == 0)
    {
        sum_of_above[0] = 0.;
    }
    __syncthreads();
    float lin = 0.;
    for (int i = threadIdx.x; i < k_numbers; i = i + blockDim.x)
    {
        lin = lin + beta_k[i] * nkexpbetaku[i];
    }
    atomicAdd(sum_of_above, lin);
}

__global__ void SITS_For_Enhanced_Force_Sum_Of_nkExpBetakU(const int    k_numbers,
                                                           const float* nkexpbetaku,
                                                           float*       sum_of_below)
{
    if (threadIdx.x == 0)
    {
        sum_of_below[0] = 0.;
    }
    __syncthreads();
    float lin = 0.;
    for (int i = threadIdx.x; i < k_numbers; i = i + blockDim.x)
    {
        lin = lin + nkexpbetaku[i];
        // printf("%f\n", nkexpbetaku[i]);
    }
    atomicAdd(sum_of_below, lin);
}

__global__ void SITS_Classical_Enhanced_Force(const int     atom_numbers,
                                              const int     protein_atom_numbers,
                                              const float   pwwp_factor,
                                              VECTOR*       md_frc,
                                              const VECTOR* pw_frc,
                                              const float*  pp_ene,
                                              const float*  pw_ene,
                                              const int     k_numbers,
                                              float*        nkexpbetaku,
                                              const float*  beta_k,
                                              const float*  n_k,
                                              float*        sum_a,
                                              float*        sum_b,
                                              float*        factor,
                                              const float   beta_0,
                                              const float   pe_a,
                                              const float   pe_b,
                                              const float   fb_bias)
{
    float ene = pp_ene[0] + pwwp_factor * pw_ene[0];
    ene       = pe_a * ene + pe_b;
    if (ene > 0)
    {
        SITS_For_Enhanced_Force_Calculate_nkExpBetakU_1<<<1, 64>>>(k_numbers, beta_k, n_k,
                                                                   nkexpbetaku, ene);
    }
    else
    {
        SITS_For_Enhanced_Force_Calculate_nkExpBetakU_2<<<1, 64>>>(k_numbers, beta_k, n_k,
                                                                   nkexpbetaku, ene);
    }

    SITS_For_Enhanced_Force_Sum_Of_nkExpBetakU<<<1, 128>>>(k_numbers, nkexpbetaku, sum_b);

    SITS_For_Enhanced_Force_Sum_Of_Above<<<1, 128>>>(k_numbers, nkexpbetaku, beta_k, sum_a);


    factor[0] = sum_a[0] / sum_b[0] / beta_0 + fb_bias;
    //这段代码是避免fc_ball变化太大而造成体系崩溃的
    if (!isinf(factor[0]) && !isnan(factor[0]) && (factor[0] > 0.5 * factor[1])
        && (factor[0] < 2 * factor[1]))
    {
        factor[1] = factor[0];
    }
    else
    {
        factor[0] = factor[1];
    }
    float fc = factor[0];

    //	printf("factor %e sum0 %e %e ene %f lfactor %e\n", fc, sum_a[0], sum_b[0], ene, factor[1]);
    __syncthreads();


    // line
    // fc = (ene - 20.) / 80./2. + 0.2;
    SITS_For_Enhanced_Force_Protein<<<1, 128>>>(protein_atom_numbers, md_frc, pw_frc, fc,
                                                pwwp_factor * fc + 1.0 - pwwp_factor);
    SITS_For_Enhanced_Force_Water<<<1, 128>>>(protein_atom_numbers, atom_numbers, md_frc, pw_frc,
                                              pwwp_factor * fc + 1.0 - pwwp_factor);
}

void SITS::SITS_Classical_Update_Info(int steps)
{
    if (!classical_info.constant_nk && steps % classical_info.record_interval == 0)
    {
        SITS_Record_Ene<<<1, 1>>>(classical_info.ene_recorded, d_total_protein_water_atom_energy,
                                  d_total_pp_atom_energy, classical_info.energy_multiple,
                                  classical_info.energy_shift, info.pwwp_enhance_factor);

        SITS_Update_gf<<<ceilf((float)classical_info.k_numbers / 32.), 32>>>(
                classical_info.k_numbers, classical_info.gf, classical_info.ene_recorded,
                classical_info.log_nk, classical_info.beta_k);

        SITS_Update_gfsum<<<1, 1>>>(classical_info.k_numbers, classical_info.gfsum, classical_info.gf);


        SITS_Update_log_pk<<<ceilf((float)classical_info.k_numbers / 32.), 32>>>(
                classical_info.k_numbers, classical_info.log_pk, classical_info.gf,
                classical_info.gfsum, classical_info.reset);

        classical_info.reset = 0;
        classical_info.record_count++;

        if (classical_info.record_count % classical_info.update_interval == 0)
        {
            SITS_Update_log_mk_inv<<<ceilf((float)classical_info.k_numbers / 32.), 32>>>(
                    classical_info.k_numbers, classical_info.log_weight, classical_info.log_mk_inv,
                    classical_info.log_norm_old, classical_info.log_norm, classical_info.log_pk,
                    classical_info.log_nk);

            SITS_Update_log_nk_inv<<<1, 1>>>(classical_info.k_numbers, classical_info.log_nk_inv,
                                             classical_info.log_mk_inv);

            SITS_Update_nk<<<ceilf((float)classical_info.k_numbers / 32.), 32>>>(
                    classical_info.k_numbers, classical_info.log_nk, classical_info.nk,
                    classical_info.log_nk_inv);


            classical_info.record_count = 0;
            classical_info.reset        = 1;

            if (!classical_info.constant_nk)
            {
                cudaMemcpy(classical_info.log_nk_recorded_cpu, classical_info.nk,
                           sizeof(float) * classical_info.k_numbers, cudaMemcpyDeviceToHost);
                fwrite(classical_info.log_nk_recorded_cpu, sizeof(float), classical_info.k_numbers,
                       classical_info.nk_traj_file);
                cudaMemcpy(classical_info.log_norm_recorded_cpu, classical_info.log_norm,
                           sizeof(float) * classical_info.k_numbers, cudaMemcpyDeviceToHost);
                fwrite(classical_info.log_norm_recorded_cpu, sizeof(float),
                       classical_info.k_numbers, classical_info.norm_traj_file);
            }
        }
    }
}

void SITS::clear_sits_energy()
{
    Reset_List<<<ceilf((float)info.atom_numbers / 32), 32>>>(
            info.protein_atom_numbers, d_protein_water_atom_energy, 0.); //清空用于记录AB相互作用的列表
}
void SITS::Calculate_Total_SITS_Energy(float* d_atom_energy)
{
    Sum_Of_List<<<1, 1024>>>(0, info.protein_atom_numbers, d_atom_energy, d_total_pp_atom_energy);
    Sum_Of_List<<<1, 1024>>>(info.protein_atom_numbers, info.atom_numbers, d_atom_energy,
                             d_total_ww_atom_energy);
    Sum_Of_List<<<1, 1024>>>(0, info.protein_atom_numbers, d_protein_water_atom_energy,
                             d_total_protein_water_atom_energy);
}

void SITS::Print()
{
    if (info.sits_mode == CLASSICAL_SITS_MODE)
    {
        cudaMemcpy(&info.fc_ball, classical_info.factor, sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(&h_total_pp_atom_energy, d_total_pp_atom_energy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_total_ww_atom_energy, d_total_ww_atom_energy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_total_protein_water_atom_energy, d_total_protein_water_atom_energy, sizeof(float),
               cudaMemcpyDeviceToHost);
    printf("SITS: ______AA______ ______BB______ ______AB______ fc_ball\n");
    printf("      %14.4f %14.4f %14.4f %7.4f\n", h_total_pp_atom_energy, h_total_ww_atom_energy,
           h_total_protein_water_atom_energy, info.fc_ball);
    fprintf(sits_ene_record_out, "%f %f %f %f\n", h_total_pp_atom_energy, h_total_ww_atom_energy,
            h_total_protein_water_atom_energy, info.fc_ball);
}

void SITS::Prepare_For_Calculate_Force(int* need_atom_energy, int isPrintStep)
{
    Reset_List<<<ceilf((float)3. * info.atom_numbers / 128), 128>>>(3 * info.atom_numbers,
                                                                    (float*)protein_water_frc, 0.);
    if (info.sits_mode <= 1 && isPrintStep)
        *need_atom_energy += 1;
    if (*need_atom_energy > 0)
        clear_sits_energy();
}

void SITS::sits_enhance_force(int steps, VECTOR* frc)
{
    if (info.sits_mode == SIMPLE_SITS_MODE)
    {
        //确定fc_ball数值（通过在给定势场中随机游走刷新，以保证fc_ball的制定分布）
        if (!simple_info.is_constant_fc_ball)
        {
            fc_ball_random_walk();
        }
        else
        {
            info.fc_ball = simple_info.constant_fc_ball;
        }
        SITS_For_Enhanced_Force_Protein<<<1, 128>>>(
                info.protein_atom_numbers, frc, protein_water_frc, info.fc_ball,
                info.pwwp_enhance_factor * info.fc_ball + 1.0 - info.pwwp_enhance_factor);
        SITS_For_Enhanced_Force_Water<<<1, 128>>>(
                info.protein_atom_numbers, info.atom_numbers, frc, protein_water_frc,
                info.pwwp_enhance_factor * info.fc_ball + 1.0 - info.pwwp_enhance_factor);
    }
    else if (info.sits_mode == CLASSICAL_SITS_MODE)
    {
        SITS_Classical_Update_Info(steps);
        SITS_Classical_Enhanced_Force<<<1, 1>>>(
                info.atom_numbers, info.protein_atom_numbers, info.pwwp_enhance_factor, frc,
                protein_water_frc, d_total_pp_atom_energy, d_total_protein_water_atom_energy,
                classical_info.k_numbers, classical_info.nkExpBetakU, classical_info.beta_k,
                classical_info.nk, classical_info.sum_a, classical_info.sum_b,
                classical_info.factor, classical_info.beta0, classical_info.energy_multiple,
                classical_info.energy_shift, classical_info.fb_shift);
    }
    else
    {
        SITS_For_Enhanced_Force_Protein<<<1, 128>>>(
                info.protein_atom_numbers, frc, protein_water_frc, info.fc_ball,
                info.pwwp_enhance_factor * info.fc_ball + 1.0 - info.pwwp_enhance_factor);
        SITS_For_Enhanced_Force_Water<<<1, 128>>>(
                info.protein_atom_numbers, info.atom_numbers, frc, protein_water_frc,
                info.pwwp_enhance_factor * info.fc_ball + 1.0 - info.pwwp_enhance_factor);
    }
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

void SITS::fc_ball_random_walk()
{
    float fc_test = info.fc_ball + simple_info.move_length * ((float)rand() / RAND_MAX - 0.5); //尝试更新的fc_ball数值
    float p2      = simple_info.get_fc_probability(fc_test); //新位置的势能
    float p       = p2 / simple_info.current_fc_probability; //新旧位置的势能比,（向势场高处走，最后的分布就正比于势能曲线）
    if (p > ((float)rand() / RAND_MAX))
    {
        info.fc_ball                       = fc_test;
        simple_info.current_fc_probability = p2;
        if (info.fc_ball >= simple_info.fc_max)
        {
            info.fc_ball = simple_info.fc_max;
        }
        else if (info.fc_ball <= simple_info.fc_min)
        {
            info.fc_ball = simple_info.fc_min;
        }
    }
}

void SITS::CLASSICAL_SITS_INFORMATION::Export_Restart_Information_To_File()
{
    FILE* nk;
    Open_File_Safely(&nk, nk_rest_file, "w");
    cudaMemcpy(log_nk_recorded_cpu, nk, sizeof(float) * k_numbers, cudaMemcpyDeviceToHost);
    for (int i = 0; i < k_numbers; i++)
    {
        fprintf(nk, "%f\n", log_nk_recorded_cpu[i]);
    }
    fclose(nk);

    FILE* norm;
    Open_File_Safely(&norm, norm_rest_file, "w");
    cudaMemcpy(log_norm_recorded_cpu, log_norm, sizeof(float) * k_numbers, cudaMemcpyDeviceToHost);
    for (int i = 0; i < k_numbers; i++)
    {
        fprintf(norm, "%f\n", log_norm_recorded_cpu[i]);
    }
    fclose(norm);
}

void SITS::Clear_SITS()
{
    if (sits_ene_record_out != NULL)
    {
        fclose(sits_ene_record_out);
    }

    if ((info.sits_mode & 0x0000000F) == SIMPLE_SITS_MODE)
    {
        if (simple_info.fc_pdf != NULL)
        {
            free(simple_info.fc_pdf);
        }
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
