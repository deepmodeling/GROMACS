#ifndef SITS_COMMON_CUH
#define SITS_COMMON_CUH

//��ͷ�ļ��Ǹ�SITS������ʹ�õ���ͷ�ļ�����������ͨMD�����ݡ�
//ʹ�÷�����������ƪ����
//A selective integrated tempering method
//Self-adaptive enhanced sampling in the energy and trajectory spaces : Accelerated thermodynamics and kinetic calculations
//�������е�A,B����ԭ�ӽ�����������,ˮ��

struct SITS_INFORMATION
{
	int atom_numbers;//��ϵ������ԭ����Ŀ,����MD_INFORMATION�е�ֵ
	int protein_atom_numbers;//��ϵ�ĵ���ԭ����Ŀ,��Ϊ�趨
	int water_atom_numbers;//��ϵ��ˮԭ����Ŀ

	NEIGHBOR_LIST *protein_water_neighbor;//����ԭ����Χ��ˮ�ھӱ�����Ϊprotein_atom_numbers
	int *protein_water_numbers;//������Χˮ�ھӵ�ԭ�Ӹ���������Ϊprotein_atom_numbers
	//����-���ף�ˮ-ˮ���ڱ�ֱ������ͨMD�������Ľ��ڱ�

	VECTOR *protein_water_frc;//����-ˮԭ�Ӽ��frc������Ϊatom_numbers��ǰprotein_atom_numbers��Ϊ����ԭ���ܵ�ˮ��frc�������󲿷�Ϊˮ�ܵ�����ԭ�ӵ�frc����
	//����-���ף�ˮ-ˮ���õ�frcֱ������ͨMD��������frc

	float *protein_water_energy;//������ˮԭ�Ӽ����������������Ϊprotein_atom_numbers
	//���走��-ˮ֮�䲻���ڼ������ö�ֻ��LJ���ܺ�CF���ܣ���˵���-����������ˮ-ˮ�����ɰ���δ���ֵ����ֱ�Ӵ���һ���б���
	//���ڲ�����ĳ�������ڲ��ľ����������������˿���ֱ�Ӱ�bond angle��dihedral�����ӵ����������һ��ԭ������
	float *atom_energy;//����Ϊatom_numbers����������-����������ˮ-ˮ����

	//���Ⱦ�Ϊ1�����ڴ洢�ܵķֿ�������ָ��
	float *sum_of_water_water_energy;
	float *sum_of_protein_protein_energy;
	float *sum_of_protein_water_energy;


	int k_numbers;
	float *beta_k;
	float *nkExpBetakU;
	float *nk;
	float *sum_a;
	float *sum_b;
	float *factor;

	//����ʹ�ã�XYJ
	//���Ʊ���
	
	int record_count;
	int record_interval;
	int update_interval;
	int reset = 1;

	//�������
	//xyj��cpp������-ylj��F90������-���׶�Ӧ
	//ene_recorded - vshift - ene
	//gf - gf - log( n_k * exp(-beta_k * ene) )
	//gfsum - gfsum - log( Sum_(k=1)^N ( log( n_k * exp(-beta_k * ene) ) ) )
	//log_weight - rb - log of the weighting function
	//log_mk_inv - ratio - log(m_k^-1)
	//log_norm_old - normlold - W(j-1)
	//log_norm - norml - W(j)
	//log_pk - rbfb - log(p_k)
	//log_nk_inv - pratio - log(n_k^-1)
	//log_nk - fb - log(n_k)

	float *ene_recorded;
	float *gf;
	float *gfsum;
	float *log_weight;
	float *log_mk_inv;
	float *log_norm_old;
	float *log_norm;
	float *log_pk;
	float *log_nk_inv;
	float *log_nk;


};

//��¼ʱ���º���

__global__ void SITS_Record_Ene(float *ene_record, const float *pw_ene, const float *pp_ene, const float pe_a, const float pe_b);

__global__ void SITS_Update_gf(const int kn, float *gf,
	const float *ene_record, const float *log_nk, const float *beta_k);

__global__ void SITS_Update_gfsum(const int kn, float *gfsum, const float *gf);

__global__ void SITS_Update_log_pk(const int kn, float *log_pk,
	const float *gf, const float *gfsum, const int reset);

//����ʱ���º���

__global__ void SITS_Update_log_mk_inv(const int kn,
	float *log_weight, float *log_mk_inv, float *log_norm_old,
	float *log_norm, const float *log_pk, const float *log_nk);

__global__ void SITS_Update_log_nk_inv(const int kn,
	float *log_nk_inv, const float *log_mk_inv);

__global__ void SITS_Update_nk(const int kn,
	float *log_nk, float *nk, const float *log_nk_inv);



#endif //SITS_COMMON_CUH(SITS_common.cuh)