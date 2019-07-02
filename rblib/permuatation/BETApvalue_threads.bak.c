/*
Prog Description:
		
Author: Rong Zheng-Qin
Email: rongzhengqin@genomics.org.cn  or  zju3351689@gmail.com
2011-12-22 13:12:00
*/

#include<stdio.h>
#include<stdlib.h>
#include"sr_msort.h"
#include<string.h>
#include<math.h>

double ** malloc_data(dimen_t demensionx, dimen_t demensiony);
void err_open(void);
double ** zeros_init(dimen_t demensionx, dimen_t demensiony);

int get_pvalue(double *pvalue, dimen_t num, dimen_t ntimes);

int cmpmatrix(double **pvalue, double **beta, double **per, dimen_t demensionx, dimen_t demensiony);
int cmpvector(double *pvalue, double *beta, double *per, dimen_t demensionx);
int recover_pvalue(double *pvalue, dimen_t * index, dimen_t num);
void fabs_map(double *vector, dimen_t num);

void print_data(double **p, dimen_t demensionx, dimen_t demensiony);

int main(int argc, char *argv[])
{
	FILE *fp_beta;
	FILE *fp_per;
	//double **beta;
	double **per_adv;
	//double **per_hyb;

	double **pvalue_adv;
	//double **pvalue_hyb;

	dimen_t demensionx_adv, demensiony;
	//dimen_t demensionx_hyb;
	//int num_thread;

	demensionx_adv = 100;
	//demensionx_hyb = 151438;

	demensiony = 1;
	dimen_t ntimes = 1000;
	dimen_t i,j,k;

	msortind_t **beta_ind_adv;
	//msortind_t **beta_ind_hyb;
	
	//beta = malloc_data(demensionx, demensiony);
	per_adv = malloc_data(demensionx_adv, demensiony);
	//per_hyb = malloc_data(demensionx_hyb, demensiony);

	pvalue_adv = zeros_init(demensionx_adv, demensiony);
	//pvalue_hyb = zeros_init(demensionx_hyb, demensiony);

	beta_ind_adv = (msortind_t **) malloc(sizeof(msortind_t *) * demensiony);
	//beta_ind_hyb = (msortind_t **) malloc(sizeof(msortind_t *) * demensiony);

	for(i=0;i<demensiony;i++)
	{//malloc the beta matrix for each Y
		beta_ind_adv[i] = (msortind_t *) malloc(sizeof(msortind_t));
		beta_ind_adv[i]->num = demensionx_adv;
		beta_ind_adv[i]->vector = (double *) malloc(sizeof(double) * beta_ind_adv[i]->num);
		msortind_init(beta_ind_adv[i], beta_ind_adv[i]->num);

		//beta_ind_hyb[i] = (msortind_t *) malloc(sizeof(msortind_t));
		//beta_ind_hyb[i]->num = demensionx_hyb;
		//beta_ind_hyb[i]->vector = (double *) malloc(sizeof(double) * beta_ind_hyb[i]->num);
		//msortind_init(beta_ind_hyb[i], beta_ind_hyb[i]->num);
	}

	if ((fp_beta=fopen(argv[1],"rb"))==NULL)
		err_open();
	if ((fp_per=fopen(argv[2],"rb"))==NULL)
		err_open();

	while(!feof(fp_beta))
	{//read the beta matrix from file
		for(i=0;i<demensiony;i++)
		{
			fread(beta_ind_adv[i]->vector,sizeof(double),demensionx_adv,fp_beta);
			fabs_map(beta_ind_adv[i]->vector, beta_ind_adv[i]->num);
			//fread(beta_ind_hyb[i]->vector,sizeof(double),demensionx_hyb,fp_beta);
			//fabs_map(beta_ind_hyb[i]->vector, beta_ind_hyb[i]->num);
			//print_sort(beta_ind_adv[i]->vector,beta_ind_adv[i]->num);
			//print_sort(beta_ind_hyb[i]->vector,beta_ind_hyb[i]->num);
			mergesort_ind(beta_ind_adv[i]);
			//mergesort_ind(beta_ind_hyb[i]);
			//print_sort_ind(beta_ind_adv[i]);
			//print_sort_ind(beta_ind_hyb[i]);
		}
		break;
	}
	fclose(fp_beta);

	//while(!feof(fp_per))
	for(k=0;k<ntimes;k++)
	{
		for(i=0;i<demensiony;i++)
		{
			fread(per_adv[i],sizeof(double),demensionx_adv,fp_per);
			//fread(per_hyb[i],sizeof(double),demensionx_hyb,fp_per);
			fabs_map(per_adv[i], demensionx_adv);
			//fabs_map(per_hyb[i], demensionx_hyb);
			mergeSort(per_adv[i],demensionx_adv);
			//mergesort(per_hyb[i],demensionx_hyb);
			cmpvector(pvalue_adv[i],beta_ind_adv[i]->vector,per_adv[i],demensionx_adv);
			//cmpvector(pvalue_hyb[i],beta_ind_hyb[i]->vector,per_hyb[i],demensionx_hyb);
		}
		//print_data(per,demensionx,demensiony);
		//cmpmatrix(pvalue, beta, per, demensionx, demensiony);
		//break;
	}
	fclose(fp_per);
	
	//print_data(pvalue_adv, demensionx_adv, demensiony);
	//print_data(pvalue_hyb, demensionx_hyb, demensiony);

	for(i=0;i<demensiony;i++)
	{
		recover_pvalue(pvalue_adv[i], beta_ind_adv[i]->index, beta_ind_adv[i]->num);
		//recover_pvalue(pvalue_hyb[i], beta_ind_hyb[i]->index, beta_ind_hyb[i]->num);
	}

	//print_data(pvalue_adv, demensionx_adv, demensiony);
	//print_data(pvalue_hyb, demensionx_hyb, demensiony);
	for(i=0;i<demensiony;i++)
	{
		get_pvalue(pvalue_adv[i], demensionx_adv, ntimes);
		//get_pvalue(pvalue_hyb[i], demensionx_hyb, ntimes);
	}

	//output
	for(i=0;i<demensionx_adv;i++)
	{
		for(j=0;j<demensiony;j++)
			printf("%.14f\t", pvalue_adv[j][i]);
		printf("\n");
	}
	/*for(i=0;i<demensionx_hyb;i++)
	{
		for(j=0;j<demensiony;j++)
			printf("%.14f\t", pvalue_hyb[j][i]);
		printf("\n");
	}*/

	for(i=0;i<demensiony;i++)
	{
		free(beta_ind_adv[i]->vector);
		free(beta_ind_adv[i]->index);
		free(beta_ind_adv[i]);
		free(per_adv[i]);

		/*free(beta_ind_hyb[i]->vector);
		free(beta_ind_hyb[i]->index);
		free(beta_ind_hyb[i]);
		free(per_hyb[i]);*/
	}
	free(beta_ind_adv);
	//free(beta_ind_hyb);
	free(per_adv);
	//free(per_hyb);
	return 0;
}

void fabs_map(double *vector, dimen_t num)
{
	dimen_t i;
	for(i=0;i<num;i++)
		vector[i] = fabs(vector[i]);
}

int get_pvalue(double *pvalue, dimen_t num, dimen_t ntimes)
{
	dimen_t i;
	for(i=0;i<num;i++)
		pvalue[i] = pvalue[i]/num/ntimes;
	return 0;
}

void print_data(double **p, dimen_t demensionx, dimen_t demensiony)
{
	dimen_t i,j;
	for(i=0;i<demensiony;i++)
	{
		for(j=0;j<demensionx;j++)
			printf("%0.3f,",p[i][j]);
		printf("\n");
	}
}

void err_open(void)
{
	printf("error to open the file,please check!!");
	abort();
}

double ** malloc_data(dimen_t demensionx, dimen_t demensiony)
{
	double **p;
	dimen_t i;
	p = (double **) malloc (sizeof(double *)* demensiony);
	for(i=0;i<demensiony;i++)
		p[i] = (double *) malloc (sizeof(double)* demensionx);
	return p;
}

double ** zeros_init(dimen_t demensionx,dimen_t demensiony)
{
	double **p;
	dimen_t i;
	p = (double **) malloc (sizeof(double *)* demensiony);
	for(i=0;i<demensiony;i++)
		p[i] = (double *) calloc (demensionx,sizeof(double));
	return p;
}

int cmpmatrix(double **pvalue, double **beta, double **per, dimen_t demensionx, dimen_t demensiony)
{
	dimen_t i,j,k;
	for(k=0;k<demensiony;k++)
	{
		for(i=0;i<demensionx;i++)
		{
			for(j=0;j<demensionx;j++)
				pvalue[k][i] += beta[k][i]>per[k][j] ? 1:0;
		}
	}
	return 1;
}

int cmpvector(double *pvalue, double *beta, double *per, dimen_t demensionx)
{
	double value = 0.0;
	dimen_t i;
	double * temppos;
	//dimen_t count_beta=0;
	dimen_t count_per = 0;
	double total;

	total = (double) demensionx;
	temppos = per;
	for(i=0;i<demensionx;i++)
	{
		for(;;)
		{
			if((count_per < demensionx) && (beta[i] > *temppos))
			{
				value += 1.0;
				temppos++;
				count_per++;
				//printf("%lu\n",count_per);
				//fflush(stdout);
			}
			else
			{
				pvalue[i] +=(total - value);
				break;
			}
		}
	}
	return 0;
}


int recover_pvalue(double *pvalue, dimen_t * index, dimen_t num)
{
	double *pvaluetemp;
	dimen_t i;

	pvaluetemp = (double *) malloc (sizeof(double) * num);
	memcpy(pvaluetemp, pvalue, sizeof(double)*num);

	for(i=0;i<num;i++)
		pvalue[index[i]] = pvaluetemp[i];
	free(pvaluetemp);
	return 0;
}
