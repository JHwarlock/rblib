/*
 * Prog Description:
 *
 * Author: Rong Zheng-Qin
 * Email: rongzhengqin@genomics.org.cn  or  zju3351689@gmail.com
 * */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include"sr_msort.h"

static inline dimen_t _min_msort(dimen_t num1, dimen_t num2);
static int _merge(double * vector, double * temp, dimen_t part1start, dimen_t part1end, dimen_t part2start, dimen_t part2end);
static int _merge_ind(msortind_t *data_ind, msortind_t * tmpdata_ind, dimen_t part1start,dimen_t part1end,dimen_t part2start,dimen_t part2end);
static int _test_reverse_sorted(double *vector, dimen_t num);
static int _max_msort(double *vector, dimen_t num);
static int _max_msort_ind(msortind_t *data_ind);

//start to test sr_msort module
/*int main()
{
	dimen_t num,i;
	double *testdata;
	num = 17;
	testdata = (double *) malloc( sizeof( double ) * num );
	for( i=0; i<num; i++)
	{
		testdata[i] = (double) (1.0*rand() / RAND_MAX *(58) );
		if(i==4)
			testdata[i]=26;
		printf("%.3f, ",testdata[i]);
	}
	printf("\n");
	mergeSort(testdata,num);
	print_sort(testdata, num);
	reverse_sort(testdata , num);
	print_sort(testdata, num);
	free(testdata);
}*/

/*
{
	double *testdata;
	msortind_t data;
	msortind_t *testdataind;
	msortind_t *data_rec;
	dimen_t num,i;

	num=31;
	testdataind = &data;
	testdata = (double *) malloc( sizeof( double ) * num );
	for( i=0; i<num; i++)
	{
		testdata[i] = (double) (1.0*rand() / RAND_MAX *(58) );
		printf("%.3f, ",testdata[i]);
	}
	printf("\n");
	msortind_init(testdataind, num);
	testdataind->vector = testdata;
	print_sort_ind(testdataind);
	mergesort_ind(testdataind);
	print_sort_ind(testdataind);
	data_rec = recover_ind(testdataind);
	print_sort_ind(data_rec);
}*/

int reverse_sort(double *vector, dimen_t num)
{
	double * tempvector;
	dimen_t i,end;
	
	end = num-1;
	if(_test_reverse_sorted(vector, num))
		return 1;
	mergeSort(vector,num);
	tempvector = (double *) malloc(sizeof(double)*num);
	memcpy(tempvector,vector,sizeof(double)*num);
	for(i=0;i<num;i++)
		vector[i]=tempvector[end--];
	free(tempvector);
	return 1;
}

static int _test_reverse_sorted(double *vector, dimen_t num)
{
	dimen_t i;
	
	for(i=0;i<num-1;i++)
	{
		if(vector[i]<vector[i+1])
			return 0;
	}
	return 1;
}

msortind_t *recover_ind(msortind_t *data_ind)
{
	msortind_t *data_rec;
	dimen_t i;

	data_rec = (msortind_t *) malloc (sizeof(msortind_t));
	data_rec->num = data_ind->num;
	data_rec->vector = (double *) malloc (sizeof(double) * data_rec->num);
	msortind_init(data_rec, data_rec->num);
	for(i=0;i<data_rec->num;i++)
		data_rec->vector[data_ind->index[i]]=data_ind->vector[i];
	return data_rec;
}


int msortind_init(msortind_t *data_ind,dimen_t num)
{
	assert(data_ind != NULL);
	dimen_t i;

	data_ind->num = num;
	data_ind->index = (dimen_t *) malloc(sizeof(dimen_t) * num);
	for(i=0;i<num;i++)
		data_ind->index[i]=i;
	return 1;
}

void print_sort_ind(msortind_t *data_ind)
{
	dimen_t i;

	for(i=0; i<data_ind->num; i++)
		printf("%.3f, ",data_ind->vector[i]);
	printf("\n");
	for(i=0; i<data_ind->num;i++)
		printf("%u, ",data_ind->index[i]);
	printf("\n");
}

void print_sort(double *vector,dimen_t num)
{
	dimen_t i;

	for( i=0; i<num; i++)
	{
		printf("%.3f, ",vector[i]);
	}
	printf("\n");
}

static int _merge_ind(msortind_t *data_ind, msortind_t * tmpdata_ind, dimen_t part1start,dimen_t part1end,dimen_t part2start,dimen_t part2end)
{
	dimen_t temp_position, count, i, start_vector;
	
	temp_position = part1start;
	count = part2end - part1start + 1;
	start_vector = part1start;
	while(part1start <= part1end && part2start <= part2end)
	{
		if(data_ind->vector[part1start] < data_ind->vector[part2start])
		{
			tmpdata_ind->vector[temp_position]=data_ind->vector[part1start];
			tmpdata_ind->index[temp_position]=data_ind->index[part1start];
			temp_position++; part1start++;
		}
		else
		{
			tmpdata_ind->vector[temp_position]=data_ind->vector[part2start];
			tmpdata_ind->index[temp_position]=data_ind->index[part2start];
			temp_position++; part2start++;
		}
	}
	while(part1start <= part1end)
	{
		tmpdata_ind->vector[temp_position]=data_ind->vector[part1start];
		tmpdata_ind->index[temp_position]=data_ind->index[part1start];
		temp_position++; part1start++;
	}
	while(part2start <= part2end)
	{
		tmpdata_ind->vector[temp_position]=data_ind->vector[part2start];
		tmpdata_ind->index[temp_position]=data_ind->index[part2start];
		temp_position++; part2start++;
	}
	for(i=0; i < count ;i++,start_vector++)
	{
		data_ind->vector[start_vector]=tmpdata_ind->vector[start_vector];
		data_ind->index[start_vector]=tmpdata_ind->index[start_vector];
	}
	return 0;
}

static int _merge(double * vector, double * temp, dimen_t part1start, dimen_t part1end, dimen_t part2start, dimen_t part2end)
{
	dimen_t temp_position, count, i, start_vector;
	
	temp_position = part1start;
	count = part2end - part1start + 1;
	start_vector = part1start;
	while (part1start <= part1end && part2start <= part2end)
	{
		if(vector[part1start] < vector[part2start])
			temp[temp_position++] = vector[part1start++];
		else
			temp[temp_position++] = vector[part2start++];
	}
	while(part1start <= part1end)
		temp[temp_position++] = vector[part1start++];
	while (part2start <= part2end)
		temp[temp_position++] = vector[part2start++];
	for (i=0; i < count ;i++,start_vector++)
		vector[start_vector] = temp[start_vector];
	return 0;
}

int mergesort_ind(msortind_t *data_ind)
{
	msortind_t * tmpdata_ind;
	dimen_t part1start,part1end,part2start,part2end,subsize,subs;
	
	if(_max_msort_ind(data_ind))
		return 1;
	tmpdata_ind = (msortind_t *) malloc (sizeof(msortind_t));
	tmpdata_ind->num = data_ind->num;
	tmpdata_ind->vector =(double *) malloc(sizeof(double)*tmpdata_ind->num);
	tmpdata_ind->index= (dimen_t *)malloc(sizeof(dimen_t)*tmpdata_ind->num);
	
	if(!tmpdata_ind->vector)
	{
		printf("Note: out of memory, please check");
		exit(1);
	}
	for(subsize = 1; subsize <data_ind->num-1; subsize <<=1)
	{
		part1start = 0;
		subs = 0;
		while(subs<(data_ind->num))
		{
			part2start = part1start+subsize;
			part1end  = part2start  - 1;
			part2end = _min_msort(data_ind->num-1,part2start + subsize - 1);
			if(part1end < part2end)
				_merge_ind(data_ind, tmpdata_ind, part1start, part1end, part2start, part2end);
			else
				_merge_ind(data_ind, tmpdata_ind, part1start, part2end, part2start, part2end);
			part1start = part2end + 1;
			subs += subsize<<1;
		}
	}
	free(tmpdata_ind->vector);
	free(tmpdata_ind->index);
	free(tmpdata_ind);
	return 1;
}

int mergeSort(double *vector, dimen_t num)
{
	double *tmpvector;
	dimen_t part1start,part1end,part2start,part2end,subsize,subs;
	
	if(_max_msort(vector,num))
		return 1;
	tmpvector = (double *) malloc (sizeof(double) *num);
	if(tmpvector == NULL)
	{
		printf("Note: out of memory, please check");
		exit(1);
	}
	for(subsize = 1; subsize < num-1; subsize <<=1)
	{
		part1start = 0;
		subs = 0;
		while(subs<num)
		{
			part2start = part1start+subsize;
			part1end  = part2start  - 1;
			part2end = _min_msort(num-1,part2start + subsize - 1);
			if (part1end < part2end)
				_merge( vector, tmpvector, part1start, part1end, part2start, part2end );
			else
				_merge( vector, tmpvector, part1start, part2end, part2start, part2end );
			part1start = part2end + 1;
			subs += subsize<<1;
		}
		//print_sort(vector,num);
	}
	free(tmpvector);
	return 1;
}


static int _max_msort_ind(msortind_t *data_ind)
{
	double max_value;
	dimen_t max_index,i;
	int flag = 1;
	
	max_value = data_ind->vector[data_ind->num-1];
	max_index = data_ind->num-1;
	for(i=0;i<data_ind->num-1;i++)
	{
		if(max_value < data_ind->vector[i])
		{
			max_value = data_ind->vector[i];
			max_index = i;
		}
		if(flag)
		{
			if(data_ind->vector[i]>data_ind->vector[i+1])
				flag = 0;
		}
	}
	data_ind->vector[max_index] = data_ind->vector[data_ind->num-1];
	data_ind->vector[data_ind->num-1] = max_value;
	data_ind->index[max_index] = data_ind->index[data_ind->num-1];
	data_ind->index[data_ind->num-1] = max_index;
	return flag;
}

static int _max_msort(double *vector, dimen_t num)
{
	//get max to the last
	double max_value;
	dimen_t max_index,i;
	int flag = 1;

	max_value = vector[num-1];
	max_index = num-1;
	for(i=0;i<num-1;i++)
	{
		if(max_value < vector[i])
		{
			max_value = vector[i];
			max_index = i;
		}
		//to confirm whether vector has been sorted or not;
		//if return 0, means that vector has not been sorted;
		if(flag)
		{
			if(vector[i] > vector[i+1])
				flag = 0;
		}
	}
	//printf("\nmax: %.3f\n",max_value);
	vector[max_index] = vector[num-1];
	vector[num-1] = max_value;
	return flag;
}

static inline dimen_t _min_msort(dimen_t num1, dimen_t num2)
{
	return num1 > num2 ? num2 : num1;
}

