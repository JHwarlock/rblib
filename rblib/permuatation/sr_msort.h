/*
 * Prog Description:
 *
 * Author: Rong Zheng-Qin
 * Email: rongzhengqin@genomics.org.cn  or  rongzhengqin@basepedia.com
 * */

#ifndef SR_MSORT_H
#define SR_MSORT_H

#include<stdint.h>

#ifndef DIMEN_T
#define DIMEN_T
typedef uint32_t dimen_t;
#endif

#ifndef MSORTIND_T
#define MSORTIND_T
typedef struct
{
	double *vector;
	dimen_t *index;
	dimen_t num;
}msortind_t;
#endif

int mergeSort(double *vector, dimen_t num);
int msortind_init(msortind_t *data_ind,dimen_t num);
int mergesort_ind(msortind_t *data);
msortind_t *recover_ind(msortind_t *data);
int reverse_sort(double *vector, dimen_t num);

/*
static inline dimen_t _min_msort(dimen_t num1, dimen_t num2);
static int _merge(double * vector, double * temp, dimen_t part1start, dimen_t part1end, dimen_t part2start, dimen_t part2end);
static int _merge_ind(msortind_t *data_ind, msortind_t * tmpdata_ind, dimen_t part1start,dimen_t part1end,dimen_t part2start,dimen_t part2end);
static int _test_reverse_sorted(double *vector, dimen_t num);
*/

//int test_sorted(double *vector,dimen_t num);
/*
static int _max_msort(double *vector, dimen_t num);
static int _max_msort_ind(msortind_t *data_ind);
*/


//void swap_msort(double *a,double *b);

// which is used just for test
void print_sort(double *A,dimen_t N);
void print_sort_ind(msortind_t *data);


#endif
