//Tesla k20c 
// L1 Cache	16 KB (per SMX)
// L2 Cache	1280 KB
// SMX Count	13
// Memory Size	5 GB
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define DATATYPE int
#define SMEMSIZE 1024
#define REP 128
//bank conflict degree
#define conflictnum 1	//2 4 8 16 32

//input:  int *in1,int *in2,int its
//output: int *out,double *time
__global__ void shared_model_1(double *time,DATATYPE *in1,DATATYPE *in2,DATATYPE *out,int its)
{
	__shared__ DATATYPE smem1[SMEMSIZE];
	__shared__ DATATYPE smem2[SMEMSIZE];

	//threadIdx per block 
	unsigned int tid=threadIdx.x;
	//initial array
	while(tid<SMEMSIZE)
	{
		smem1[tid]=in1[tid];
		smem2[tid]=in2[tid];
		tid+=blockDim.x;
	}
	//q is index of the accessed array in shared memory
	//连续thread的q相差32，每conflictnum重复+1
	//连续thread访问同一个shared memory bank
	DATATYPE p,q=(threadIdx.x%conflictnum)*32+(threadIdx.x/conflictnum);
	double time_tmp=0.0;
	unsigned int start_time=0,stop_time=0;
	unsigned int i,j;
	for (i=0;i<its;i++)
	{
		//块内线程同步
		//确保线程块中的每个线程都执行完 __syncthreads()前面的语句后，才会执行下一条语句
		__syncthreads();
		start_time=clock();
#pragma unroll //没指定次数，对于常数次的循环，循环将完全展开，对于不确定次数的循环，循环将不会被展开
		for (j=0;j<REP;j++)
		{
			p=smem1[q];
			q=smem2[p];
		}
		stop_time=clock();
		time_tmp+=(stop_time-start_time);
	}
	time_tmp=time_tmp/REP/its;
	//calculate the results of p+q to avoid compiler optimizations
	out[blockDim.x*blockIdx.x+threadIdx.x] = p+q;
	time[blockDim.x*blockIdx.x+threadIdx.x] = time_tmp;
}

int main_test(int blocks,int threads,DATATYPE *h_in1,DATATYPE *h_in2)
{
	int its=30;
	//int blocks=1,threads=32;
	//分配array显存
	DATATYPE *d_in1,*d_in2;
	cudaMalloc((void**)&d_in1,sizeof(DATATYPE)*SMEMSIZE);
	cudaMalloc((void**)&d_in2,sizeof(DATATYPE)*SMEMSIZE);

	//copy数据到GPU上的array
	cudaMemcpy(d_in1,h_in1,sizeof(DATATYPE)*SMEMSIZE,cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2,h_in2,sizeof(DATATYPE)*SMEMSIZE,cudaMemcpyHostToDevice);

	//each thread has it's own time
	double *h_time,*d_time;
	DATATYPE *d_out;
	h_time=(double*)malloc(sizeof(double)*blocks*threads);
	cudaMalloc((void**)&d_time,sizeof(double)*blocks*threads);
	cudaMalloc((void**)&d_out,sizeof(DATATYPE)*blocks*threads);

	//d_time 记录每个thread的执行时间
	shared_model_1<<<blocks,threads>>>(d_time,d_in1,d_in1,d_out,its);	
	cudaMemcpy(h_time,d_time,sizeof(double)*blocks*threads,cudaMemcpyDeviceToHost);

	//统计threads执行时间的平均值和最值
	double avert=0.0,maxt=0.0,mint=99999.9;
	int nn=0;
	for (int i=0;i<blocks;i++)
	{
		for (int j=0;j<threads;j+=32)
		{
			avert+=h_time[i*threads+j];
			nn++;
			if (maxt<h_time[i*threads+j])
			{
				maxt=h_time[i*threads+j];
			}
			if (mint>h_time[i*threads+j])
			{
				mint=h_time[i*threads+j];
			}
		}
	}
	avert/=nn;

	//打印blocks、threads数，执行时间的平均值和最值
	//printf("blocks number：%d\t threads number：%d\t\t avert Exeu_time：%f\t mint Exeu_time：%f\t maxt Exeu_time：%f\n", blocks,threads,avert,mint,maxt);
	printf("%d\t %d\t\t %f\t %f\t %f\n", blocks,threads,avert,mint,maxt);
	cudaFree(d_time);
	cudaFree(d_out);
	cudaFree(d_in1);
	cudaFree(d_in2);
	free(h_time);
	return 0;
}

//初始化数组，a[i]=i
void init_order(DATATYPE *a,int n)
{
	for (int i=0;i<n;i++)
	{
		a[i]=i;
	}
}

int main()
{
	//初始化shared memory大小的数组h_in1
	DATATYPE *h_in1;
	h_in1=(DATATYPE*)malloc(sizeof(DATATYPE)*SMEMSIZE);

	init_order(h_in1,SMEMSIZE);


/*
	for (int i=0;i<SMEMSIZE;i+=32)
	{
		for (int j=0;j<32;j++)
		{
			printf("%d\t",h_in3[i+j]);
		}
		printf("\n");
	}
*/
	printf("conflictnum is： %d\n",conflictnum);
	printf("blocks\t threads\t aver \t\t min \t\t max \t(clocks)\n");

	//main_test(1,32,h_in1,h_in1,1);
	//main_test(1,32,h_in2,h_in2,2);
	//main_test(1,32,h_in3,h_in3,3);
	//main_test(1,512,h_in1,h_in1,1);
	//main_test(1,512,h_in2,h_in2,2);
	//main_test(1,512,h_in3,h_in3,3);



	for (int i=0;i<=1;i+=32)
	{
		int blocks=i;
		if (i==0)
		{
			blocks=1;
		}
		for (int j=0;j<=512;j+=32)
		{
			int threads=j;
			if (j==0)
			{
				threads=1;
			}
			main_test(blocks,threads,h_in1,h_in1);
		}
	}




/*
	for (int i=0;i<=1024;i+=32)
	{
		int blocks=i;
		if (i==0)
		{
			blocks=1;
		}
		for (int j=256;j<=256;j+=32)
		{
			int threads=j;
			if (j==0)
			{
				threads=1;
			}
			main_test(blocks,threads,h_in1,h_in1);
		}
	}
*/


	free(h_in1);

	return 0;
}
