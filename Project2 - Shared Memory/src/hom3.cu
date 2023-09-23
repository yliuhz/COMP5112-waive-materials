#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <cstring>

using namespace std;

#define REAL double
#define HANDLE_ERROR checkCudaErrors

#define START_GPU {\
cudaEvent_t     start, stop;\
float   elapsedTime;\
checkCudaErrors(cudaEventCreate(&start)); \
checkCudaErrors(cudaEventCreate(&stop));\
checkCudaErrors(cudaEventRecord(start, 0));\

#define END_GPU(xx) \
checkCudaErrors(cudaEventRecord(stop, 0));\
checkCudaErrors(cudaEventSynchronize(stop));\
checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
printf("%s", xx); \
printf("GPU Time used:  %3.1f ms\n", elapsedTime);\
checkCudaErrors(cudaEventDestroy(start));\
checkCudaErrors(cudaEventDestroy(stop));}


#define START_CPU {\
double start = omp_get_wtime();

#define END_CPU \
double end = omp_get_wtime();\
double duration = end - start;\
printf("CPU Time used: %3.1f ms\n", duration * 1000);}

#define DataNum 1024*1024*1024
REAL* hA;
REAL* dA, *dS, *dS1, *dS2, *dS3;
REAL resultOnCpu, resultOnGpuTotalAtomic, resultOnGpuPartialAtomic_1, \
resultOnGpuPartialAtomic_2, resultOnGpuPartialAtomic_3, \
resultOnGpuPartialAtomic_shared;

#define BLOCK_SIZE 256
#define INTER_SIZE_1 1024
#define INTER_SIZE_2 256
#define INTER_SIZE_3 16
#define INTER_SIZE_SHARED 1024

enum {
	TOTAL_ATOMIC,
	PARTIAL_ATOMIC_1, //
	PARTIAL_ATOMIC_2,
	PARTIAL_ATOMIC_3,
	PARTIAL_ATOMIC_SHARED,
	NO_ATOMIC
};

void prepareData()
{
	size_t size = DataNum * sizeof(REAL);

	hA = (REAL*)malloc(size);

	// Initialize the host input vectors
	for (int i = 0; i < DataNum; ++i)
	{
		hA[i] = rand() / (REAL)RAND_MAX;
	}
}

void pushData()
{
	size_t size = DataNum * sizeof(REAL);
	size_t dSSize = 1 * sizeof(REAL);
	size_t dS1Size = INTER_SIZE_1 * sizeof(REAL);
	size_t dS2Size = INTER_SIZE_2 * sizeof(REAL);
	size_t dS3Size = INTER_SIZE_3 * sizeof(REAL);

	cudaMalloc((void**)&dA, size);
	cudaMalloc((void**)&dS, dSSize);
	cudaMalloc((void**)&dS1, dS1Size);
	cudaMalloc((void**)&dS2, dS2Size);
	cudaMalloc((void**)&dS3, dS3Size);

	cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
}

void sumOnCpu()
{
	START_CPU

	for (int i = 0; i < DataNum; i++) {
		resultOnCpu += hA[i];
	}

	END_CPU
}

__global__ void sumOnGpuTotalAtomic(REAL* a, int n, REAL* result)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (id < n) {
		atomicAdd(&result[0], a[id]);
		id += stride;
	}
}

__global__ void sumOnGpuPartialAtomic(REAL* a, int n, REAL* result, REAL rate)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (id < n) {
		int interIndex = (int)(id * 1.0 / rate);
		atomicAdd(&result[interIndex], a[id]);
		id += stride;
	}
}

__global__ void sumOnGpuPartialAtomicShared(REAL* a, int n, REAL* result, REAL rate)
{
	__shared__ REAL temp[INTER_SIZE_SHARED];
	temp[threadIdx.x] = 0.0;
	__syncthreads();

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (id < n) {
		int interIndex = (int)(id * 1.0 / rate);
		atomicAdd(&temp[interIndex], a[id]);
		id += stride;
	}

	__syncthreads();
	
	atomicAdd(&result[0], temp[threadIdx.x]);
}

void sumOnGpu(int atomic)
{
	char str[100] = "";
	
	START_GPU

	cudaDeviceProp  prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
	int blocks = prop.multiProcessorCount;

	cudaMemset(dS, 0, 1 * sizeof(REAL));
	if (atomic == TOTAL_ATOMIC){
		sumOnGpuTotalAtomic << <blocks * 2, 256 >> > (dA, DataNum, dS);
		cudaMemcpy(&resultOnGpuTotalAtomic, &dS[0], sizeof(REAL), cudaMemcpyDeviceToHost);
		sprintf(str, "Total Atomic ");
	}
	else if (atomic == PARTIAL_ATOMIC_1) {
		//** 1st layer
		cudaMemset(dS1, 0, INTER_SIZE_1 * sizeof(REAL));
		REAL rate = DataNum * 1.0 / INTER_SIZE_1;
		sumOnGpuPartialAtomic << <blocks * 2, 256 >> > (dA, DataNum, dS1, rate);
		//** output layer
		sumOnGpuTotalAtomic << <blocks * 2, 256 >> > (dS1, INTER_SIZE_1, dS);
		cudaMemcpy(&resultOnGpuPartialAtomic_1, &dS[0], sizeof(REAL), cudaMemcpyDeviceToHost);
		sprintf(str, "Partial Atomic(1 layer, %d) ", INTER_SIZE_1);
	}
	else if (atomic == PARTIAL_ATOMIC_2) {
		//** 1st layer
		cudaMemset(dS1, 0, INTER_SIZE_1 * sizeof(REAL));
		REAL rate = DataNum * 1.0 / INTER_SIZE_1;
		sumOnGpuPartialAtomic << <blocks * 2, 256 >> > (dA, DataNum, dS1, rate);
		//** 2nd layer
		cudaMemset(dS2, 0, INTER_SIZE_2 * sizeof(REAL));
		rate = INTER_SIZE_1 * 1.0 / INTER_SIZE_2;
		sumOnGpuPartialAtomic << <blocks * 2, 256 >> > (dS1, INTER_SIZE_1, dS2, rate);
		//** output layer
		sumOnGpuTotalAtomic << <blocks * 2, 256 >> > (dS2, INTER_SIZE_2, dS);
		cudaMemcpy(&resultOnGpuPartialAtomic_2, &dS[0], sizeof(REAL), cudaMemcpyDeviceToHost);
		sprintf(str, "Partial Atomic(2 layers, %d -> %d) ", INTER_SIZE_1, INTER_SIZE_2);
	}
	else if (atomic == PARTIAL_ATOMIC_3) {
		//** 1st layer
		cudaMemset(dS1, 0, INTER_SIZE_1 * sizeof(REAL));
		REAL rate = DataNum * 1.0 / INTER_SIZE_1;
		sumOnGpuPartialAtomic << <blocks * 2, 256 >> > (dA, DataNum, dS1, rate);
		//** 2nd layer
		cudaMemset(dS2, 0, INTER_SIZE_2 * sizeof(REAL));
		rate = INTER_SIZE_1 * 1.0 / INTER_SIZE_2;
		sumOnGpuPartialAtomic << <blocks * 2, 256 >> > (dS1, INTER_SIZE_1, dS2, rate);
		//** 3rd layer
		cudaMemset(dS3, 0, INTER_SIZE_3 * sizeof(REAL));
		rate = INTER_SIZE_2 * 1.0 / INTER_SIZE_3;
		sumOnGpuPartialAtomic << <blocks * 2, 256 >> > (dS2, INTER_SIZE_2, dS3, rate);
		//** output layer
		sumOnGpuTotalAtomic << <blocks * 2, 256 >> > (dS3, INTER_SIZE_3, dS);
		cudaMemcpy(&resultOnGpuPartialAtomic_3, &dS[0], sizeof(REAL), cudaMemcpyDeviceToHost);
		sprintf(str, "Partial Atomic(3 layers, %d -> %d -> %d) ", INTER_SIZE_1, INTER_SIZE_2, INTER_SIZE_3);
	}
	else if (atomic == PARTIAL_ATOMIC_SHARED) {
		//** output layer
		REAL rate = DataNum * 1.0 / INTER_SIZE_SHARED;
		sumOnGpuPartialAtomicShared << <blocks * 2, INTER_SIZE_SHARED >> > (dA, DataNum, dS, rate);

		cudaMemcpy(&resultOnGpuPartialAtomic_shared, &dS[0], sizeof(REAL), cudaMemcpyDeviceToHost);
		sprintf(str, "Partial Atomic(1 layer, shared memory, %d) ", INTER_SIZE_SHARED);
	}
	

	END_GPU(str)
}

void freeData()
{
	free(hA);

	cudaFree(dA);
	cudaFree(dS);
	cudaFree(dS1);
	cudaFree(dS2);
	cudaFree(dS3);
}

int main(void)
{
	prepareData();
	sumOnCpu();

	pushData();
	sumOnGpu(TOTAL_ATOMIC);
	sumOnGpu(PARTIAL_ATOMIC_1);
	sumOnGpu(PARTIAL_ATOMIC_2);
	sumOnGpu(PARTIAL_ATOMIC_3);
	sumOnGpu(PARTIAL_ATOMIC_SHARED);
	
	printf("***********************\n");
	printf("Cpu result: %.3f,\n\
Total atomic result -(minus) cpu result: %.3f,\n\
Partial atomic(1 layer, %d) result -(minus) cpu result: %.3f,\n\
Partial atomic(2 layers, %d -> %d) result -(minus) cpu result: %.3f,\n\
Partial atomic(3 layers, %d -> %d -> %d) result -(minus) cpu result: %.3f,\n\
Partial atomic(shared memory, 1 layer, %d) result -(minus) cpu result: %.3f\n", \
		resultOnCpu, \
			resultOnGpuTotalAtomic - resultOnCpu, \
			INTER_SIZE_1, resultOnGpuPartialAtomic_1 - resultOnCpu, \
			INTER_SIZE_1, INTER_SIZE_2, resultOnGpuPartialAtomic_2 - resultOnCpu, \
			INTER_SIZE_1, INTER_SIZE_2, INTER_SIZE_3, resultOnGpuPartialAtomic_3 - resultOnCpu, \
			INTER_SIZE_SHARED, resultOnGpuPartialAtomic_shared - resultOnCpu);

	freeData();
}
