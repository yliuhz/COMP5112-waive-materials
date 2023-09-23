#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

struct B
{
    int bb = -1;
};

struct A
{
    B *b;
    int aa = 1;
};

inline void setm(A *ha, A **da)
{
    cudaMalloc((void**)da, sizeof(A));
    cudaMalloc((void**)&((*da)->b), sizeof(B) * 50000);
    cudaMemcpy((*da)->b, ha->b, sizeof(B) * 50000, cudaMemcpyHostToDevice);
}

inline void freem(A *ha, A **da)
{
    cudaFree((*da)->b);
    cudaFree((*da));
}

int main(void)
{
    A *ha, *da;
    
    ha = new A();
    ha->b = new B[5];

    setm(ha, &da);
    freem(ha, &da);
    delete[] ha->b;
    delete ha;
}
