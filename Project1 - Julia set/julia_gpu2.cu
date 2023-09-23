/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
    __device__ cuComplex operator-(const cuComplex& a) {
        return cuComplex(r-a.r, i-a.i);
    }
};

__device__ int julia( int x, int y ) {
    const float scale = 2;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    // cuComplex c(-0.8, 0.156);
    // cuComplex c(-0.7269, 0.1889);
    cuComplex c(0.25, 0.5);
    cuComplex a(jx, jy);

    //float originMagnitude = a.magnitude2();

    // int i = 0;
    // for (i=0; i<200; i++) {
    //     a = a * a + c;
    //     if (a.magnitude2() > 1000)
    //         return 0;
    // }
    
    const int MAX_ITERATION = 200;
    const float epsilon = 0.1;
    int i=1;
    while(i<MAX_ITERATION){
        a = a * a + c;
        float delta = (a * a - a + c).magnitude2();
        if(delta < epsilon) return i;
        else if(a.magnitude2() > 1000) return -i;
        i++;
    }

    return 0;
}

__global__ void kernel( unsigned char *ptr ) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    unsigned int juliaValue = julia( x, y );

    const int colornum = 18;
    float colormap[colornum][3] = {
        255, 218, 185,
        0, 191, 255,		
        0, 255, 255,		
        0, 255, 127,		
        255, 255, 0,	
        205, 92, 92,	
        255, 20, 147,	
        138, 43, 226,	
        255, 69, 0,	
        255, 165, 0,
        139, 101, 8,	
        139, 71, 38,	
        148, 0, 211,	
        216, 191, 216,	
        255, 0, 0,	
        255, 110, 180,	
        156, 156, 156,				
        255, 225, 255		
    };

    int initcolor[3] = {250, 235, 215};
    // 渐进色, 设置初始颜色，用它乘迭代次数

    // ptr[offset*4 + 0] = (colormap[juliaValue % colornum][0]);
    // ptr[offset*4 + 1] = (colormap[juliaValue % colornum][1]);
    // ptr[offset*4 + 2] = (colormap[juliaValue % colornum][2]);

    ptr[offset*4 + 0] = (initcolor[0] * juliaValue) % 256;
    ptr[offset*4 + 1] = (initcolor[1] * juliaValue) % 256;
    ptr[offset*4 + 2] = (initcolor[2] * juliaValue) % 256;
    ptr[offset*4 + 3] = 255;

    // printf("%x\n", *(int*)ptr);
    // if(juliaValue < 0) printf("%x\n", *(int*)ptr);
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;

    dim3    grid(DIM,DIM);
    kernel<<<grid,1>>>( dev_bitmap );

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );
                              
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
                              
    bitmap.display_and_exit();
}

