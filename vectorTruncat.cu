#include <stdio.h>
#include <iostream>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include "CPrecisionTimer.h"

// float half truncation: truncate lower 16bit mantisa
void half_truncat(float src[], size_t numofelem, short comp[])
{
    //assert(sizeof(src[0])==4); // float
    short* sptr = NULL;
    //#pragma omp parallel for
    for (size_t i=0; i < numofelem ; i++)
    {    
        sptr = (short*)(&src[i]);
        comp[i] = *(sptr+1);
    }
}


/**
 * CUDA Kernel Device code
 *
 * Vector Truncation: for each float element: 32bit -> nbit .
 *
 */
__global__ void
vectorTruncat_2D(const float *src, short *comp, size_t numElements, size_t N /*total column threads*/)
{
	size_t col = blockIdx.x * blockDim.x + threadIdx.x;
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = col + row * N; // = thread index = element index

    if (index < numElements)
    {
        short* sptr = NULL;
        sptr = (short*)(&src[index]);
        comp[index] = *(sptr+1);
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its sizeu
    // size_t numElements = 2000592040;  // 7.45GB
    size_t numElements = 1073741824; // 4GB
    // size_t numElements  = 536870912; // 2GB
    // size_t numElements = 268435456; // 1GB
    // size_t numElements = 131072000; // 500 MB
    // size_t numElements = 15728640; // 60 MB
   
    size_t size = numElements * sizeof(float);
    printf("[Vector Truncation of %d elements]\n", numElements);

    // Allocate the host input vector
    float *h_raw_data = (float *)malloc(size);
    // Allocate the CPU truncation results 
    short *h_CPU_res = (short *)malloc(size/2);
    // Allocate the GPU truncation results 
    short *h_GPU_res = (short *)malloc(size/2);
    // Verify that allocations succeeded
    if (h_raw_data == NULL || h_CPU_res == NULL || h_GPU_res == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (size_t i = 0; i < numElements; ++i)
    {
        h_raw_data[i] = rand()/(float)RAND_MAX;
    }
    printf("**************** Input Data Generated ****************\n");

    // Allocate the device input vector A
    float *d_raw_data = NULL;
    err = cudaMalloc((void **)&d_raw_data, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device result
    short *d_GPU_res = NULL;
    err = cudaMalloc((void **)&d_GPU_res, size/2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    //----------------------- CPU Truncation ------------------------------------------------
    CPrecisionTimer Timer; Timer.Start();
    half_truncat(h_raw_data, numElements, h_CPU_res);
    printf("*********** CPU Truncation Finished: Time = %f second ********************\n", Timer.Stop());
    //---------------------------------------------------------------------------------------


    //----------------------- GPU Truncation ------------------------------------------------
    // Copy the host input vectors to the device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_raw_data, h_raw_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
	CPrecisionTimer Timer2; Timer2.Start();
	dim3 dimBlock(32,32); // Block = 32 x 32 = 1024 threads
	size_t numofBlocks = (numElements-1)/1024 + 1;
	size_t M = ((size_t) sqrt(numofBlocks)) + 1;
	dim3 dimGrid(M,M); // Grid = M x M Blocks 
	assert( M*M*1024 >= numElements );
    printf("CUDA kernel launch with (%d,%d) blocks, each block = (32,32) threads\n", M,M);
     
    vectorTruncat_2D<<<dimGrid, dimBlock>>>(d_raw_data, d_GPU_res, numElements, M*32);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorTruncat kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("*********** GPU Truncation Finished: Pure Kernel Time = %f second ********************\n", Timer2.Stop());

    // Copy the device result vector in device memory to the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_GPU_res, d_GPU_res, size/2, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }  
    //---------------------------------------------------------------------------------------
    
    // Verify that the result vector is correct
    for (size_t i = 0; i < numElements; ++i)
    {
        if ( h_CPU_res[i] != h_GPU_res[i] )
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("********** Result Comparison: PASSED *************\n");

    // Free device global memory
    err = cudaFree(d_raw_data);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_GPU_res);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_raw_data);
    free(h_CPU_res);
    free(h_GPU_res);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    /*err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/

    printf("Done\n");

    return 0;
}

