#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

# define TILE_WIDTH 8
# define new_TILE_WIDTH 16

__constant__ float MASK[7*7*16*4];

/*
Modify this function to implement the forward pass described in Chapter 16.
We have added an additional dimension to the tensors to support an entire mini-batch
The goal here is to be correct AND fast.

Function paramter definitions:
y - output
x - input
k - kernel
B - batch_size (number of images in x)      100/1000/10000
M - number of output feature maps           4   16
C - number of input feature maps            1   4
H - input height dimension                  86  40
W - input width dimension                   86  40
K - kernel height and width (K x K)         7   7
*/

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int w_grid = ceil(1.*W_out/TILE_WIDTH);
    int h_grid = ceil(1.*H_out/TILE_WIDTH);

    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = blockIdx.z/w_grid*TILE_WIDTH + threadIdx.y;
    int w = blockIdx.z%w_grid*TILE_WIDTH + threadIdx.x;


    if ((w < (W_out)) && (h < (H_out))) {
        float acc = 0.0f;
        for (int c = 0; c<C; c++){          // loop all input channels
            for (int p = 0; p<K; p++){      // loop over k*k filter
                for (int q = 0; q<K; q++){
                    acc += x4d(n,c,h+p,w+q)*k4d(m,c,p,q);
                }
            }
        }
        y4d(n,m,h,w) = acc;
    }
    #undef y4d
    #undef x4d
    #undef k4d
}


// modified host code, using overlap to enlarge efficiency
__host__ void GPUInterface::conv_forward_gpu_prolog(float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int stream_num = 5;
    int times = 10;
    int small_batch = B/stream_num/times;

    cudaMalloc((void**) device_x_ptr, sizeof(float)*B*C*H*W);
    cudaMalloc((void**) device_y_ptr, sizeof(float)*B*M*H_out*W_out);
    cudaMalloc((void**) device_k_ptr, sizeof(float)*K*K*C*M);

    int w_grid = ceil(1.*W_out/TILE_WIDTH);
    int h_grid = ceil(1.*H_out/TILE_WIDTH);
    int Z = w_grid*h_grid;

    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridDim(small_batch,M,Z);


    cudaStream_t stream0,stream1,stream2,stream3,stream4,stream5,stream6,stream7,stream8,stream9;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    // cudaStreamCreate(&stream5);
    // cudaStreamCreate(&stream6);
    // cudaStreamCreate(&stream7);
    // cudaStreamCreate(&stream8);
    // cudaStreamCreate(&stream9);

    for (int i=0; i<times; i++)
    {
        cudaMemcpyAsync(*device_x_ptr + (0+stream_num*i)*small_batch*C*H*W, host_x+(0+stream_num*i)*small_batch*C*H*W, sizeof(float)*small_batch*C*H*W, cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(*device_x_ptr + (1+stream_num*i)*small_batch*C*H*W, host_x+(1+stream_num*i)*small_batch*C*H*W, sizeof(float)*small_batch*C*H*W, cudaMemcpyHostToDevice,stream1);
        cudaMemcpyAsync(*device_x_ptr + (2+stream_num*i)*small_batch*C*H*W, host_x+(2+stream_num*i)*small_batch*C*H*W, sizeof(float)*small_batch*C*H*W, cudaMemcpyHostToDevice,stream2);
        cudaMemcpyAsync(*device_x_ptr + (3+stream_num*i)*small_batch*C*H*W, host_x+(3+stream_num*i)*small_batch*C*H*W, sizeof(float)*small_batch*C*H*W, cudaMemcpyHostToDevice,stream3);
        cudaMemcpyAsync(*device_x_ptr + (4+stream_num*i)*small_batch*C*H*W, host_x+(4+stream_num*i)*small_batch*C*H*W, sizeof(float)*small_batch*C*H*W, cudaMemcpyHostToDevice,stream4);
        // cudaMemcpyAsync(*device_x_ptr + (5+stream_num*i)*small_batch*C*H*W, host_x+(5+stream_num*i)*small_batch*C*H*W, sizeof(float)*small_batch*C*H*W, cudaMemcpyHostToDevice,stream5);
        // cudaMemcpyAsync(*device_x_ptr + (6+stream_num*i)*small_batch*C*H*W, host_x+(6+stream_num*i)*small_batch*C*H*W, sizeof(float)*small_batch*C*H*W, cudaMemcpyHostToDevice,stream6);
        // cudaMemcpyAsync(*device_x_ptr + (7+stream_num*i)*small_batch*C*H*W, host_x+(7+stream_num*i)*small_batch*C*H*W, sizeof(float)*small_batch*C*H*W, cudaMemcpyHostToDevice,stream7);
        // cudaMemcpyAsync(*device_x_ptr + (8+stream_num*i)*small_batch*C*H*W, host_x+(8+stream_num*i)*small_batch*C*H*W, sizeof(float)*small_batch*C*H*W, cudaMemcpyHostToDevice,stream8);
        // cudaMemcpyAsync(*device_x_ptr + (9+stream_num*i)*small_batch*C*H*W, host_x+(9+stream_num*i)*small_batch*C*H*W, sizeof(float)*small_batch*C*H*W, cudaMemcpyHostToDevice,stream9);
    
        conv_forward_kernel<<<gridDim,blockDim,0,stream0>>>(*device_y_ptr + (0+stream_num*i)*small_batch*M*H_out*W_out,*device_x_ptr + (0+stream_num*i)*small_batch*C*H*W,*device_k_ptr,small_batch,M,C,H,W,K);
        conv_forward_kernel<<<gridDim,blockDim,0,stream1>>>(*device_y_ptr + (1+stream_num*i)*small_batch*M*H_out*W_out,*device_x_ptr + (1+stream_num*i)*small_batch*C*H*W,*device_k_ptr,small_batch,M,C,H,W,K);
        conv_forward_kernel<<<gridDim,blockDim,0,stream2>>>(*device_y_ptr + (2+stream_num*i)*small_batch*M*H_out*W_out,*device_x_ptr + (2+stream_num*i)*small_batch*C*H*W,*device_k_ptr,small_batch,M,C,H,W,K);
        conv_forward_kernel<<<gridDim,blockDim,0,stream3>>>(*device_y_ptr + (3+stream_num*i)*small_batch*M*H_out*W_out,*device_x_ptr + (3+stream_num*i)*small_batch*C*H*W,*device_k_ptr,small_batch,M,C,H,W,K);
        conv_forward_kernel<<<gridDim,blockDim,0,stream4>>>(*device_y_ptr + (4+stream_num*i)*small_batch*M*H_out*W_out,*device_x_ptr + (4+stream_num*i)*small_batch*C*H*W,*device_k_ptr,small_batch,M,C,H,W,K);
        // conv_forward_kernel_shared_constant_multikernel<<<gridDim_choice,blockDim_choice,shemem_size_choice,stream5>>>(tile_size, *device_y_ptr + (5+stream_num*i)*small_batch*M*H_out*W_out,*device_x_ptr + (5+stream_num*i)*small_batch*C*H*W,*device_k_ptr,small_batch,M,C,H,W,K);
        // conv_forward_kernel_shared_constant_multikernel<<<gridDim_choice,blockDim_choice,shemem_size_choice,stream6>>>(tile_size, *device_y_ptr + (6+stream_num*i)*small_batch*M*H_out*W_out,*device_x_ptr + (6+stream_num*i)*small_batch*C*H*W,*device_k_ptr,small_batch,M,C,H,W,K);
        // conv_forward_kernel_shared_constant_multikernel<<<gridDim_choice,blockDim_choice,shemem_size_choice,stream7>>>(tile_size, *device_y_ptr + (7+stream_num*i)*small_batch*M*H_out*W_out,*device_x_ptr + (7+stream_num*i)*small_batch*C*H*W,*device_k_ptr,small_batch,M,C,H,W,K);
        // conv_forward_kernel_shared_constant_multikernel<<<gridDim_choice,blockDim_choice,shemem_size_choice,stream8>>>(tile_size, *device_y_ptr + (8+stream_num*i)*small_batch*M*H_out*W_out,*device_x_ptr + (8+stream_num*i)*small_batch*C*H*W,*device_k_ptr,small_batch,M,C,H,W,K);
        // conv_forward_kernel_shared_constant_multikernel<<<gridDim_choice,blockDim_choice,shemem_size_choice,stream9>>>(tile_size, *device_y_ptr + (9+stream_num*i)*small_batch*M*H_out*W_out,*device_x_ptr + (9+stream_num*i)*small_batch*C*H*W,*device_k_ptr,small_batch,M,C,H,W,K);
    
        cudaMemcpyAsync(host_y+(0+stream_num*i)*small_batch*M*H_out*W_out, *device_y_ptr + (0+stream_num*i)*small_batch*M*H_out*W_out, sizeof(float)*small_batch*M*H_out*W_out,cudaMemcpyDeviceToHost,stream0);
        cudaMemcpyAsync(host_y+(1+stream_num*i)*small_batch*M*H_out*W_out, *device_y_ptr + (1+stream_num*i)*small_batch*M*H_out*W_out, sizeof(float)*small_batch*M*H_out*W_out,cudaMemcpyDeviceToHost,stream1);
        cudaMemcpyAsync(host_y+(2+stream_num*i)*small_batch*M*H_out*W_out, *device_y_ptr + (2+stream_num*i)*small_batch*M*H_out*W_out, sizeof(float)*small_batch*M*H_out*W_out,cudaMemcpyDeviceToHost,stream2);
        cudaMemcpyAsync(host_y+(3+stream_num*i)*small_batch*M*H_out*W_out, *device_y_ptr + (3+stream_num*i)*small_batch*M*H_out*W_out, sizeof(float)*small_batch*M*H_out*W_out,cudaMemcpyDeviceToHost,stream3);
        cudaMemcpyAsync(host_y+(4+stream_num*i)*small_batch*M*H_out*W_out, *device_y_ptr + (4+stream_num*i)*small_batch*M*H_out*W_out, sizeof(float)*small_batch*M*H_out*W_out,cudaMemcpyDeviceToHost,stream4);
        // cudaMemcpyAsync(host_y+(5+stream_num*i)*small_batch*M*H_out*W_out, *device_y_ptr + (5+stream_num*i)*small_batch*M*H_out*W_out, sizeof(float)*small_batch*M*H_out*W_out,cudaMemcpyDeviceToHost,stream5);
        // cudaMemcpyAsync(host_y+(6+stream_num*i)*small_batch*M*H_out*W_out, *device_y_ptr + (6+stream_num*i)*small_batch*M*H_out*W_out, sizeof(float)*small_batch*M*H_out*W_out,cudaMemcpyDeviceToHost,stream6);
        // cudaMemcpyAsync(host_y+(7+stream_num*i)*small_batch*M*H_out*W_out, *device_y_ptr + (7+stream_num*i)*small_batch*M*H_out*W_out, sizeof(float)*small_batch*M*H_out*W_out,cudaMemcpyDeviceToHost,stream7);
        // cudaMemcpyAsync(host_y+(8+stream_num*i)*small_batch*M*H_out*W_out, *device_y_ptr + (8+stream_num*i)*small_batch*M*H_out*W_out, sizeof(float)*small_batch*M*H_out*W_out,cudaMemcpyDeviceToHost,stream8);
        // cudaMemcpyAsync(host_y+(9+stream_num*i)*small_batch*M*H_out*W_out, *device_y_ptr + (9+stream_num*i)*small_batch*M*H_out*W_out, sizeof(float)*small_batch*M*H_out*W_out,cudaMemcpyDeviceToHost,stream9);    
    }
 
    //Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
}





__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
