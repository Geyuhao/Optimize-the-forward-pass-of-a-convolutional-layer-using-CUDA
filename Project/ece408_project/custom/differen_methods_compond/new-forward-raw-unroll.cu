#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

# define TILE_WIDTH1 16
# define TILE_WIDTH2 16

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


// Shared memory matrix multiplication and input matrix unrolling
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                        int numARows, int numAColumns,
                                        int numBRows, int numBColumns,
                                        int numCRows, int numCColumns){

    __shared__ float subTileM[TILE_WIDTH1][TILE_WIDTH1];
    __shared__ float subTileN[TILE_WIDTH1][TILE_WIDTH1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by*TILE_WIDTH1 + ty;
    int col = bx*TILE_WIDTH1 + tx;
    float pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH1 + 1; m++)
    {
        if (row < numARows && m*TILE_WIDTH1 + tx < numAColumns) {
            subTileM[ty][tx] = A[row*numAColumns + m*TILE_WIDTH1 + tx];
        } else {   
            subTileM[ty][tx] = 0;
        }

        if (col < numBColumns && m*TILE_WIDTH1 + ty < numBRows) {
            subTileN[ty][tx] = B[(m*TILE_WIDTH1 + ty)*numBColumns + col];
        } else{
            subTileN[ty][tx] = 0;
        }

        __syncthreads();
        if (row < numCRows && col < numCColumns){
            for (int k = 0; k < TILE_WIDTH1; k++){
                pvalue += subTileM[ty][k] * subTileN[k][tx];
            }    
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns){
        C[row*numCColumns + col] = pvalue;
    }
}

__global__ void unroll_kernel(int C, int H, int W, int K, const float* X, float* X_unroll) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int c, s, h_out, w_out, h_unroll, w_base, p, q, w_unroll;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    if (idx < C * W_unroll) {
        c = idx / W_unroll;
        s = idx % W_unroll;     // index in the input feature
        h_out = s / W_out;
        w_out = s % W_out;      // index in the output feature
        w_unroll = h_out * W_out + w_out;
        w_base = c * K * K;
        for(p = 0; p < K; p++) {
            for(q = 0; q < K; q++) {
                h_unroll = w_base + p * K + q;
                X_unroll[h_unroll * W_unroll + w_unroll] = X[c * H * W + (h_out+p) * W + w_out+q];
            }
        }
    }
}






__host__ void GPUInterface::conv_forward_gpu_prolog(float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    cudaMalloc((void**) device_x_ptr, sizeof(float)*B*C*H*W);
    cudaMalloc((void**) device_y_ptr, sizeof(float)*B*M*H_out*W_out);
    cudaMalloc((void**) device_k_ptr, sizeof(float)*K*K*C*M);

    cudaMemcpy(*device_x_ptr,host_x,sizeof(float)*B*C*H*W,cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr,host_k,sizeof(float)*K*K*C*M,cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(MASK,host_k,K*K*C*M*sizeof(float));
    
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

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
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    

    // used for unroll
    int H_unroll = C * K * K;
    int W_unroll = H_out * W_out;
    float* X_unroll;
    cudaMalloc(&X_unroll, sizeof(float)* W_unroll * H_unroll);

    dim3 gridDim2 (ceil((1.*C*H_out*W_out) / TILE_WIDTH1), 1, 1);
    dim3 blockDim2 (TILE_WIDTH1, 1, 1);

    dim3 gridDim1 (ceil(1.0 * W_unroll / TILE_WIDTH1),  ceil(1.0 *  M/ TILE_WIDTH1), 1);
    dim3 blockDim1 (TILE_WIDTH1, TILE_WIDTH1, 1);

    for (int b = 0; b < B; b += 1) {
        unroll_kernel<<<gridDim2, blockDim2>>>(C, H, W, K, device_x+b*C*H*W, X_unroll);
        matrixMultiplyShared<<<gridDim1, blockDim1>>>(device_k, X_unroll, device_y + b*M*W_unroll, 
                                                    M, H_unroll, 
                                                    H_unroll, W_unroll,
                                                    M, W_unroll);     // H_unroll is the height of X, M is the width of X
    }
    cudaFree(X_unroll);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    cudaMemcpy(host_y, device_y, sizeof(float)*B*M*H_out*W_out,cudaMemcpyDeviceToHost);
    // Free device memory
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
