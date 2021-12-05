// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len){

  __shared__ float T[2*BLOCK_SIZE];
  
  if (2*blockIdx.x*blockDim.x+threadIdx.x<len){
    T[threadIdx.x] = input[2*blockIdx.x*blockDim.x + threadIdx.x];
  } else{
    T[threadIdx.x] = 0;
  }

  if (2*blockIdx.x*blockDim.x+blockDim.x+threadIdx.x<len){
    T[blockDim.x+threadIdx.x] = input[2*blockIdx.x*blockDim.x+blockDim.x+threadIdx.x];
  } else{
    T[blockDim.x+threadIdx.x] = 0;
  }

  for (int stride = 1; stride < 2*BLOCK_SIZE; stride *=2){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if (index < 2*BLOCK_SIZE && (index-stride) >= 0) T[index] += T[index-stride];
  }
  
  for (int stride = BLOCK_SIZE/2; stride > 0; stride /=2){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2-1;
    if ((index+stride) < 2*BLOCK_SIZE) T[index+stride] += T[index];
  }
  
  __syncthreads();
  if (2*blockIdx.x*blockDim.x+threadIdx.x<len){
    output[2*blockIdx.x*blockDim.x + threadIdx.x] = T[threadIdx.x];
  } else{
    output[2*blockIdx.x*blockDim.x + threadIdx.x] = 0;
  }

  if (2*blockIdx.x*blockDim.x+blockDim.x+threadIdx.x<len){
    output[2*blockIdx.x*blockDim.x + blockDim.x + threadIdx.x] = T[blockDim.x+threadIdx.x];
  } else{
    output[2*blockIdx.x*blockDim.x + blockDim.x + threadIdx.x] = 0;
  }
}

__global__ void find_sum(float *input, float *output,int len){
  
  if (threadIdx.x == 0){
    output[threadIdx.x] = 0;
  }
  else if (threadIdx.x<len){
    output[threadIdx.x] = input[threadIdx.x*2*BLOCK_SIZE-1];
  }
}

__global__ void add_sum(float *output, float *block_sum, int len, int gridnum) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  
  if (2*blockIdx.x*blockDim.x + threadIdx.x < len){
    output[2*blockIdx.x*blockDim.x + threadIdx.x] += block_sum[blockIdx.x];
  }
  
  if (2*blockIdx.x*blockDim.x + blockDim.x + threadIdx.x < len){
    output[2*blockIdx.x*blockDim.x + blockDim.x + threadIdx.x] += block_sum[blockIdx.x];
  }
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *block_sum;
    
  int numElements; // number of elements in the list
  
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);  
  int gridnum = ceil(numElements*1./2/BLOCK_SIZE);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  
  wbTime_stop(Generic, "Importing data and creating memory on host");
  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&block_sum, gridnum * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  
  
  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  
  
  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  
  
  //@@ Initialize the grid and block dimensions here
  dim3 GridDim1(gridnum,1,1);
  dim3 BlockDim1(BLOCK_SIZE,1,1);
  
  wbLog(TRACE, "The grid dimention is ",gridnum ," block size is ",BLOCK_SIZE);
    
  dim3 GridDim2(1,1,1);  // Assume the block sums's length is smaller the block length, so the first kernal can be reused
  dim3 BlockDim2(BLOCK_SIZE,1,1); 
  
  wbLog(TRACE, "The grid dimention is 1"," block size is ",BLOCK_SIZE);

  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  scan<<<GridDim1,BlockDim1>>>(deviceInput,deviceOutput,numElements);
  cudaDeviceSynchronize();
  find_sum<<<GridDim2,BlockDim2>>>(deviceOutput,block_sum,gridnum);
  cudaDeviceSynchronize();
  scan<<<GridDim2,BlockDim2>>>(block_sum,block_sum,gridnum);
  cudaDeviceSynchronize();
  add_sum<<<GridDim1,BlockDim1>>>(deviceOutput,block_sum,numElements,gridnum);
  cudaDeviceSynchronize();
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");             
  
  
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}