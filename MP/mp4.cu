#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
__constant__ float Mask[3][3][3];
const int Tile_Width = 4;


__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float subTile[Tile_Width+2][Tile_Width+2][Tile_Width+2];
  
  int hig_o = blockIdx.z * Tile_Width + threadIdx.z;
  int row_o = blockIdx.y * Tile_Width + threadIdx.y;
  int col_o = blockIdx.x * Tile_Width + threadIdx.x;

  int hig_i = hig_o-1;
  int row_i = row_o-1;
  int col_i = col_o-1;
  
  float value = 0;
  if ((hig_i>=0)&&(hig_i<z_size)&&(row_i>=0)&&(row_i<y_size)&&(col_i>=0)&&(col_i<x_size))
  {
    subTile[threadIdx.z][threadIdx.y][threadIdx.x] = input[hig_i*(x_size*y_size)+row_i*x_size+col_i];
  } else
  {
    subTile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();
  
  if (threadIdx.z < Tile_Width && threadIdx.y < Tile_Width && threadIdx.x < Tile_Width)
  {
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        for (int m = 0; m < 3; m++)
        {
          value += Mask[i][j][m] * subTile[threadIdx.z+i][threadIdx.y+j][threadIdx.x+m];
        }
      }
    }
    if (hig_o < z_size && row_o < y_size && col_o < x_size)
    {
      output[hig_o*(x_size*y_size)+row_o*x_size+col_o] = value;
    }
  }
}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, sizeof(float)*(inputLength-3));
  cudaMalloc((void**) &deviceOutput, sizeof(float)*(inputLength-3));
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput,&hostInput[0]+3,sizeof(float)*(inputLength-3),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mask,hostKernel,kernelLength*sizeof(float));
  
  for (int u = 0; u < 27;u++)
  {
    printf("%.f ",hostKernel[u]);
  }
  printf("\n");
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 BlockDim(Tile_Width+3-1, Tile_Width+3-1, Tile_Width+3-1);
  dim3 GridDim(ceil(x_size/1.0/Tile_Width), ceil(y_size/1.0/Tile_Width), ceil(z_size/1.0/Tile_Width));
  
  wbLog(TRACE, "The BlockDim is ", Tile_Width+3-1,Tile_Width+3-1,Tile_Width+3-1);
  wbLog(TRACE, "The GridDim is ",ceil(x_size/1.0/Tile_Width), ceil(y_size/1.0/Tile_Width), ceil(z_size/1.0/Tile_Width));
  
  //@@ Launch the GPU kernel here
  conv3d<<<GridDim,BlockDim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[0]+3,deviceOutput,sizeof(float)*(inputLength-3),cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}