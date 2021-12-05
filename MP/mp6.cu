// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

__global__ void Cast_char(float* input, unsigned char* output, int length)
{  
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  if (ix < length){
    output[ix] = (unsigned char) (255*input[ix]);
  }
}

__global__ void Cast_float (unsigned char* input, float* output, int length)
{  
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  if (ix < length){
    output[ix] = (float) (input[ix]/255.0);
  }
}

__global__ void Convert_Gray (unsigned char* input, unsigned char* output, int height, int weight)
{
  int ix = blockIdx.x*blockDim.x + threadIdx.x;
  int iy = blockIdx.y*blockDim.y + threadIdx.y;
  
  if (ix < weight && iy < height){
    int idx = iy*weight + ix;
    float r = input[3*idx];
    float g = input[3*idx+1];
    float b = input[3*idx+2];
    output[idx] = (unsigned char) (0.21*r+0.71*g+0.07*b);
  }
}

__global__ void gray2histogram (unsigned char *input, unsigned int *histo, int width, int height){

  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
  int t = threadIdx.x + threadIdx.y*blockDim.x;

  if (t < 256){
    histo_private[t] = 0;
  }

  __syncthreads();

  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  int ty = threadIdx.y + blockDim.y*blockIdx.y;

  if (tx < width && ty < height){
    int idx = ty*width + tx;
    atomicAdd( &(histo_private[input[idx]]), 1);
  }

  __syncthreads();

  if (t < 256){
    atomicAdd( &(histo[t]),histo_private[t]);
  }
}

// The input is the grey image, the output is the histogram which is modified into probability
__global__ void histo (unsigned char* input, unsigned int num_elements, unsigned int* histo, unsigned int num_bins)
{
  __shared__ unsigned int histo_small[256];
  unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
  
  //Privatized bins
  if (ix < num_bins){      // assume the threads number is larger than the number of bins(256)
    histo_small[ix] = 0u;
  }
  __syncthreads();
  
  if (ix < num_elements){
    atomicAdd(&(histo_small[input[ix]]),1);
  }
  __syncthreads();
  
  atomicAdd(&(histo[threadIdx.x]),histo_small[threadIdx.x]);
  __syncthreads();
}

__global__ void scan(unsigned int *input, float *output, int size,int len){

  __shared__ float T[2*128];
  
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

  for (int stride = 1; stride < 2*128; stride *=2){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if (index < 2*128 && (index-stride) >= 0) T[index] += T[index-stride];
  }
  
  for (int stride = 128/2; stride > 0; stride /=2){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2-1;
    if ((index+stride) < 2*128) T[index+stride] += T[index];
  }
  
  __syncthreads();
  if (2*blockIdx.x*blockDim.x+threadIdx.x<len){
    output[2*blockIdx.x*blockDim.x + threadIdx.x] = T[threadIdx.x]/1./size;
  } else{
    output[2*blockIdx.x*blockDim.x + threadIdx.x] = 0;
  }

  if (2*blockIdx.x*blockDim.x+blockDim.x+threadIdx.x<len){
    output[2*blockIdx.x*blockDim.x + blockDim.x + threadIdx.x] = T[blockDim.x+threadIdx.x]/1./size;
  } else{
    output[2*blockIdx.x*blockDim.x + blockDim.x + threadIdx.x] = 0;
  }
}


__global__ void correct_color(float* old_cdf, unsigned char* input, unsigned char* output,float minimum, float maximal, int size)
{
  unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
  if (ix<size){
    unsigned int index = input[ix];
    unsigned char new_val = (unsigned char) 255 * (old_cdf[index]-minimum)/(1.0-minimum);
    if (new_val > 255){
      new_val = 255;
    }
    output[ix] = new_val;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData_raw;
  unsigned char *deviceInputImageData;
  unsigned char *deviceGreyImageData;
  unsigned int* deviceHistogram;
  float* deviceHistogram_cdf;
  unsigned char* newdeviceOutput;
  float* minimum;
  float* deviceOutputImageData;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  
  // Allocate CPU memory
  minimum = (float*)malloc(sizeof(float));
  *minimum = 0.0;
  
  // Allocate cuda memory
  cudaMalloc((void **)&deviceInputImageData_raw, imageWidth * imageHeight*imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight *imageChannels* sizeof(unsigned char));
  cudaMalloc((void **)&deviceGreyImageData, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHistogram, 256 * sizeof(unsigned int));
  cudaMalloc((void **)&deviceHistogram_cdf, 256 * sizeof(float));
  cudaMalloc((void **)&newdeviceOutput, imageWidth * imageHeight *  imageChannels * sizeof(unsigned char));
  cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight *imageChannels* sizeof(float));
  
  // Copy memory from host to device
  cudaMemcpy(deviceInputImageData_raw, hostInputImageData, imageWidth * imageHeight * imageChannels*sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 gridDim(ceil(1.*imageChannels*imageHeight*imageWidth/256),1,1);
  dim3 blockDim(256,1,1);
  wbLog(TRACE, "The grid dimention is ",ceil(1.*imageChannels*imageHeight*imageWidth/256) ," block size is ",256);
  Cast_char<<<gridDim,blockDim>>>(deviceInputImageData_raw, deviceInputImageData, imageWidth * imageHeight*imageChannels);
  cudaDeviceSynchronize();
  
  dim3 gridDim1(ceil(1.*imageWidth/32), ceil(1.*imageWidth/32),1);
  dim3 blockDim1(32,32,1);
  wbLog(TRACE, "The grid dimention is ",ceil(1.*imageWidth/32), "x",ceil(1.*imageWidth/32)," block size is ",32,"*",32);
  Convert_Gray<<<gridDim1,blockDim1>>>(deviceInputImageData, deviceGreyImageData, imageHeight, imageWidth);
  cudaDeviceSynchronize(); 
  
  dim3 gridDim2(ceil(1.*imageHeight*imageWidth/256),1,1);
  dim3 blockDim2(256,1,1);
  //wbLog(TRACE, "The grid dimention is ",ceil(1.*imageHeight*imageWidth/256)," block size is ",256);
  //histo<<<gridDim2,blockDim2>>>(deviceGreyImageData,imageHeight*imageWidth, deviceHistogram, 256);
  ///cudaDeviceSynchronize();
  
  dim3 DimGrid = dim3(ceil(imageWidth/(1.0*32)),ceil(imageHeight/(1.0*32)),1);
  dim3 DimBlock = dim3(32,32,1);
  gray2histogram<<<DimGrid,DimBlock>>>(deviceGreyImageData,deviceHistogram,imageWidth,imageHeight);
  cudaDeviceSynchronize();
  
  dim3 GridDim3(ceil(256*1./2/128),1,1);
  dim3 BlockDim3(128,1,1);
  wbLog(TRACE, "The grid dimention is ",ceil(imageHeight*imageWidth*1./2/128)," block size is ",128);
  scan<<<GridDim3,BlockDim3>>>(deviceHistogram,deviceHistogram_cdf, imageHeight*imageWidth,256);
  cudaDeviceSynchronize();
  
  cudaMemcpy(minimum, deviceHistogram_cdf, 1 * sizeof(float),cudaMemcpyDeviceToHost);
  correct_color<<<gridDim,blockDim>>>(deviceHistogram_cdf, deviceInputImageData, newdeviceOutput,*minimum, 1,imageChannels* imageHeight*imageWidth);
  
  dim3 gridDim4(ceil(1.*imageChannels*imageHeight*imageWidth/256),1,1);
  dim3 blockDim4(256,1,1);
  Cast_float<<<gridDim4,blockDim4>>>(newdeviceOutput, deviceOutputImageData, imageWidth * imageHeight*imageChannels);
  cudaDeviceSynchronize();
  
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  
  wbSolution(args, outputImage);

  //@@ insert code here
  
  cudaFree(deviceInputImageData_raw);
  cudaFree(deviceInputImageData);
  cudaFree(deviceGreyImageData);
  cudaFree(deviceHistogram);
  cudaFree(deviceHistogram_cdf);
  cudaFree(newdeviceOutput);
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}

