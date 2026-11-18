// Standalone installation check; does not change MatConvNet or its models.
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

__global__ void fill(float* data, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) data[i] = 1.0f;
}

int main() {
  cudaDeviceProp prop;
  cudaError_t error = cudaGetDeviceProperties(&prop, 0);
  if (error != cudaSuccess) {
    std::printf("CUDA_DEVICE_ERROR %s\n", cudaGetErrorString(error)); return 1;
  }
  std::printf("GPU %s compute=%d.%d\n", prop.name, prop.major, prop.minor);
  std::printf("CUDA_BUILD_VERSION %d CUDNN_RUNTIME_VERSION %llu\n",
              CUDART_VERSION, (unsigned long long)cudnnGetVersion());
  float *input = 0, *filter = 0, *output = 0;
  cudaMalloc((void**)&input, 64*sizeof(float));
  cudaMalloc((void**)&filter, 9*sizeof(float));
  cudaMalloc((void**)&output, 64*sizeof(float));
  fill<<<1,64>>>(input,64); fill<<<1,64>>>(filter,9);
  error = cudaDeviceSynchronize();
  std::printf("CUDA_PTX_KERNEL %s\n", cudaGetErrorString(error));
  if (error != cudaSuccess) return 2;
  float alpha = 1, beta = 0, result = 0;
  cublasHandle_t blas;
  cublasStatus_t bs = cublasCreate(&blas);
  if (bs == CUBLAS_STATUS_SUCCESS)
    bs = cublasSgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,1,1,9,
                    &alpha,input,1,filter,9,&beta,output,1);
  error = cudaDeviceSynchronize();
  cudaMemcpy(&result,output,sizeof(float),cudaMemcpyDeviceToHost);
  bool blasPass = bs == CUBLAS_STATUS_SUCCESS && error == cudaSuccess && result == 9;
  std::printf("CUBLAS_GEMM status=%d cuda=%s result=%g %s\n",
              (int)bs,cudaGetErrorString(error),result,blasPass ? "PASS" : "FAIL");
  cudnnHandle_t handle;
  cudnnStatus_t ds = cudnnCreate(&handle);
  if (ds != CUDNN_STATUS_SUCCESS) {
    std::printf("CUDNN_CREATE %s\n",cudnnGetErrorString(ds)); return 3;
  }
  cudnnTensorDescriptor_t inDesc, outDesc;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnCreateTensorDescriptor(&inDesc); cudnnCreateTensorDescriptor(&outDesc);
  cudnnCreateFilterDescriptor(&filterDesc); cudnnCreateConvolutionDescriptor(&convDesc);
  cudnnSetTensor4dDescriptor(inDesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,1,8,8);
  cudnnSetTensor4dDescriptor(outDesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,1,8,8);
  cudnnSetFilter4dDescriptor(filterDesc,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,1,1,3,3);
  cudnnSetConvolution2dDescriptor(convDesc,1,1,1,1,1,1,
                                 CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
  ds = cudnnConvolutionForward(handle,&alpha,inDesc,input,filterDesc,filter,convDesc,
                                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                0,0,&beta,outDesc,output);
  error = cudaDeviceSynchronize();
  cudaMemcpy(&result,output+4*8+4,sizeof(float),cudaMemcpyDeviceToHost);
  bool cudnnPass = ds == CUDNN_STATUS_SUCCESS && error == cudaSuccess && result == 9;
  std::printf("CUDNN_CONV status=%s cuda=%s center=%g %s\n",
              cudnnGetErrorString(ds),cudaGetErrorString(error),result,
              cudnnPass ? "PASS" : "FAIL");
  cudnnDestroyTensorDescriptor(inDesc); cudnnDestroyTensorDescriptor(outDesc);
  cudnnDestroyFilterDescriptor(filterDesc); cudnnDestroyConvolutionDescriptor(convDesc);
  cudnnDestroy(handle); cublasDestroy(blas);
  cudaFree(input); cudaFree(filter); cudaFree(output);
  return blasPass && cudnnPass ? 0 : 4;
}
