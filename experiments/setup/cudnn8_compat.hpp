// MatConvNet beta25 compatibility for APIs removed by cuDNN 8.
// Algorithm selection uses cuDNN's v7 heuristic API and honors workspace limits.
// No model weights or convolution definitions are changed.
#ifndef THESIS_CUDNN8_COMPAT_HPP
#define THESIS_CUDNN8_COMPAT_HPP
#include <cudnn.h>
#if CUDNN_MAJOR >= 8
typedef enum {
  CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0,
  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1,
  CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2
} cudnnConvolutionFwdPreference_t;
typedef enum {
  CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0,
  CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1,
  CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2
} cudnnConvolutionBwdFilterPreference_t;
typedef enum {
  CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0,
  CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1,
  CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2
} cudnnConvolutionBwdDataPreference_t;

inline bool thesis_cudnn_workspace_fits(int preference, size_t bytes, size_t limit) {
  return preference == 1 || (preference == 0 ? bytes == 0 : bytes <= limit);
}

inline cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle_t handle, cudnnTensorDescriptor_t x, cudnnFilterDescriptor_t w,
    cudnnConvolutionDescriptor_t conv, cudnnTensorDescriptor_t y,
    cudnnConvolutionFwdPreference_t preference, size_t limit,
    cudnnConvolutionFwdAlgo_t* algorithm) {
  int count = 0, returned = 0;
  cudnnStatus_t status = cudnnGetConvolutionForwardAlgorithmMaxCount(handle,&count);
  if (status != CUDNN_STATUS_SUCCESS) return status;
  cudnnConvolutionFwdAlgoPerf_t* results = new cudnnConvolutionFwdAlgoPerf_t[count];
  status = cudnnGetConvolutionForwardAlgorithm_v7(handle,x,w,conv,y,count,&returned,results);
  if (status == CUDNN_STATUS_SUCCESS) {
    status = CUDNN_STATUS_NOT_SUPPORTED;
    for (int i = 0; i < returned; ++i) {
      if (results[i].status == CUDNN_STATUS_SUCCESS &&
          thesis_cudnn_workspace_fits((int)preference,results[i].memory,limit)) {
        *algorithm = results[i].algo; status = CUDNN_STATUS_SUCCESS; break;
      }
    }
  }
  delete[] results;
  return status;
}

inline cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t dy,
    cudnnConvolutionDescriptor_t conv, cudnnFilterDescriptor_t dw,
    cudnnConvolutionBwdFilterPreference_t preference, size_t limit,
    cudnnConvolutionBwdFilterAlgo_t* algorithm) {
  int count = 0, returned = 0;
  cudnnStatus_t status = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle,&count);
  if (status != CUDNN_STATUS_SUCCESS) return status;
  cudnnConvolutionBwdFilterAlgoPerf_t* results = new cudnnConvolutionBwdFilterAlgoPerf_t[count];
  status = cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle,x,dy,conv,dw,count,&returned,results);
  if (status == CUDNN_STATUS_SUCCESS) {
    status = CUDNN_STATUS_NOT_SUPPORTED;
    for (int i = 0; i < returned; ++i) {
      if (results[i].status == CUDNN_STATUS_SUCCESS &&
          thesis_cudnn_workspace_fits((int)preference,results[i].memory,limit)) {
        *algorithm = results[i].algo; status = CUDNN_STATUS_SUCCESS; break;
      }
    }
  }
  delete[] results;
  return status;
}

inline cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, cudnnFilterDescriptor_t w, cudnnTensorDescriptor_t dy,
    cudnnConvolutionDescriptor_t conv, cudnnTensorDescriptor_t dx,
    cudnnConvolutionBwdDataPreference_t preference, size_t limit,
    cudnnConvolutionBwdDataAlgo_t* algorithm) {
  int count = 0, returned = 0;
  cudnnStatus_t status = cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle,&count);
  if (status != CUDNN_STATUS_SUCCESS) return status;
  cudnnConvolutionBwdDataAlgoPerf_t* results = new cudnnConvolutionBwdDataAlgoPerf_t[count];
  status = cudnnGetConvolutionBackwardDataAlgorithm_v7(handle,w,dy,conv,dx,count,&returned,results);
  if (status == CUDNN_STATUS_SUCCESS) {
    status = CUDNN_STATUS_NOT_SUPPORTED;
    for (int i = 0; i < returned; ++i) {
      if (results[i].status == CUDNN_STATUS_SUCCESS &&
          thesis_cudnn_workspace_fits((int)preference,results[i].memory,limit)) {
        *algorithm = results[i].algo; status = CUDNN_STATUS_SUCCESS; break;
      }
    }
  }
  delete[] results;
  return status;
}
#endif
#endif
