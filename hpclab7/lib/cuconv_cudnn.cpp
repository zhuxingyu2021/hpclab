#include "cuconv.h"
#include <cudnn.h>

#define checkCudnnErrors( a ) do { \
    cudnnStatus_t cudnn_status; \
    if (CUDNN_STATUS_SUCCESS != (cudnn_status = a)) { \
        fprintf(stderr, "Cudnn runtime error in line %d of file %s \
        : %s \n", __LINE__, __FILE__, cudnnGetErrorString(cudnn_status) ); \
        exit(EXIT_FAILURE); \
        } \
    } while(0);

float cuconv_cudnn(cuconv_descriptor* d, float* input_img, float* filter, float* output_img)
{
    float* d_input_img, * d_filter, * d_output_img;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

	cudnnHandle_t handle;
	checkCudnnErrors(cudnnCreate(&handle));
    
    cudnnTensorDescriptor_t input_img_descriptor;
    checkCudnnErrors(cudnnCreateTensorDescriptor(&input_img_descriptor));
    checkCudnnErrors(cudnnSetTensor4dDescriptor(input_img_descriptor,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        d->N_batch, d->input_c, d->input_h, d->input_w));

    cudnnFilterDescriptor_t filter_descriptor;
    checkCudnnErrors(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCudnnErrors(cudnnSetFilter4dDescriptor(filter_descriptor,
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        d->output_c, d->input_c, d->filter_h, d->filter_w));

    cudnnConvolutionDescriptor_t conv_descriptor;
    checkCudnnErrors(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    checkCudnnErrors(cudnnSetConvolution2dDescriptor(conv_descriptor,
        d->pad_h, d->pad_w, d->stride_h, d->stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnTensorDescriptor_t output_img_descriptor;
    checkCudnnErrors(cudnnCreateTensorDescriptor(&output_img_descriptor));
    checkCudnnErrors(cudnnSetTensor4dDescriptor(output_img_descriptor,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        d->N_batch, d->output_c, d->output_h, d->output_w));

    size_t workspace_size = 0;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(handle,
        input_img_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_img_descriptor,
        algo,
        &workspace_size));

    void* workspace = nullptr;
    checkCudaErrors(cudaMalloc(&workspace, workspace_size));

    checkCudaErrors(cudaMalloc(&d_input_img, sizeof(float) *
        d->N_batch *
        d->input_c *
        d->input_h *
        d->input_w));

    checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) *
        d->output_c *
        d->input_c *
        d->filter_h *
        d->filter_w));

    checkCudaErrors(cudaMalloc(&d_output_img, sizeof(float) *
        d->N_batch *
        d->output_c *
        d->output_h *
        d->output_w));

    checkCudaErrors(cudaMemset(d_output_img, 0, sizeof(float) *
        d->N_batch *
        d->output_c *
        d->output_h *
        d->output_w));

    checkCudaErrors(cudaMemcpy(d_input_img, input_img, sizeof(float) *
        d->N_batch *
        d->input_c *
        d->input_h *
        d->input_w, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_filter, filter, sizeof(float) *
        d->output_c *
        d->input_c *
        d->filter_h *
        d->filter_w, cudaMemcpyHostToDevice));
    
    float alpha = 1.0, beta = 0.0;

    cudaEventRecord(start, 0);
    checkCudnnErrors(cudnnConvolutionForward(handle, &alpha,
        input_img_descriptor, d_input_img,
        filter_descriptor, d_filter,
        conv_descriptor, algo, workspace, workspace_size, &beta,
        output_img_descriptor, d_output_img));
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    checkCudaErrors(cudaMemcpy(output_img, d_output_img, sizeof(float) *
        d->N_batch *
        d->output_c *
        d->output_h *
        d->output_w, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(workspace));

    checkCudnnErrors(cudnnDestroyTensorDescriptor(input_img_descriptor));
    checkCudnnErrors(cudnnDestroyTensorDescriptor(output_img_descriptor));
    checkCudnnErrors(cudnnDestroyFilterDescriptor(filter_descriptor));
    checkCudnnErrors(cudnnDestroyConvolutionDescriptor(conv_descriptor));

    checkCudnnErrors(cudnnDestroy(handle));

    return elapsedTime;
}
