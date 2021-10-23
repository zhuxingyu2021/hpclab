#ifndef _CU_CONV_H_
#define _CU_CONV_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct{
    const int N_batch = 1;
    int input_c; //number of input channels
    int input_h; //height of input image
    int input_w; //width of input image

    int output_c; //number of output channels
    int filter_h; //height of filter
    int filter_w; //width of filter

    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;

    int output_h; //height of output image
    int output_w; //width of output image
}cuconv_descriptor;

//Image uses NCHW memory layout
float cuconv_naive(cuconv_descriptor* d, float* input_img,float* filter, float* output_img);
float cuconv_cudnn(cuconv_descriptor* d, float* input_img, float* filter, float* output_img);
float cuconv_im2col(cuconv_descriptor* d, float* input_img, float* filter, float* output_img);

void cuconv_getoutputsize(cuconv_descriptor* d);

void random_initialize(float* tensor, int size);

void output_tensor_dim4_to_file(const char* filename, float* tensor, int dim1, int dim2, int dim3, int dim4);

#define checkCudaErrors( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);

#endif