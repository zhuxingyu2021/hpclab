#ifndef _CU_CONV_H_
#define _CU_CONV_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct{
    const int N_batch = 1;
    const int input_c = 3; //number of input channels
    int input_h; //height of input image
    int input_w; //width of input image

    int output_c = 3; //number of output channels
    const int filter_h = 3; //height of filter
    const int filter_w = 3; //width of filter

    int stride_h;
    int stride_w;
    const int pad_h = 1;
    const int pad_w = 1;

    int output_h; //height of output image
    int output_w; //width of output image
}cuconv_descriptor;

//Image uses NCHW memory layout
float cuconv_naive_filter_3x3x3x3(cuconv_descriptor* d, float* input_img,float* filter, float* output_img);

float cuconv_cudnn(cuconv_descriptor* d, float* input_img, float* filter, float* output_img);

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