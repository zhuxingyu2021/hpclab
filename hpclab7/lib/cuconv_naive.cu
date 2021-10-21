#include "cuconv.h"
#include <cassert>

#define KERNEL_SIZE 16

#define d_input_img(n,c,h,w) d_input_img[(n)*input_channel*input_w_padded*input_h_padded+(c)*input_w_padded*input_h_padded+(h)*input_w_padded+(w)]
#define input_img(n,c,h,w) input_img[(n)*input_channel*d->input_w*d->input_h+(c)*d->input_w*d->input_h+(h)*d->input_w+(w)]
#define d_output_img(n,c,h,w) d_output_img[(n)*output_channel*output_w*output_h+(c)*output_w*output_h+(h)*output_w+(w)]
#define d_filter(co,ci,h,w) d_filter[(co)*input_channel*filter_h*filter_w+(ci)*filter_h*filter_w+(h)*filter_w+(w)]

//Image uses NCHW memory layout
__global__ void cuconv_naive_kernel(float* d_input_img, int input_h_padded, int input_w_padded,
    float* d_output_img, int output_h, int output_w,
    float* d_filter, int filter_h, int filter_w, 
    int input_channel, int output_channel,
    int stride_h, int stride_w)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int start_w = j * stride_w;
    int start_h = i * stride_h;

    if (i < output_h && j < output_w) {
        for (int oc = 0; oc < output_channel; oc++)
        {
            float v = 0;
            for (int ic = 0; ic < input_channel; ic++)
            {
                for (int fh = 0; fh < filter_h; fh++)
                {
                    for (int fw = 0; fw < filter_w; fw++)
                    {
                        v += d_input_img(0, ic, start_h + fh, start_w + fw) *
                            d_filter(oc, ic, fh, fw);
                    }
                }
            }
            d_output_img(0, oc, i, j) = v;
        }
    }
}

float cuconv_naive(cuconv_descriptor* d, float* input_img, float* filter, float* output_img)
{
    assert(d->output_h == (d->input_h + d->pad_h * 2 - d->filter_h) / d->stride_h + 1);
    assert(d->output_w == (d->input_w + d->pad_w * 2 - d->filter_w) / d->stride_w + 1);
    assert(d->N_batch == 1);
    int input_channel = d -> input_c;

    float *d_input_img, *d_filter, *d_output_img;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaMalloc(&d_input_img, sizeof(float) *
        d->N_batch *
        d->input_c *
        (d->input_h + 2 * d->pad_h) *
        (d->input_w + 2 * d->pad_w)));

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

    checkCudaErrors(cudaMemset(d_input_img, 0, sizeof(float) *
        d->N_batch *
        d->input_c *
        (d->input_h + 2 * d->pad_h) *
        (d->input_w + 2 * d->pad_w)));
    
    checkCudaErrors(cudaMemset(d_output_img, 0, sizeof(float) *
        d->N_batch *
        d->output_c *
        d->output_h *
        d->output_w));

    int input_w_padded = d->input_w + 2 * d->pad_w;
    int input_h_padded = d->input_h + 2 * d->pad_h;
    for(int ic = 0; ic < d->input_c; ic++){
        checkCudaErrors(cudaMemcpy2D(&d_input_img(0, ic, d->pad_h, d->pad_w), input_w_padded * sizeof(float),
            &input_img(0, ic, 0, 0), d->input_w * sizeof(float), d->input_w * sizeof(float), 
            d->input_h, cudaMemcpyHostToDevice));
    }

    checkCudaErrors(cudaMemcpy(d_filter, filter, sizeof(float) *
        d->output_c *
        d->input_c *
        d->filter_h *
        d->filter_w, cudaMemcpyHostToDevice));
    
    dim3 dim_block((d->output_h - 1) / (KERNEL_SIZE) + 1, (d->output_w - 1) / (KERNEL_SIZE) + 1, 1),
        dim_thread(KERNEL_SIZE, KERNEL_SIZE, 1);

    cudaEventRecord(start, 0);
    cuconv_naive_kernel << <dim_block, dim_thread >> > (d_input_img, input_h_padded, input_w_padded,
        d_output_img, d->output_h, d->output_w, d_filter, d->filter_h, d->filter_w, d->input_c, d->output_c,
        d->stride_h, d->stride_w);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    checkCudaErrors(cudaMemcpy(output_img, d_output_img, sizeof(float) *
        d->N_batch *
        d->output_c *
        d->output_h *
        d->output_w, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_input_img));
    checkCudaErrors(cudaFree(d_output_img));
    checkCudaErrors(cudaFree(d_filter));

    return elapsedTime;
}
