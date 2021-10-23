#include "cuconv.h"
#include <cassert>
#include <string.h>

#define __CUDACC__

#include "cuda_texture_types.h"
#include "texture_fetch_functions.h"

typedef texture<float, cudaTextureType3D, cudaReadModeElementType> floatTex;

floatTex teximg(0, cudaFilterModePoint, cudaAddressModeBorder);

#define d_input_img(n,c,h,w) d_input_img[(n)*input_channel*input_w_padded*input_h_padded+(c)*input_w_padded*input_h_padded+(h)*input_w_padded+(w)]
#define input_img(n,c,h,w) input_img[(n)*input_channel*d->input_w*d->input_h+(c)*d->input_w*d->input_h+(h)*d->input_w+(w)]

#define d_output_img(i, j) d_output_img[(gemm_n_padded)*(i)+(j)]

#define KERNELX 1
#define KERNELY 128
#define REGTILESIZEX 3
#define REGTILESIZEY 4
#define TILEM (KERNELX*REGTILESIZEX)
#define TILEN (KERNELY*REGTILESIZEY)
#define TILEK KERNELX

#define D_FILTER_W 28
#define D_FILTER_H 3
__constant__ float d_filter[D_FILTER_H][D_FILTER_W];

//Image uses NCHW memory layout
__global__ void cuconv_im2col_kernel(int gemm_n_padded, int gemm_k_padded,
    float* d_output_img, int output_h, int output_w, int stride_h, int stride_w, int gemm_k)
{
    const int filter_h = 3;
    const int filter_w = 3;
    const int input_channel = 3;
    const int pad_h = 1;
    const int pad_w = 1;

    __shared__ float sm_inputimg[TILEK][TILEN];
    int Ci = TILEM * blockIdx.x + threadIdx.x * REGTILESIZEX;
    int Cj = TILEN * blockIdx.y + threadIdx.y * REGTILESIZEY;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float4 vec_c0_03 = make_float4(0.0, 0.0, 0.0, 0.0);
    float4 vec_c1_03 = make_float4(0.0, 0.0, 0.0, 0.0);
    float4 vec_c2_03 = make_float4(0.0, 0.0, 0.0, 0.0);

    float reg_a0, reg_a1, reg_a2;
    float4 vec_b0_3;

    int ic, h_res, fh, fw, oh, ow, ih, iw;

    for (int po = 0; po < gemm_k; po += TILEK)
    {
        ic = (po + tx) / (filter_h * filter_w);
        h_res = (po + tx) % (filter_h * filter_w);
        fh = h_res / filter_w;
        fw = h_res % filter_w;
        oh = Cj / output_w;
        ow = Cj % output_w;

        ih = oh * stride_h + fh - pad_h;
        iw = ow * stride_w + fw - pad_w;

        sm_inputimg[tx][ty * REGTILESIZEY + 0] = tex3D(teximg, iw, ih, ic);

        oh = (Cj + 1) / output_w;
        ow = (Cj + 1) % output_w;

        ih = oh * stride_h + fh - pad_h;
        iw = ow * stride_w + fw - pad_w;

        sm_inputimg[tx][ty * REGTILESIZEY + 1] = tex3D(teximg, iw, ih, ic);

        oh = (Cj + 2) / output_w;
        ow = (Cj + 2) % output_w;

        ih = oh * stride_h + fh - pad_h;
        iw = ow * stride_w + fw - pad_w;

        sm_inputimg[tx][ty * REGTILESIZEY + 2] = tex3D(teximg, iw, ih, ic);

        oh = (Cj + 3) / output_w;
        ow = (Cj + 3) % output_w;

        ih = oh * stride_h + fh - pad_h;
        iw = ow * stride_w + fw - pad_w;

        sm_inputimg[tx][ty * REGTILESIZEY + 3] = tex3D(teximg, iw, ih, ic);

        __syncthreads();
        for (int pi = 0; pi < TILEK; pi++)
        {
            reg_a0 = d_filter[Ci + 0][po + pi];
            reg_a1 = d_filter[Ci + 1][po + pi];
            reg_a2 = d_filter[Ci + 2][po + pi];

            vec_b0_3 = *reinterpret_cast<float4*>(&sm_inputimg[pi][ty * REGTILESIZEY + 0]);

            vec_c0_03.x += reg_a0 * vec_b0_3.x; vec_c0_03.y += reg_a0 * vec_b0_3.y; vec_c0_03.z += reg_a0 * vec_b0_3.z; vec_c0_03.w += reg_a0 * vec_b0_3.w;
            vec_c1_03.x += reg_a1 * vec_b0_3.x; vec_c1_03.y += reg_a1 * vec_b0_3.y; vec_c1_03.z += reg_a1 * vec_b0_3.z; vec_c1_03.w += reg_a1 * vec_b0_3.w;
            vec_c2_03.x += reg_a2 * vec_b0_3.x; vec_c2_03.y += reg_a2 * vec_b0_3.y; vec_c2_03.z += reg_a2 * vec_b0_3.z; vec_c2_03.w += reg_a2 * vec_b0_3.w;

        }
        __syncthreads();
    }
    *reinterpret_cast<float4*>(&d_output_img(Ci + 0, Cj + 0)) = vec_c0_03;
    *reinterpret_cast<float4*>(&d_output_img(Ci + 1, Cj + 0)) = vec_c1_03;
    *reinterpret_cast<float4*>(&d_output_img(Ci + 2, Cj + 0)) = vec_c2_03;
}


float cuconv_im2col(cuconv_descriptor* d, float* input_img, float* filter, float* output_img)
{
    assert(d->output_h == (d->input_h + d->pad_h * 2 - d->filter_h) / d->stride_h + 1);
    assert(d->output_w == (d->input_w + d->pad_w * 2 - d->filter_w) / d->stride_w + 1);
    assert(d->N_batch == 1);
    assert(d->input_c == 3);
    assert(d->output_c == 3);
    assert(d->filter_h == 3);
    assert(d->filter_w == 3);
    assert(d->pad_h == 1);
    assert(d->pad_w == 1);

    int input_channel = d->input_c;

    float* d_output_img;
    cudaArray* d_input_img;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    int GEMM_M = d->output_c;
    int GEMM_K = d->input_c * d->filter_h * d->filter_w;
    int GEMM_N = d->output_h * d->output_w;

    int D_GEMM_M = ((GEMM_M - 1) / TILEM + 1) * TILEM;
    int D_GEMM_K = ((GEMM_K - 1) / TILEK + 1) * TILEK;
    int D_GEMM_N = ((GEMM_N - 1) / TILEN + 1) * TILEN;

    size_t pitch_n;


    checkCudaErrors(cudaMalloc(&d_input_img, sizeof(float) *
        d->N_batch *
        d->input_c *
        d->input_h *
        d->input_w));

    const cudaExtent inputSize = make_cudaExtent(d->input_w, d->input_h, d->input_c);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMalloc3DArray(&d_input_img, &channelDesc, inputSize));

    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr((void*)input_img, inputSize.width * sizeof(float), inputSize.width, inputSize.height);
    copyParams.dstArray = d_input_img;
    copyParams.extent = inputSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));
    checkCudaErrors(cudaBindTextureToArray(&teximg, d_input_img, &channelDesc));


    checkCudaErrors(cudaMallocPitch(&d_output_img, &pitch_n, sizeof(float) * D_GEMM_N, D_GEMM_M));
    checkCudaErrors(cudaMemset(d_output_img, 0, pitch_n * D_GEMM_M));
    int d_ldc = pitch_n / sizeof(float);


    float* filter_tmp = (float*)malloc(sizeof(float) * D_FILTER_H * D_FILTER_W);
    memset(filter_tmp, 0, sizeof(float) * D_FILTER_H * D_FILTER_W);
    for (int oc = 0; oc < d->output_c; oc++) {
        memcpy(filter_tmp + D_FILTER_W * oc, filter + d->filter_h * d->filter_w * d->input_c * oc, sizeof(float) * D_FILTER_W);
    }
    checkCudaErrors(cudaMemcpyToSymbol(d_filter, filter_tmp, sizeof(float) * D_FILTER_H * D_FILTER_W, 0, cudaMemcpyHostToDevice));

    dim3 dim_block((D_GEMM_M - 1) / TILEM + 1, (D_GEMM_N - 1) / TILEN + 1, 1),
        dim_thread(KERNELX, KERNELY, 1);

    cudaEventRecord(start, 0);
    cuconv_im2col_kernel << <dim_block, dim_thread >> > (D_GEMM_N, D_GEMM_K,
        d_output_img, d->output_h, d->output_w, d->stride_h, d->stride_w, GEMM_K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    checkCudaErrors(cudaMemcpy2D(output_img, GEMM_N * sizeof(float), d_output_img, d_ldc * sizeof(float),
        GEMM_N * sizeof(float), GEMM_M, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaUnbindTexture(&teximg));
    checkCudaErrors(cudaFree(d_output_img));
    free(filter_tmp);

    return elapsedTime;
}
