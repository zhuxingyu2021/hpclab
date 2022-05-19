#include "GEMM.h"
#include "common_define.h"
#include <immintrin.h>
#include <string.h>
#include <iostream>
using namespace std;

#define KERNEL_M 4
#define KERNEL_N 24

#define MIN(i,j) ((i>j)?(j):(i))

const int BK = 256;
const int BN = 240;

// 4*24的kernel
inline void gemm_kernel_4x24(int k,
                             float* packed_A, int lda,
                             float* packed_B, int ldb,
                             float* C, int ldc)
{
    __m256 c0_0, c0_1, c0_2;
    __m256 c1_0, c1_1, c1_2;
    __m256 c2_0, c2_1, c2_2;
    __m256 c3_0, c3_1, c3_2;
    __m256 a, b0, b1, b2;

    float* p_packed_A = packed_A;
    float* p_packed_B = packed_B;

    c0_0 = _mm256_load_ps(&C(0, 0)); c0_1 = _mm256_load_ps(&C(0, 8)); c0_2 = _mm256_load_ps(&C(0, 16));
    c1_0 = _mm256_load_ps(&C(1, 0)); c1_1 = _mm256_load_ps(&C(1, 8)); c1_2 = _mm256_load_ps(&C(1, 16));
    c2_0 = _mm256_load_ps(&C(2, 0)); c2_1 = _mm256_load_ps(&C(2, 8)); c2_2 = _mm256_load_ps(&C(2, 16));
    c3_0 = _mm256_load_ps(&C(3, 0)); c3_1 = _mm256_load_ps(&C(3, 8)); c3_2 = _mm256_load_ps(&C(3, 16));

    for (int p = 0; p < k; p++) {
        b0 = _mm256_load_ps(p_packed_B + 0);
        b1 = _mm256_load_ps(p_packed_B + 8);
        b2 = _mm256_load_ps(p_packed_B + 16);
        p_packed_B += 24;

        a = _mm256_set1_ps(*(p_packed_A + 0));
        c0_0 = _mm256_fmadd_ps(a, b0, c0_0);
        c0_1 = _mm256_fmadd_ps(a, b1, c0_1);
        c0_2 = _mm256_fmadd_ps(a, b2, c0_2);

        a = _mm256_set1_ps(*(p_packed_A + 1));
        c1_0 = _mm256_fmadd_ps(a, b0, c1_0);
        c1_1 = _mm256_fmadd_ps(a, b1, c1_1);
        c1_2 = _mm256_fmadd_ps(a, b2, c1_2);

        a = _mm256_set1_ps(*(p_packed_A + 2));
        c2_0 = _mm256_fmadd_ps(a, b0, c2_0);
        c2_1 = _mm256_fmadd_ps(a, b1, c2_1);
        c2_2 = _mm256_fmadd_ps(a, b2, c2_2);

        a = _mm256_set1_ps(*(p_packed_A + 3));
        c3_0 = _mm256_fmadd_ps(a, b0, c3_0);
        c3_1 = _mm256_fmadd_ps(a, b1, c3_1);
        c3_2 = _mm256_fmadd_ps(a, b2, c3_2);

        p_packed_A += 4;
    }

    _mm256_store_ps(&C(0, 0), c0_0); _mm256_store_ps(&C(0, 8), c0_1); _mm256_store_ps(&C(0, 16), c0_2);
    _mm256_store_ps(&C(1, 0), c1_0); _mm256_store_ps(&C(1, 8), c1_1); _mm256_store_ps(&C(1, 16), c1_2);
    _mm256_store_ps(&C(2, 0), c2_0); _mm256_store_ps(&C(2, 8), c2_1); _mm256_store_ps(&C(2, 16), c2_2);
    _mm256_store_ps(&C(3, 0), c3_0); _mm256_store_ps(&C(3, 8), c3_1); _mm256_store_ps(&C(3, 16), c3_2);
}

// 4*24kernel（在边缘带有padding的情况）
inline void gemm_kernel_4x24_with_padding(int k,
                                          float* packed_A, int lda,
                                          float* packed_B, int ldb,
                                          float* C, int ldc,
                                          int kernelm,int kerneln)
{
    float kernel_buf[KERNEL_M][KERNEL_N];
    __m256 c0_0, c0_1, c0_2;
    __m256 c1_0, c1_1, c1_2;
    __m256 c2_0, c2_1, c2_2;
    __m256 c3_0, c3_1, c3_2;
    __m256 a, b0, b1, b2;

    float* p_packed_A = packed_A;
    float* p_packed_B = packed_B;

    c0_0 = _mm256_setzero_ps(); c0_1 = _mm256_setzero_ps(); c0_2 = _mm256_setzero_ps();
    c1_0 = _mm256_setzero_ps(); c1_1 = _mm256_setzero_ps(); c1_2 = _mm256_setzero_ps();
    c2_0 = _mm256_setzero_ps(); c2_1 = _mm256_setzero_ps(); c2_2 = _mm256_setzero_ps();
    c3_0 = _mm256_setzero_ps(); c3_1 = _mm256_setzero_ps(); c3_2 = _mm256_setzero_ps();

    for (int p = 0; p < k; p++) {
        b0 = _mm256_load_ps(p_packed_B + 0);
        b1 = _mm256_load_ps(p_packed_B + 8);
        b2 = _mm256_load_ps(p_packed_B + 16);
        p_packed_B += 24;

        a = _mm256_set1_ps(*(p_packed_A + 0));
        c0_0 = _mm256_fmadd_ps(a, b0, c0_0);
        c0_1 = _mm256_fmadd_ps(a, b1, c0_1);
        c0_2 = _mm256_fmadd_ps(a, b2, c0_2);

        a = _mm256_set1_ps(*(p_packed_A + 1));
        c1_0 = _mm256_fmadd_ps(a, b0, c1_0);
        c1_1 = _mm256_fmadd_ps(a, b1, c1_1);
        c1_2 = _mm256_fmadd_ps(a, b2, c1_2);

        a = _mm256_set1_ps(*(p_packed_A + 2));
        c2_0 = _mm256_fmadd_ps(a, b0, c2_0);
        c2_1 = _mm256_fmadd_ps(a, b1, c2_1);
        c2_2 = _mm256_fmadd_ps(a, b2, c2_2);

        a = _mm256_set1_ps(*(p_packed_A + 3));
        c3_0 = _mm256_fmadd_ps(a, b0, c3_0);
        c3_1 = _mm256_fmadd_ps(a, b1, c3_1);
        c3_2 = _mm256_fmadd_ps(a, b2, c3_2);

        p_packed_A += 4;
    }

    _mm256_store_ps(&kernel_buf[0][0], c0_0); _mm256_store_ps(&kernel_buf[0][8], c0_1); _mm256_store_ps(&kernel_buf[0][16], c0_2);
    _mm256_store_ps(&kernel_buf[1][0], c1_0); _mm256_store_ps(&kernel_buf[1][8], c1_1); _mm256_store_ps(&kernel_buf[1][16], c1_2);
    _mm256_store_ps(&kernel_buf[2][0], c2_0); _mm256_store_ps(&kernel_buf[2][8], c2_1); _mm256_store_ps(&kernel_buf[2][16], c2_2);
    _mm256_store_ps(&kernel_buf[3][0], c3_0); _mm256_store_ps(&kernel_buf[3][8], c3_1); _mm256_store_ps(&kernel_buf[3][16], c3_2);

    for(int i = 0; i < kernelm; i++){
        for(int j = 0; j < kerneln; j++){
            C(i,j) += kernel_buf[i][j];
        }
    }

}

inline void packA(int k, float* packed_A, float* A, int lda) {
    float* p_packed_A = packed_A;

    float* p_a0 = &A(0, 0);
    float* p_a1 = &A(1, 0);
    float* p_a2 = &A(2, 0);
    float* p_a3 = &A(3, 0);

    for (int p = 0; p < k; p++) {
        *p_packed_A = *p_a0++;
        *(p_packed_A + 1) = *p_a1++;
        *(p_packed_A + 2) = *p_a2++;
        *(p_packed_A + 3) = *p_a3++;
        p_packed_A += 4;
    }
}


inline void packA_with_padding(int k, float* packed_A, float* A, int lda, int psa) {
    float* p_packed_A = packed_A;

    float* p_a[4];
    for(int i = 0; i < psa; i++){
        p_a[i] = &A(i, 0);
    }

    for (int p = 0; p < k; p++) {
        for(int i = 0; i < 4; i++){
            if(i >= psa) *p_packed_A = 0;
            else *p_packed_A = *p_a[i]++;
            p_packed_A++;
        }
    }
}

inline void packB(int k,float* packed_B,float* B,int ldb) {
    __m256 tmp1, tmp2, tmp3;
    float* p_packed_B = packed_B;

    for (int p = 0; p < k; p++) {
        tmp1 = _mm256_load_ps(&B(p, 0));
        tmp2 = _mm256_load_ps(&B(p, 8));
        tmp3 = _mm256_load_ps(&B(p, 16));
        _mm256_store_ps(p_packed_B, tmp1);
        _mm256_store_ps(p_packed_B + 8, tmp2);
        _mm256_store_ps(p_packed_B + 16, tmp3);
        p_packed_B += 24;
    }
}

inline void packB_with_padding(int k,float* packed_B,float* B,int ldb,int psb) {
    float* p_packed_B = packed_B;
    float* p_B;

    for (int p = 0; p < k; p++) {
        p_B = &B(p, 0);

        for(int i = 0; i < 24; i++){
            if(i >= psb) *p_packed_B= 0;
            else *p_packed_B = *(p_B + i);
            p_packed_B++;
        }
    }
}

void sgemm_fast(int k, int m, int n,
                float* A, int lda,
                float* B, int ldb,
                float* C, int ldc, int n_threads)
{
    int BMx = ((2048 / n_threads) / 4) * 4;
    if(m < BMx * n_threads){
        BMx = ((m / n_threads)/4) * 4;
        if(!BMx) BMx = 4;
    }
    const int BM = BMx;

    int io, ii, jo, ji, p;

#pragma omp parallel for private(io, ii, jo, ji, p)  num_threads(n_threads) SCHEDULE
    for (io = 0; io < m; io += BM) {
        int BMsz = MIN(m - io, BM);
        int raw_BMsz = BMsz;
        if(BMsz % KERNEL_M != 0)  BMsz = (BMsz / KERNEL_M) * KERNEL_M + KERNEL_M;
        for (p = 0; p < k; p += BK) {
            int BKsz = MIN(k - p, BK);
            float* packed_A = (float*)aligned_malloc(sizeof(float) * BMsz * BKsz, GEMM_CACHELINE_SIZE);
            for (jo = 0; jo < n; jo += BN) {
                int BNsz = MIN(n - jo, BN);
                int raw_BNsz = BNsz;
                if(BNsz % KERNEL_N != 0)  BNsz = (BNsz / KERNEL_N) * KERNEL_N + KERNEL_N;
                float* packed_B = (float*)aligned_malloc(sizeof(float) * BKsz * BNsz, GEMM_CACHELINE_SIZE);
                for (ii = 0; ii < BMsz; ii += KERNEL_M) {
                    if (jo == 0) {
                        if(raw_BMsz - ii < KERNEL_M) {
                            packA_with_padding(BKsz, packed_A + BKsz * ii, &A(io + ii, p), lda, raw_BMsz - ii);
                        }
                        else {
                            packA(BKsz, packed_A + BKsz * ii, &A(io + ii, p), lda);
                        }
                    }
                    for (ji = 0; ji < BNsz; ji += KERNEL_N) {
                        if (ii == 0){
                            if(raw_BNsz - ji < KERNEL_N) {
                                packB_with_padding(BKsz, packed_B + BKsz * ji, &B(p, jo + ji), ldb, raw_BNsz - ji);
                            }
                            else {
                                packB(BKsz, packed_B + BKsz * ji, &B(p, jo + ji), ldb);
                            }
                        }
                        if(raw_BMsz - ii < KERNEL_M || raw_BNsz - ji < KERNEL_N) {
                            gemm_kernel_4x24_with_padding(BKsz, packed_A + BKsz * ii, lda,
                                                          packed_B + BKsz * ji, ldb,
                                                          &C(io + ii, jo + ji), ldc,
                                                          MIN(raw_BMsz - ii, KERNEL_M),
                                                          MIN(raw_BNsz - ji, KERNEL_N));
                        }
                        else {
                            gemm_kernel_4x24(BKsz, packed_A + BKsz * ii, lda,
                                             packed_B + BKsz * ji, ldb,
                                             &C(io + ii, jo + ji), ldc);
                        }
                    }
                }
                //if(p == k - 1) output_matrix_tofile("packB.csv", BKsz, BNsz, packed_B);
                aligned_free(packed_B);
            }
            //if(p == k - 1) output_matrix_tofile("packA.csv", BMsz, BKsz, packed_A);
            aligned_free(packed_A);
        }
    }
}
