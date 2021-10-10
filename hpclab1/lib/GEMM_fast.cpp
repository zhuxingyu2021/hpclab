#include "GEMM.h"
#include <immintrin.h>
#include <string.h>
#include <iostream>
using namespace std;

#define MIN(i,j) ((i>j)?(j):(i))

typedef union{
    __m256 v;
    float d[8];
}avx2_256v;

inline void gemm_kernel_8x8_alignedC(int k,
    float* packed_A, int lda,
    float* packed_B, int ldb,
    float* C, int ldc)
{
    __m256 c0, c1, c2, c3, c4, c5, c6, c7;
    __m256 a, b;

    float* p_packed_A = packed_A;
    float* p_packed_B = packed_B;

    c0 = _mm256_load_ps(&C(0, 0));
    c1 = _mm256_load_ps(&C(1, 0));
    c2 = _mm256_load_ps(&C(2, 0));
    c3 = _mm256_load_ps(&C(3, 0));
    c4 = _mm256_load_ps(&C(4, 0));
    c5 = _mm256_load_ps(&C(5, 0));
    c6 = _mm256_load_ps(&C(6, 0));
    c7 = _mm256_load_ps(&C(7, 0));

    for (int p = 0; p < k; p++) {
        b = _mm256_load_ps(p_packed_B);
        p_packed_B += 8;
        
        a = _mm256_set1_ps(*p_packed_A);
        c0 = _mm256_fmadd_ps(a, b, c0);

        a = _mm256_set1_ps(*(p_packed_A + 1));
        c1 = _mm256_fmadd_ps(a, b, c1);

        a = _mm256_set1_ps(*(p_packed_A + 2));
        c2 = _mm256_fmadd_ps(a, b, c2);

        a = _mm256_set1_ps(*(p_packed_A + 3));
        c3 = _mm256_fmadd_ps(a, b, c3);

        a = _mm256_set1_ps(*(p_packed_A + 4));
        c4 = _mm256_fmadd_ps(a, b, c4);

        a = _mm256_set1_ps(*(p_packed_A + 5));
        c5 = _mm256_fmadd_ps(a, b, c5);

        a = _mm256_set1_ps(*(p_packed_A + 6));
        c6 = _mm256_fmadd_ps(a, b, c6);

        a = _mm256_set1_ps(*(p_packed_A + 7));
        c7 = _mm256_fmadd_ps(a, b, c7);

        p_packed_A += 8;
    }

    _mm256_store_ps(&C(0, 0), c0);
    _mm256_store_ps(&C(1, 0), c1);
    _mm256_store_ps(&C(2, 0), c2);
    _mm256_store_ps(&C(3, 0), c3);
    _mm256_store_ps(&C(4, 0), c4);
    _mm256_store_ps(&C(5, 0), c5);
    _mm256_store_ps(&C(6, 0), c6);
    _mm256_store_ps(&C(7, 0), c7);
}

inline void gemm_kernel_8x8_woalignedC(int k,
                            float* packed_A, int lda,
                            float* packed_B, int ldb,
                            float* C, int ldc)
{
    avx2_256v c0, c1, c2, c3, c4, c5, c6, c7;
    __m256 a, b;

    float* p_packed_A = packed_A;
    float* p_packed_B = packed_B;

    c0.v = _mm256_setzero_ps();
    c1.v = _mm256_setzero_ps();
    c2.v = _mm256_setzero_ps();
    c3.v = _mm256_setzero_ps();
    c4.v = _mm256_setzero_ps();
    c5.v = _mm256_setzero_ps();
    c6.v = _mm256_setzero_ps();
    c7.v = _mm256_setzero_ps();

    for (int p = 0; p < k; p++) {
        b = _mm256_load_ps(p_packed_B);
        p_packed_B += 8;

        a = _mm256_set1_ps(*p_packed_A);
        c0.v = _mm256_fmadd_ps(a, b, c0.v);

        a = _mm256_set1_ps(*(p_packed_A + 1));
        c1.v = _mm256_fmadd_ps(a, b, c1.v);

        a = _mm256_set1_ps(*(p_packed_A + 2));
        c2.v = _mm256_fmadd_ps(a, b, c2.v);

        a = _mm256_set1_ps(*(p_packed_A + 3));
        c3.v = _mm256_fmadd_ps(a, b, c3.v);

        a = _mm256_set1_ps(*(p_packed_A + 4));
        c4.v = _mm256_fmadd_ps(a, b, c4.v);

        a = _mm256_set1_ps(*(p_packed_A + 5));
        c5.v = _mm256_fmadd_ps(a, b, c5.v);

        a = _mm256_set1_ps(*(p_packed_A + 6));
        c6.v = _mm256_fmadd_ps(a, b, c6.v);

        a = _mm256_set1_ps(*(p_packed_A + 7));
        c7.v = _mm256_fmadd_ps(a, b, c7.v);

        p_packed_A += 8;
    }

    C(0,0) += c0.d[0];  C(0,1) += c0.d[1];  C(0,2) += c0.d[2];  C(0,3) += c0.d[3];  C(0,4) += c0.d[4];  C(0,5) += c0.d[5];  C(0,6) += c0.d[6];  C(0,7) += c0.d[7];
    C(1,0) += c1.d[0];  C(1,1) += c1.d[1];  C(1,2) += c1.d[2];  C(1,3) += c1.d[3];  C(1,4) += c1.d[4];  C(1,5) += c1.d[5];  C(1,6) += c1.d[6];  C(1,7) += c1.d[7];
    C(2,0) += c2.d[0];  C(2,1) += c2.d[1];  C(2,2) += c2.d[2];  C(2,3) += c2.d[3];  C(2,4) += c2.d[4];  C(2,5) += c2.d[5];  C(2,6) += c2.d[6];  C(2,7) += c2.d[7];
    C(3,0) += c3.d[0];  C(3,1) += c3.d[1];  C(3,2) += c3.d[2];  C(3,3) += c3.d[3];  C(3,4) += c3.d[4];  C(3,5) += c3.d[5];  C(3,6) += c3.d[6];  C(3,7) += c3.d[7];
    C(4,0) += c4.d[0];  C(4,1) += c4.d[1];  C(4,2) += c4.d[2];  C(4,3) += c4.d[3];  C(4,4) += c4.d[4];  C(4,5) += c4.d[5];  C(4,6) += c4.d[6];  C(4,7) += c4.d[7];
    C(5,0) += c5.d[0];  C(5,1) += c5.d[1];  C(5,2) += c5.d[2];  C(5,3) += c5.d[3];  C(5,4) += c5.d[4];  C(5,5) += c5.d[5];  C(5,6) += c5.d[6];  C(5,7) += c5.d[7];
    C(6,0) += c6.d[0];  C(6,1) += c6.d[1];  C(6,2) += c6.d[2];  C(6,3) += c6.d[3];  C(6,4) += c6.d[4];  C(6,5) += c6.d[5];  C(6,6) += c6.d[6];  C(6,7) += c6.d[7];
    C(7,0) += c7.d[0];  C(7,1) += c7.d[1];  C(7,2) += c7.d[2];  C(7,3) += c7.d[3];  C(7,4) += c7.d[4];  C(7,5) += c7.d[5];  C(7,6) += c7.d[6];  C(7,7) += c7.d[7];

}

inline void gemm_kernel_8x8_with_padding(int k,
                                       float* packed_A, int lda,
                                       float* packed_B, int ldb,
                                       float* C, int ldc,
                                       int pm,int pn)
{
    avx2_256v c0, c1, c2, c3, c4, c5, c6, c7;
    __m256 a, b;

    float* p_packed_A = packed_A;
    float* p_packed_B = packed_B;

    c0.v = _mm256_setzero_ps();
    c1.v = _mm256_setzero_ps();
    c2.v = _mm256_setzero_ps();
    c3.v = _mm256_setzero_ps();
    c4.v = _mm256_setzero_ps();
    c5.v = _mm256_setzero_ps();
    c6.v = _mm256_setzero_ps();
    c7.v = _mm256_setzero_ps();

    for (int p = 0; p < k; p++) {
        b = _mm256_load_ps(p_packed_B);
        p_packed_B += 8;

        a = _mm256_set1_ps(*p_packed_A);
        c0.v = _mm256_fmadd_ps(a, b, c0.v);

        a = _mm256_set1_ps(*(p_packed_A + 1));
        c1.v = _mm256_fmadd_ps(a, b, c1.v);

        a = _mm256_set1_ps(*(p_packed_A + 2));
        c2.v = _mm256_fmadd_ps(a, b, c2.v);

        a = _mm256_set1_ps(*(p_packed_A + 3));
        c3.v = _mm256_fmadd_ps(a, b, c3.v);

        a = _mm256_set1_ps(*(p_packed_A + 4));
        c4.v = _mm256_fmadd_ps(a, b, c4.v);

        a = _mm256_set1_ps(*(p_packed_A + 5));
        c5.v = _mm256_fmadd_ps(a, b, c5.v);

        a = _mm256_set1_ps(*(p_packed_A + 6));
        c6.v = _mm256_fmadd_ps(a, b, c6.v);

        a = _mm256_set1_ps(*(p_packed_A + 7));
        c7.v = _mm256_fmadd_ps(a, b, c7.v);

        p_packed_A += 8;
    }

    for(int i = 0; i < pm; i++){
        avx2_256v ci;
        switch (i) {
            case 0: ci = c0; break;
            case 1: ci = c1; break;
            case 2: ci = c2; break;
            case 3: ci = c3; break;
            case 4: ci = c4; break;
            case 5: ci = c5; break;
            case 6: ci = c6; break;
            case 7: ci = c7; break;
            default: ;
        }
        for(int j = 0; j < pn; j++){
            C(i,j) += ci.d[j];
        }
    }

}

inline void packA(int k, float* packed_A, float* A, int lda) {
    float* p_packed_A = packed_A;

    float* p_a0 = &A(0, 0);
    float* p_a1 = &A(1, 0);
    float* p_a2 = &A(2, 0);
    float* p_a3 = &A(3, 0);
    float* p_a4 = &A(4, 0);
    float* p_a5 = &A(5, 0);
    float* p_a6 = &A(6, 0);
    float* p_a7 = &A(7, 0);

    for (int p = 0; p < k; p++) {
        *p_packed_A = *p_a0++;
        *(p_packed_A + 1) = *p_a1++;
        *(p_packed_A + 2) = *p_a2++;
        *(p_packed_A + 3) = *p_a3++;
        *(p_packed_A + 4) = *p_a4++;
        *(p_packed_A + 5) = *p_a5++;
        *(p_packed_A + 6) = *p_a6++;
        *(p_packed_A + 7) = *p_a7++;
        p_packed_A += 8;
    }
}

inline void packA_with_padding(int k, float* packed_A, float* A, int lda, int psa) {
    float* p_packed_A = packed_A;

    float* p_a[8];
    for(int i = 0; i < psa; i++){
        p_a[i] = &A(i, 0);
    }

    for (int p = 0; p < k; p++) {
        for(int i = 0; i < 8; i++){
            if(i >= psa) *p_packed_A = 0;
            else *p_packed_A = *p_a[i]++;
            p_packed_A++;
        }
    }
}

inline void packB(int k,float* packed_B,float* B,int ldb) {
    __m256 tmp;
    float* p_packed_B = packed_B;
    
    for (int p = 0; p < k; p++) {
        tmp = _mm256_load_ps(&B(p, 0));
        _mm256_store_ps(p_packed_B, tmp);
        p_packed_B += 8;
    }
}

inline void packB_woaligned(int k,float* packed_B,float* B,int ldb) {
    float* p_packed_B = packed_B;
    float* p_B;
    
    for (int p = 0; p < k; p++) {
        p_B = &B(p, 0);

        *p_packed_B = *p_B;
        *(p_packed_B + 1) = *(p_B + 1);
        *(p_packed_B + 2) = *(p_B + 2);
        *(p_packed_B + 3) = *(p_B + 3);
        *(p_packed_B + 4) = *(p_B + 4);
        *(p_packed_B + 5) = *(p_B + 5);
        *(p_packed_B + 6) = *(p_B + 6);
        *(p_packed_B + 7) = *(p_B + 7);

        p_packed_B += 8;
    }
}

inline void packB_with_padding(int k,float* packed_B,float* B,int ldb,int psb) {
    float* p_packed_B = packed_B;
    float* p_B;

    for (int p = 0; p < k; p++) {
        p_B = &B(p, 0);

        for(int i = 0; i < 8; i++){
            if(i >= psb) *p_packed_B= 0;
            else *p_packed_B = *(p_B + i);
            p_packed_B++;
        }
    }
}


inline void gemm_fast_multiply_aligned(int k, int m, int n,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
    int io, ii, jo, ji, p;

    for (io = 0; io < m; io += BM) {
        int BMsz = MIN(m - io, BM);
        for (p = 0; p < k; p += BK) {
            int BKsz = MIN(k - p, BK);
            float* packed_A = (float*)aligned_malloc(sizeof(float) * BMsz * BKsz, GEMM_CACHELINE_SIZE);
            for (jo = 0; jo < n; jo += BN) {
                int BNsz = MIN(n - jo, BN);
                float* packed_B = (float*)aligned_malloc(sizeof(float) * BKsz * BNsz, GEMM_CACHELINE_SIZE);
                for (ii = 0; ii < BMsz; ii += 8) {
                    if (jo == 0) packA(BKsz, packed_A + BKsz * ii, &A(io + ii, p), lda);
                    for (ji = 0; ji < BNsz; ji += 8) {
                        if (ii == 0) packB(BKsz, packed_B + BKsz * ji, &B(p, jo + ji), ldb);
                        gemm_kernel_8x8_alignedC(BKsz, packed_A + BKsz * ii, lda, packed_B + BKsz * ji, ldb, &C(io + ii, jo + ji), ldc);
                    }
                }
                aligned_free(packed_B);
            }
            aligned_free(packed_A);
        }
    }
}

inline void gemm_fast_multiply_woaligned(int k, int m, int n,
                                       float* A, int lda,
                                       float* B, int ldb,
                                       float* C, int ldc,
                                       bool aligned_B,
                                       bool aligned_C)
{
    int io, ii, jo, ji, p;

    for (io = 0; io < m; io += BM) {
        int BMsz = MIN(m - io, BM);
        for (p = 0; p < k; p += BK) {
            int BKsz = MIN(k - p, BK);
            float* packed_A = (float*)aligned_malloc(sizeof(float) * BMsz * BKsz, GEMM_CACHELINE_SIZE);
            for (jo = 0; jo < n; jo += BN) {
                int BNsz = MIN(n - jo, BN);
                float* packed_B = (float*)aligned_malloc(sizeof(float) * BKsz * BNsz, GEMM_CACHELINE_SIZE);
                for (ii = 0; ii < BMsz; ii += 8) {
                    if (jo == 0) packA(BKsz, packed_A + BKsz * ii, &A(io + ii, p), lda);
                    for (ji = 0; ji < BNsz; ji += 8) {
                        if (ii == 0){
                            if(!aligned_B) packB_woaligned(BKsz, packed_B + BKsz * ji, &B(p, jo + ji), ldb);
                            else packB(BKsz, packed_B + BKsz * ji, &B(p, jo + ji), ldb);
                        }
                        if(aligned_C) gemm_kernel_8x8_alignedC(BKsz, packed_A + BKsz * ii, lda, packed_B + BKsz * ji, ldb, &C(io + ii, jo + ji), ldc);
                        else gemm_kernel_8x8_woalignedC(BKsz, packed_A + BKsz * ii, lda, packed_B + BKsz * ji, ldb, &C(io + ii, jo + ji), ldc);
                    }
                }
                aligned_free(packed_B);
            }
            aligned_free(packed_A);
        }
    }
}

inline void gemm_fast_multiply_wo_factor8(int k, int m, int n,
                                         float* A, int lda,
                                         float* B, int ldb,
                                         float* C, int ldc)
{
    int io, ii, jo, ji, p;

    for (io = 0; io < m; io += BM) {
        int BMsz = MIN(m - io, BM);
        int raw_BMsz = BMsz;
        if(BMsz % 8 != 0)  BMsz = (BMsz / 8) * 8 + 8;
        for (p = 0; p < k; p += BK) {
            int BKsz = MIN(k - p, BK);
            float* packed_A = (float*)aligned_malloc(sizeof(float) * BMsz * BKsz, GEMM_CACHELINE_SIZE);
            for (jo = 0; jo < n; jo += BN) {
                int BNsz = MIN(n - jo, BN);
                int raw_BNsz = BNsz;
                if(BNsz % 8 != 0)  BNsz = (BNsz / 8) * 8 + 8;
                float* packed_B = (float*)aligned_malloc(sizeof(float) * BKsz * BNsz, GEMM_CACHELINE_SIZE);
                for (ii = 0; ii < BMsz; ii += 8) {
                    if (jo == 0) {
                        if(raw_BMsz - ii < 8) {
                            packA_with_padding(BKsz, packed_A + BKsz * ii, &A(io + ii, p), lda, raw_BMsz - ii);
                        }
                        else {
                            packA(BKsz, packed_A + BKsz * ii, &A(io + ii, p), lda);
                        }
                    }
                    for (ji = 0; ji < BNsz; ji += 8) {
                        if (ii == 0){
                            if(raw_BNsz - ji < 8) packB_with_padding(BKsz, packed_B + BKsz * ji, &B(p, jo + ji), ldb, raw_BNsz - ji);
                            else packB_woaligned(BKsz, packed_B + BKsz * ji, &B(p, jo + ji), ldb);
                        }
                        if(raw_BMsz - ii < 8 || raw_BNsz - ji < 8) gemm_kernel_8x8_with_padding(BKsz, packed_A + BKsz * ii, lda,
                                                                                                packed_B + BKsz * ji, ldb, &C(io + ii, jo + ji), ldc,
                                                                                                MIN(raw_BMsz - ii, 8), MIN(raw_BNsz - ji, 8));
                        else gemm_kernel_8x8_woalignedC(BKsz, packed_A + BKsz * ii, lda, packed_B + BKsz * ji, ldb, &C(io + ii, jo + ji), ldc);
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

void sgemm_fast(int k, int m, int n,
                float* A, int lda,
                float* B, int ldb,
                float* C, int ldc) {
    if ((m % 8 == 0) && (n % 8 == 0)) {
        bool aligned_B = true, aligned_C = true;

        if((size_t)B % GEMM_AVX2ALIGN_SIZE) aligned_B = false;
        if((size_t)C % GEMM_AVX2ALIGN_SIZE) aligned_C = false;

        if(aligned_B && aligned_C) gemm_fast_multiply_aligned(k, m, n, A, lda, B, ldb, C, ldc);
        else gemm_fast_multiply_woaligned(k, m, n, A, lda, B, ldb, C, ldc, aligned_B, aligned_C);
    }
    else {
        gemm_fast_multiply_wo_factor8(k, m, n, A, lda, B, ldb, C, ldc);
    }
}
