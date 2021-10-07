#include "GEMM.h"
#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

void random_initalize_matrix(int M, int N, float* mat) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            //mat[i * N + j] = rand() / float(RAND_MAX);
            mat[i * N + j] = float(rand() % 2);
        }
    }
}

void output_matrix_tofile(const char* filename, int M, int N, float* mat) {
    ofstream outfile(filename, ios::trunc);

    if (!outfile.is_open()) {
        cerr << "File Open Error!" << endl;
        exit(-1);
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            outfile << mat[i * N + j];
            if (j != N - 1)
                outfile << ",";
        }
        outfile << endl;
    }

    outfile.close();
}


bool verify_matrix(int M, int N, float* mat1, float* mat2) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (abs(mat1[i * N + j] - mat2[i * N + j]) > TOLERENCE) {
                return false;
            }
        }
    }
    return true;
}

void debug_print_matrix(int M, int N, float* mat) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << mat[i * N + j] << " ";
        }
        cout << endl;
    }
}

void* aligned_malloc(size_t required_bytes, size_t alignment)
{
    int offset = alignment - 1 + sizeof(void*);
    void* p1 = (void*)malloc(required_bytes + offset);

    if (p1 == NULL)
        return NULL;

    void** p2 = (void**)(((size_t)p1 + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

void aligned_free(void* p2)
{
    void* p1 = ((void**)p2)[-1];
    free(p1);
}
