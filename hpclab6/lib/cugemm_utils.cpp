#include "cugemm.h"
#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;

void random_initialize_matrix(int M, int N, float* mat) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            mat[i * N + j] = rand() / float(RAND_MAX);
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

