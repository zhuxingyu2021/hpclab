#include "cuconv.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

void cuconv_getoutputsize(cuconv_descriptor* d)
{
    d->output_h = (d->input_h + d->pad_h * 2 - d->filter_h)/d->stride_h + 1;
    d->output_w = (d->input_w + d->pad_w * 2 - d->filter_w)/d->stride_w + 1;
}

void random_initialize(float* tensor, int size)
{
    for(int i = 0; i < size; i++)
    {
        tensor[i] = rand() / float(RAND_MAX);
    }
}

void output_tensor_dim4_to_file(const char* filename, float* tensor, int dim1, int dim2, int dim3, int dim4)
{
    int tensorsize = dim1 * dim2 * dim3 * dim4;
    ofstream outfile(filename, ios::trunc);

    if (!outfile.is_open()) {
        cerr << "File Open Error!" << endl;
        exit(-1);
    }

    outfile << dim1 << " " << dim2 << " " << dim3 << " " << dim4 << endl;

    for (int i = 0; i < tensorsize; i++) {
        outfile << tensor[i] << " ";
    }
    outfile << endl;

    outfile.close();
}
