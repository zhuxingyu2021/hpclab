#include "cuconv.h"
#include "cmdline.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    cmdline::parser cmdparser;
    cmdparser.add<int>("H", 'H', "input image height", true, 512, cmdline::range(1, 65536));
    cmdparser.add<int>("W", 'W', "input image width", true, 512, cmdline::range(1, 65536));
    cmdparser.add<int>("stride", 's', "stride", true, 1, cmdline::range(1, 3));
    cmdparser.add<int>("Multiple-runs", 'm', "enable multiple runs",
        false, 0, cmdline::oneof(0, 1));
    cmdparser.add<int>("run-times", 't', "kernel run times",
        false, 1, cmdline::range(0, 3600));
    cmdparser.add<int>("Easy", 'e', "easier cmdline output",
        false, 0, cmdline::oneof(0, 1));
    cmdparser.add<int>("output", 'o', "output to file",
        false, 0, cmdline::oneof(0, 1));
    cmdparser.parse_check(argc, argv);

    cuconv_descriptor conv_parameters;
    conv_parameters.input_h = cmdparser.get<int>("H");
    conv_parameters.input_w = cmdparser.get<int>("W");
    conv_parameters.stride_h = cmdparser.get<int>("stride");
    conv_parameters.stride_w = cmdparser.get<int>("stride");
    conv_parameters.filter_h = 3;
    conv_parameters.filter_w = 3;
    conv_parameters.pad_h = 1;
    conv_parameters.pad_w = 1;
    conv_parameters.input_c = 3;
    conv_parameters.output_c = 3;

    int multiple_runs = cmdparser.get<int>("Multiple-runs");
    int run_times = 1;
    if (multiple_runs) run_times = cmdparser.get<int>("run-times");
    int easy_cmd = cmdparser.get<int>("Easy");
    int output_to_file = cmdparser.get<int>("output");

    float* input_img = (float*)malloc(sizeof(float) * 
                        conv_parameters.N_batch *
                        conv_parameters.input_c *
                        conv_parameters.input_h *
                        conv_parameters.input_w);
    
    float* filter = (float*)malloc(sizeof(float) *
                    conv_parameters.output_c *
                    conv_parameters.input_c *
                    conv_parameters.filter_h *
                    conv_parameters.filter_w);
    
    random_initialize(input_img, 
                    conv_parameters.N_batch *
                    conv_parameters.input_c *
                    conv_parameters.input_h *
                    conv_parameters.input_w);

    random_initialize(filter,
                    conv_parameters.output_c *
                    conv_parameters.input_c *
                    conv_parameters.filter_h *
                    conv_parameters.filter_w);

    if (output_to_file) {
        output_tensor_dim4_to_file("input.tensor", input_img, conv_parameters.N_batch,
            conv_parameters.input_c, conv_parameters.input_h, conv_parameters.input_w);
        output_tensor_dim4_to_file("filter.tensor", filter, conv_parameters.output_c,
            conv_parameters.input_c, conv_parameters.filter_h, conv_parameters.filter_w);
    }

    cuconv_getoutputsize(&conv_parameters);
    float* output_img_naive = (float*)malloc(sizeof(float) *
        conv_parameters.N_batch *
        conv_parameters.output_c *
        conv_parameters.output_h *
        conv_parameters.output_w);
    memset(output_img_naive, 0, sizeof(float) *
            conv_parameters.N_batch *
            conv_parameters.output_c *
            conv_parameters.output_h *
            conv_parameters.output_w);

    float* output_img_cudnn = (float*)malloc(sizeof(float) *
        conv_parameters.N_batch *
        conv_parameters.output_c *
        conv_parameters.output_h *
        conv_parameters.output_w);
    memset(output_img_cudnn, 0, sizeof(float) *
        conv_parameters.N_batch *
        conv_parameters.output_c *
        conv_parameters.output_h *
        conv_parameters.output_w);

    float* output_img_im2col = (float*)malloc(sizeof(float) *
        conv_parameters.N_batch *
        conv_parameters.output_c *
        conv_parameters.output_h *
        conv_parameters.output_w);
    memset(output_img_im2col, 0, sizeof(float) *
        conv_parameters.N_batch *
        conv_parameters.output_c *
        conv_parameters.output_h *
        conv_parameters.output_w);

    float time1 = cuconv_naive(&conv_parameters, input_img, filter, output_img_naive);

    float time2 = cuconv_cudnn(&conv_parameters, input_img, filter, output_img_cudnn);

    cout << "Time cost by naive conv: " << time1 / 1000.0 << "s" << endl;
    cout << "Time cost by cudnn conv: " << time2 / 1000.0 << "s" << endl;
    
    if (output_to_file) {
        output_tensor_dim4_to_file("output_img_naive.tensor", output_img_naive, conv_parameters.N_batch,
            conv_parameters.output_c, conv_parameters.output_h, conv_parameters.output_w);
        output_tensor_dim4_to_file("output_img_cudnn.tensor", output_img_cudnn, conv_parameters.N_batch,
            conv_parameters.output_c, conv_parameters.output_h, conv_parameters.output_w);
    }

    float time3 = cuconv_im2col(&conv_parameters, input_img, filter, output_img_im2col);
    if (output_to_file) {
        output_tensor_dim4_to_file("output_img_im2col.tensor", output_img_im2col, conv_parameters.N_batch,
            conv_parameters.output_c, conv_parameters.output_h, conv_parameters.output_w);
    }
    cout << "Time cost by im2col conv: " << time3 / 1000.0 << "s" << endl;
    
    free(input_img);
    free(filter);
    free(output_img_naive);
    free(output_img_cudnn);
    free(output_img_im2col);
    
    return 0;
}

