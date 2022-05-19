import numpy as np

def read_tensor_file(filename):
    with open(filename, "r") as f:
        dim1, dim2 ,dim3 ,dim4 = [int(dim) for dim in f.readline().split(' ')]
        tensor = []
        for d in f.readline().split(' '):
            if d != '\n':
                tensor.append(float(d))
        assert len(tensor) == dim1 * dim2 * dim3 * dim4
        tensor = np.array(tensor).reshape(dim1, dim2 ,dim3 ,dim4)
        return tensor

input = read_tensor_file("input.tensor")
filter = read_tensor_file("filter.tensor")
output_naive = read_tensor_file("output_img_naive.tensor")
output_cudnn = read_tensor_file("output_img_cudnn.tensor")
output_im2col = read_tensor_file("output_img_im2col.tensor")

diff1 = output_cudnn - output_naive
if np.sum(diff1) > 1e-2:
    print("wrong!")
else:
    print("ok!")

diff2 = output_im2col - output_naive
if np.sum(diff1) > 1e-2:
    print("wrong!")
else:
    print("ok!")
