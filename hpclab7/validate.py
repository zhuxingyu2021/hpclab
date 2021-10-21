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

input = read_tensor_file("input.txt")
filter = read_tensor_file("filter.txt")
output_naive = read_tensor_file("output_img_naive.txt")
output_cudnn = read_tensor_file("output_img_cudnn.txt")

diff = output_cudnn - output_naive
if np.sum(diff) > 1e-2:
    print("wrong!")
else:
    print("ok!")
