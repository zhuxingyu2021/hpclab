#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import operator


# In[2]:


outfilepath = "test.tmp/output.txt"


# In[3]:


time_dicts=[]
d_count=0


# In[4]:


with open(outfilepath, "r") as fp:
    first3 = [int(tmp) for tmp in fp.readline().split()]
    while(len(first3) == 3):
        M, N, K = first3
        assert M == N
        assert N == K
        time_dicts.append({})
        time_dicts[d_count]["MNK"] = M
        time_dicts[d_count]["MKL_t"] = float(fp.readline())
        time_dicts[d_count]["My_t"] = float(fp.readline())
        fp.readline()
        first3 = [int(tmp) for tmp in fp.readline().split()]
        d_count += 1

time_dicts = sorted(time_dicts, key=operator.itemgetter('MNK'))


# In[5]:


MNK = np.zeros(len(time_dicts))
MKL_tflop = np.zeros(len(time_dicts))
My_tflop = np.zeros(len(time_dicts))
Naive_tflop = np.zeros(len(time_dicts))
for i in range(len(time_dicts)):
    MNK[i] = time_dicts[i]["MNK"]
    flop = MNK[i]*MNK[i]*MNK[i]*2
    Tflop = flop/1000000000000.0
    MKL_tflop[i] = Tflop/time_dicts[i]["MKL_t"]
    My_tflop[i] = Tflop/time_dicts[i]["My_t"]


# In[6]:


plt.figure(figsize=(15,12))
plt.plot(MNK, MKL_tflop, color='red', label='cublas')
plt.plot(MNK, My_tflop, color='g', label='My Optimization')
x_major_locator=plt.MultipleLocator(2048)
y_major_locator=plt.MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim(512, 16384)
plt.ylim(0, 16)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Martix Size(M, N, K)", fontsize=20)
plt.ylabel("Tflops", fontsize=20)
plt.legend(fontsize=20, loc=7)
plt.savefig("hpclab6.png")
plt.show()

