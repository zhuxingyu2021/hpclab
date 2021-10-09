#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import operator


# In[2]:


outfilepath = "build-release/output.txt"


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
        time_dicts[d_count]["Naive_t"] = float(fp.readline())
        fp.readline()
        first3 = [int(tmp) for tmp in fp.readline().split()]
        d_count += 1

time_dicts = sorted(time_dicts, key=operator.itemgetter('MNK'))


# In[5]:


MNK = np.zeros(len(time_dicts))
MKL_gflop = np.zeros(len(time_dicts))
My_gflop = np.zeros(len(time_dicts))
Naive_gflop = np.zeros(len(time_dicts))
for i in range(len(time_dicts)):
    MNK[i] = time_dicts[i]["MNK"]
    flop = MNK[i]*MNK[i]*MNK[i]*2
    Gflop = flop/1000000000.0
    MKL_gflop[i] = Gflop/time_dicts[i]["MKL_t"]
    My_gflop[i] = Gflop/time_dicts[i]["My_t"]
    Naive_gflop[i] = Gflop/time_dicts[i]["Naive_t"]


# In[6]:


plt.figure(figsize=(15,12))
plt.plot(MNK, MKL_gflop, color='red', label='MKL')
plt.plot(MNK, My_gflop, color='g', label='My Optimization')
plt.plot(MNK, Naive_gflop, color='b', label='Naive 3-level loops')
x_major_locator=plt.MultipleLocator(256)
y_major_locator=plt.MultipleLocator(5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim(0, 2048)
plt.ylim(0, 100)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Martix Size(M, N, K)", fontsize=20)
plt.ylabel("Gflops", fontsize=20)
plt.legend(fontsize=20)
plt.savefig("hpclab1.png")
plt.show()

