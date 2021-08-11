import numpy as np
import math

from matplotlib import pyplot as plt

def filter(x,M=5,method='gauss'):
    # para:x, input vector, numpy
    #      M, size of filter kernel
    #      method,filter method 
    pad_width = int((M-1)/2)
    if method == 'mean':
        # print('mean')
        h=np.asarray([1/M for i in range(M)])
    elif method == 'gauss':
        sigema=M/4
        t=[i-M/2 for i in range(M)]
        h=[]
        for i in range(M):
            h.append(1/((2*math.pi)**0.5*sigema)*math.exp(-(t[i]**2/2/(sigema**2))))
        h=np.asarray(h)/np.sum(h)
        # print('gauss')
    # elif method == 'median':

    else:
        h=np.asarray([1/M for i in range(M)])
    # symmetric，reflect...
    y=np.pad(x, pad_width, mode='symmetric')
    # ‘valid’，The convolution product is only 
    # given for points where the signals overlap completely
    result=np.convolve(y, h,'valid')

    if method == 'median':
        result = np.ones_like(x)
        for i in range(len(result)):
            result[i] = np.median(y[i:i+M])
            
    return result



# x=[]
# for i in range(2000):
#     x.append(0.1*math.sin(i/20*math.pi)+5+math.sin(i/500*math.pi))
# t=[i/5 for i in range(2000)]
# x=np.asarray(x)
# y=filter(x,51,'gauss')


# ##
# plt.figure(figsize=(10,6))
# ax0 = plt.subplot(211)             
# ax0.plot(t,x)

# ax0 = plt.subplot(212)            
# ax0.plot(t,y)

# plt.show()