from scipy.io import loadmat
import numpy as np


data =loadmat("")
print(data.keys())


data_1 = data['']
data_shape=np.shape(data_1)


cycle_index=data_shape[0]
Q=16
I=16
M=4
N=2

list_data=list()
list_data_original=list()

count=0
for id in range(int(cycle_index/N)):
    data_2=data_1[id*N:id*N+N,:]
    for i in range(I):
        for q in range(Q):
            data_3=data_2[:,i*Q*M+q*M+0:i*Q*M+q*M+M]
            data_4=np.reshape(data_3,(1,1,M*N))
            list_data.append(data_4)

        data_5=data_2[:,i*Q*M+0:i*Q*M+Q*M]
        list_data_original.append(data_5)


list_data_1=np.reshape(list_data,(int(cycle_index/N),I,Q,M*N))

list_data_3=np.reshape(list_data_original,(int(cycle_index/N),I,N,M*Q))
print(np.shape(list_data_1))

