from torch import nn
import torch
import numpy as np
import math
import FunP1 as Fu
import datareshape



def my_ob(V,H,Q,I,M,N,batch_size,p_max,R_min,a,b,device):


    result = (V[:, 0:M, :, :]) + 1j * (V[:, M:2 * M, :, :])
    result1 = result.permute(0, 3, 2, 1)

    result2 = torch.reshape(result1, (batch_size, I, Q * M))
    objective=0
    n_zero=0
    power_batch=0
    for i  in range(batch_size):
        H_batchsize=H[i*N:i*N+N,:]
        V_batchsize= result2[i,:,:].permute(1,0)


        V_batchsize_1=np.zeros((Q*M,I),dtype='complex_')
        V_batchsize_1=torch.tensor(V_batchsize_1)
        for ii in range (M):
            for jj in range(Q):
                V_batchsize_1[ii*Q+jj,:]=V_batchsize[M*jj+ii,:]

        b=0
        for iii in range (Q*M):
            for jjj in range (I):
                if torch.abs(V_batchsize_1[iii,jjj]) == 0:

                    b+=1

        n=b/(I*Q*M)


        n_zero=n_zero+n
        power=torch.zeros(1)
        for jjjj in range(Q):
            sum1=torch.zeros(1)

            for iiii in range(I):
                v=V_batchsize_1[jjjj * M:jjjj * M + M, iiii:iiii+1]
                v1=v.real-1j*v.imag
                v2=v1.permute(1,0)
                sum1=sum1+torch.matmul(v2, V_batchsize_1[jjjj*M:jjjj*M+M,iiii])

            power+=torch.abs(sum1)-p_max


        uf,V_F=Fu.uF(H_batchsize, V_batchsize_1, I, N, Q, M,device)
        power_batch+=power
        objective=objective+uf

    return (objective)/batch_size,n_zero/batch_size


